# -----------------------------------------------------------------------------
# Title: Improved Denoising Diffusion Probabilistic Models Adaptation
# Description: 
#     This code is adapted from the official Improved Denoising Diffusion Probabilistic Models (IDDPM) 
#     repository by OpenAI. It incorporates modifications and enhancements tailored 
#     to seismic wavefield representation.
# Original Official Repository: https://github.com/openai/improved-diffusion
# Reference Paper: https://arxiv.org/abs/2102.09672
# Author: Shijun Cheng
# -----------------------------------------------------------------------------

import enum
import math
import numpy as np
import torch as th
import torch.nn.functional as F
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from .datasets import denormalizer_vel

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        pde_guide=False,
        fd_order=4,
    ):
        # The predicted output can be noise (DDPM), the mean of x(t-1), or the predicted x0
        self.model_mean_type = model_mean_type
        # There are two main types of model variance: 1. Learnable; 2. Fixed
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        # self.alphas_cumprod denote a(t)_bar
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        # self.alphas_cumprod_prev denote a(t-1)_bar
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # self.alphas_cumprod_next denote a(t+1)_bar
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # Calculate the posterior variance (corresponding to Equation 10 in the IDDPM paper)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
       # log calculation clipped because the posterior variance is 0 at the
       # beginning of the diffusion chain.
       # Here take the logarithm of self.posterior_variance, remove the first term, and then take the first term as 1
       # The reason for doing this here is to prevent self.posterior_variance from getting a number close to 0, which would make the log calculation unstable.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
         # Corresponding to the two coefficients in the posterior mean Equation in Equation 11 in the IDDPM paper
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.pde_guide = pde_guide
        self.fd_order = fd_order
        self.laplace_kernel = laplace_operator(order=fd_order)

        self.inversion_criterion = th.nn.MSELoss()

    # Corresponding to Equation 9 in the IDDPM paper, the mean and variance can be calculated based on x_0 and t
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    # Corresponding to Equation 9 in the IDDPM paper, given x0 and t, x_t can be sampled
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    # Corresponding to Equations 10 and 11 in the IDDPM paper, we can calculate the mean and variance of the posterior distribution x_(t-1) based on x_0, x_t and t
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        # The mean of the posterior
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # The variance of the posterior
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)

        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # Get the variance of the previous mean, that is, get the mean and variance of x_(t-1) from x_t
    # In the q_posterior_mean_variance(self, x_start, x_t, t) function, we can calculate the mean and variance of x_(t-1) based on x_0, x_t and t
    # However, we don't know x_0, and this x_0 needs to be predicted by the network
    # The parameter x passed in here represents x_t, we need to get the mean and variance of x_(t-1), and predict the initial x_0
    def p_mean_variance(
        self, model, x, input, t, vel, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param input: the [N x C x ...] tensor of input.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, input, self._scale_timesteps(t), vel, **model_kwargs)

        '''
         From here, we can calculate the variance of the posterior distribution in two ways:
         1. The variance is learnable
         2. The variance is fixed
        '''
        # The if statement here indicates that the logarithmic variance of the model is learnable
        # There are two types of learnable methods, one is the direct prediction method (ModelVarType.LEARNED)
        # The other is the range of the learning method (ModelVarType.LEARNED_RANGE)
        # The method of predicting the range is in Equation 15 in the IDDPM paper, where the contrast is expressed as the contrast of beta multiplied by the coefficient v and 1-v respectively, and then an exponential function exp is taken
        # Here v is the learning parameter, which has a range
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            # Under this condition, directly predict the variance and assign the predicted logarithmic variance to model_log_variance
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                # Under this condition, directly predict the variance and assign the predicted logarithmic variance to model_log_variance
                model_log_variance = model_var_values
                # Take an exponential directly on the predicted logarithmic variance to get the variance model_variance
                model_variance = th.exp(model_log_variance)
            else:
                # Under this condition, the range of the predicted variance (Equation 15 in the IDDPM paper) is between [-1, 1]
                # Equation 15 contains log(beta) and log(beta_bar)
                # beta_bar is given in Equation 10, since 1-alpha_(t-1)_bar is less than 1-alpha_(t)_bar
                # Therefore, beta_bar is less than beta, so min_log here represents log(beta_bar) in Equation 15
                # max_log represents log(beta) in Equation 15
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2  # Convert the range of [-1,1] to [0,1]
                model_log_variance = frac * max_log + (1 - frac) * min_log
                # Take an exponential directly on the predicted logarithmic variance to get the variance model_variance
                model_variance = th.exp(model_log_variance)

        # The else judgment here indicates that the variance of the model is not learnable
        # In the DDPM article, the variance is directly beta
        # In the IDDPM article, it is said that two types can be used, one is beta and the other is beta_bar
        # Here ModelVarType.FIXED_LARGE means using beta
        # Here ModelVarType.FIXED_SMALL means using beta_bar
        else:
            # What is returned here is the model_variance and model_log_variance of all moments
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                # Note that in the case of ModelVarType.FIXED_LARGE,
                # The author mentioned that it is better to use self.posterior_variance[1] for the variance at the initial moment,
                # Therefore, the initial variance here uses self.posterior_variance[1]
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
             # Select model_variance and model_log_variance from model_variance and model_log_variance according to the passed time t
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        # A function for processing prediction results
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        '''
          From here, we can calculate the mean of the posterior distribution in three ways:
          1. Predict noise (the method used by DDPM)
          2. Directly predict the mean
          3. Predict x_0
        '''
        # ModelMeanType.PREVIOUS_X means directly predicting the mean, so the model output model_output directly contains the mean
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            # Here we also calculate a pred_xstart, which means to calculate x_0 from the predicted mean
            # According to Equation (11), we have calculated the mean u_t and know x_t, so we can use this formula to calculate x_0
            # Here, the calculated x_0 has no effect on the network training process, but it is useful for the evaluation stage
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        # [ModelMeanType.START_X, ModelMeanType.EPSILON]
        # This means that we directly predict x_0 or noise
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            # Indicates direct prediction of x_0
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            # It means directly predicting the noise. Here, we need to further calculate x_0 through the predicted noise. The formula used is Equation (9) in IDDPM paper
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            # According to the calculated x0, the mean can be further calculated by relying on Equation (11):
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        # Finally, the model outputs the mean and variance of the previous moment. pred_xstart represents the initial value x0.
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    # Calculate x_0 from the predicted noise, which is Equation 9 in the IDDPM paper
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    # Equation 11, calculates x_0 from the mean xprev of x_(t-1) predicted by the network and the distribution x_t at time t
    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    # From the predicted x_0 and x_t, we can infer the noise added from x_0 to x_t.
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def pde_loss(self, x, input, dh, omega, m, m0):

        du_real_laplace = 1/(dh ** 2) * F.conv2d(x[:, 0:1], self.laplace_kernel, padding=self.fd_order//2, groups=1)
        du_imag_laplace = 1/(dh ** 2) * F.conv2d(x[:, 1:2], self.laplace_kernel, padding=self.fd_order//2, groups=1)

        loss_real = omega*omega*m*x[:, 0:1] + du_real_laplace + omega*omega*(m-m0)*input[:, 0:1]
        loss_imag = omega*omega*m*x[:, 1:2] + du_imag_laplace + omega*omega*(m-m0)*input[:, 1:2]
        pde_loss = th.sqrt((th.pow(loss_real,2)).mean() + (th.pow(loss_imag,2)).mean())

        return pde_loss

    def pde_guidance(self, x, input, dh, omega, m, m0, weight_t, scale_factor=1.0):
        with th.enable_grad():
            x.requires_grad_(True)
            du_real_laplace = 1/(dh ** 2) * F.conv2d(x[:, 0:1], self.laplace_kernel, padding=self.fd_order//2, groups=1)
            du_imag_laplace = 1/(dh ** 2) * F.conv2d(x[:, 1:2], self.laplace_kernel, padding=self.fd_order//2, groups=1)

            loss_real = omega*omega*m*x[:, 0:1] + du_real_laplace + omega*omega*(m-m0)*input[:, 0:1]
            loss_imag = omega*omega*m*x[:, 1:2] + du_imag_laplace + omega*omega*(m-m0)*input[:, 1:2]

            pde_loss = 0.001*th.sqrt((th.pow(loss_real,2)) + (th.pow(loss_imag,2))) #th.cat((loss_real, loss_imag), dim=1)
            # du_laplace = 1/(dh ** 2) * F.conv2d(x, self.laplace_kernel, padding=self.fd_order//2, groups=2)
            # pde_loss = omega*omega*m*x + du_real_laplace + omega*omega*(m-m0)*input
            grad = th.autograd.grad(th.sqrt((th.pow(loss_real,2)).mean() + (th.pow(loss_imag,2)).mean()), x, retain_graph=True)[0]
            grad_max = th.max(th.abs(grad)).item()
            grad = grad / grad_max

        return  2 * pde_loss * grad * scale_factor, th.sqrt((th.pow(loss_real,2)).mean() + (th.pow(loss_imag,2)).mean()).item()

    # Sample x_(t-1) from x_t, single-step recovery
    def p_sample(
        self, model, x, input, t, vel, dh, omega, m, m0, scale_factor=1.0, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param input: the tensor of input.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        # Get the mean, variance, logarithmic variance and predicted value of x_0 at time x_(t-1)
        out = self.p_mean_variance(
            model,
            x,
            input,
            t,
            vel,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
         # Mask matrix for non-zero moments
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        # th.exp(0.5 * out["log_variance"]) represents the standard deviation, because the variance of the logarithm is multiplied by 0.5 and then the exponent exp is equal to sqrt(variance)
        # The standard deviation is calculated because the coefficient multiplied by the noise is the standard deviation, not the variance. The square root of the variance is required to equal the standard deviation.
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        pde_loss_before = self.pde_loss(sample, input, dh, omega, m, m0)
        pde_loss_after = 0.0
        if self.pde_guide:
            # if t.item() < 100:
                #### first method
                # cond_grad, pde_loss = self.pde_guidance(x, input, dh, omega, m, m0, out["variance"], scale_factor=scale_factor)
                # sample = mean_pred + cond_grad + nonzero_mask * sigma * noise

            #### second method
            cond_grad, _ = self.pde_guidance(sample, input, dh, omega, m, m0, out["variance"], scale_factor=scale_factor)
            sample = sample - cond_grad
            pde_loss_after = self.pde_loss(sample, input, dh, omega, m, m0)

        if (t[0].item() + 1) % 100 == 0 or t[0].item() == 0:
            print(f'Time step {t[0].item()} --> PDE guider before Loss {pde_loss_before.item()} and after Loss {pde_loss_after.item()}')

        return {"sample": sample, "pred_xstart": out["pred_xstart"], "pde_loss_before": pde_loss_before.item(), "pde_loss_after": pde_loss_after.item()}

    # Call p_sample, loop
    def p_sample_loop(
        self,
        model,
        input,
        vel,
        shape,
        dh,
        omega,
        scale_factor=1.0, 
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param input: the input.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample, image_all, pred_xstart, pde_loss_before, pde_loss_after in self.p_sample_loop_progressive(
            model,
            input,
            vel,
            shape,
            dh,
            omega,
            scale_factor=scale_factor, 
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final, image_all, pred_xstart, pde_loss_before, pde_loss_after = sample, image_all, pred_xstart, pde_loss_before, pde_loss_after
        return final["sample"], image_all, pred_xstart, pde_loss_before, pde_loss_after

    def p_sample_loop_progressive(
        self,
        model,
        input,
        vel,
        shape,
        dh,
        omega,
        scale_factor=1.0, 
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image = noise
        else:
            image = th.randn(*shape, device=device)

        # Indexing time in reverse order
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        denorm_vel = denormalizer_vel(vel)
        m = 1 / (denorm_vel**2)
        m0 = th.full_like(denorm_vel, 1 / (denorm_vel.min()**2))

        image_all = []
        pred_xstart = []
        pde_loss_before = []
        pde_loss_after = []
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    image,
                    input,
                    t,
                    vel,
                    dh,
                    omega,
                    m=m,
                    m0=m0,
                    scale_factor=scale_factor, 
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if (i+1) % 50 == 0:
                    image_all.append(out["sample"])
                    pred_xstart.append(out["pred_xstart"])
                pde_loss_before.append(out["pred_loss_before"])
                pde_loss_after.append(out["pred_loss_after"])
                yield out, image_all, pred_xstart, pde_loss_before, pde_loss_after
                image = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        input,
        t,
        vel,
        dh,
        omega,
        m,
        m0,
        scale_factor=1.0, 
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        # The self.p_mean_variance function is used to get the mean and variance of the previous moment
        out = self.p_mean_variance(
            model,
            x,
            input,
            t,
            vel,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        # nonzero_mask means that when t is not equal to 0, we need sigma * noise
        # nonzero_mask means that when t is equal to 0, we remove the term sigma * noise
        # That is, in the last step, we directly output the mean and no longer need the term variance
        sample = mean_pred + nonzero_mask * sigma * noise
        pde_loss_before = self.pde_loss(sample, input, dh, omega, m, m0).item()
        pde_loss_after = 0.0
        if self.pde_guide:
            # if t.item() < 800:
                #### first method
                # cond_grad, pde_loss = self.pde_guidance(x, input, dh, omega, m, m0, out["variance"], scale_factor=scale_factor)
                # sample = mean_pred + cond_grad + nonzero_mask * sigma * noise

            #### second method
            cond_grad, _ = self.pde_guidance(sample, input, dh, omega, m, m0, out["variance"], scale_factor=scale_factor)
            sample = sample - cond_grad
            pde_loss_after = self.pde_loss(sample, input, dh, omega, m, m0).item()

        # if (t[0].item() + 1) % 50 == 0 or t[0].item() == 0:
            # print(f'Time step {t[0].item()} --> PDE guider before Loss {pde_loss_before} and after Loss {pde_loss_after}')

        return {"sample": sample, "pred_xstart": out["pred_xstart"], "pde_loss_before": pde_loss_before, "pde_loss_after": pde_loss_after}

    def ddim_sample_inversion_pdeguide(
        self,
        model,
        x,
        input,
        t,
        vel,
        dh,
        omega,
        m0,
        scale_factor=1.0, 
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            input,
            t,
            vel,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = mean_pred + nonzero_mask * sigma * noise
        denorm_vel = denormalizer_vel(vel)
        m = 1 / (denorm_vel**2)
        # pde_loss = self.pde_loss(sample, input, dh, omega, m, m0)
        # pde_loss_after = 0.0
        if self.pde_guide:
            # if t.item() < 800:
                #### first method
                # cond_grad, pde_loss = self.pde_guidance(x, input, dh, omega, m, m0, out["variance"], scale_factor=scale_factor)
                # sample = mean_pred + cond_grad + nonzero_mask * sigma * noise

            #### second method
            cond_grad, _ = self.pde_guidance(sample, input, dh, omega, m, m0, out["variance"], scale_factor=scale_factor)
            sample = sample - cond_grad
        pde_loss = self.pde_loss(sample, input, dh, omega, m, m0)

        # # if (t[0].item() + 1) % 50 == 0 or t[0].item() == 0:
            # # print(f'Time step {t[0].item()} --> PDE guider before Loss {pde_loss_before} and after Loss {pde_loss_after}')

        return {"sample": sample, "pred_xstart": out["pred_xstart"], "pde_loss": pde_loss}

    def ddim_reverse_sample(
        self,
        model,
        x,
        input,
        t,
        vel,
        dh,
        omega,
        m,
        m0,
        scale_factor=1.0,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            input,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        input,
        vel,
        shape,
        dh,
        omega,
        scale_factor=1.0,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample, image_all, pred_xstart, pde_loss_before, pde_loss_after in self.ddim_sample_loop_progressive(
            model,
            input,
            vel,
            shape,
            dh,
            omega,
            scale_factor=scale_factor,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final, image_all, pred_xstart, pde_loss_before, pde_loss_after = sample, image_all, pred_xstart, pde_loss_before, pde_loss_after
        return final["sample"], image_all, pred_xstart, pde_loss_before, pde_loss_after

    def ddim_sample_loop_progressive(
        self,
        model,
        input,
        vel,
        shape,
        dh,
        omega,
        scale_factor=1.0,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image = noise
        else:
            image = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        denorm_vel = denormalizer_vel(vel)
        m = 1 / (denorm_vel**2)
        m0 = th.full_like(denorm_vel, 1 / (denorm_vel.min()**2))

        image_all = []
        pred_xstart = []
        pde_loss_before = []
        pde_loss_after = []
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                        model,
                        image,
                        input,
                        t,
                        vel,
                        dh,
                        omega,
                        m=m,
                        m0=m0,
                        scale_factor=scale_factor, 
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                )
            if i % 50 == 0:
                image_all.append(out["sample"])
                pred_xstart.append(out["pred_xstart"])
            pde_loss_before.append(out["pde_loss_before"])
            pde_loss_after.append(out["pde_loss_after"])
            yield out, image_all, pred_xstart, pde_loss_before, pde_loss_after
            image = out["sample"]

    def ddim_sample_loop_inversion(
        self,
        obs,
        model,
        input,
        vel,
        shape,
        dh,
        omega,
        inversion_timestep=1,
        scale_factor=1.0,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample, image_all, pred_xstart, pde_loss_before, pde_loss_after, grad_all, loss_allstep in self.ddim_sample_loop_progressive_inversion(
            obs,
            model,
            input,
            vel,
            shape,
            dh,
            omega,
            inversion_timestep=inversion_timestep,
            scale_factor=scale_factor,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final, image_all, pred_xstart, pde_loss_before, pde_loss_after, grad_all, loss_allstep = sample, image_all, pred_xstart, pde_loss_before, pde_loss_after, grad_all, loss_allstep
        return final["sample"], image_all, pred_xstart, pde_loss_before, pde_loss_after, grad_all, loss_allstep

    def ddim_sample_loop_progressive_inversion(
        self,
        obs,
        model,
        input,
        vel,
        shape,
        dh,
        omega,
        inversion_timestep=1,
        scale_factor=1.0,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image = noise
        else:
            image = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        denorm_vel = denormalizer_vel(vel)
        m = 1 / (denorm_vel**2)
        m0 = th.full_like(denorm_vel, 1 / (denorm_vel.min().item()**2))

        image_all = []
        pred_xstart = []
        pde_loss_before = []
        pde_loss_after = []
        grad_all = []
        loss_allstep = []
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if i >= inversion_timestep:
                with th.no_grad():
                    out = self.ddim_sample(
                        model,
                        image,
                        input,
                        t,
                        vel,
                        dh,
                        omega,
                        m=m,
                        m0=m0,
                        scale_factor=scale_factor, 
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                    )
            else:
                out = self.ddim_sample(
                        model,
                        image,
                        input,
                        t,
                        vel,
                        dh,
                        omega,
                        m=m,
                        m0=m0,
                        scale_factor=scale_factor, 
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                )               
                loss = self.inversion_criterion(out["sample"][:, :, 0], obs[:, :, 0])
                loss.backward(retain_graph=True) 
                grad_all.append(th.autograd.grad(loss, vel, retain_graph=True)[0])
                loss_allstep.append(loss.item())

            if i % 5 == 0:
                image_all.append(out["sample"])
                pred_xstart.append(out["pred_xstart"])
            pde_loss_before.append(out["pde_loss_before"])
            pde_loss_after.append(out["pde_loss_after"])
            yield out, image_all, pred_xstart, pde_loss_before, pde_loss_after, grad_all, loss_allstep
            image = out["sample"]

    def ddim_sample_loop_inversion_pdeguide(
        self,
        obs,
        model,
        input,
        vel,
        shape,
        dh,
        omega,
        inversion_timestep=1,
        scale_factor=1.0,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample, image_all, pred_xstart, grad_all, dataloss_allstep, pdeloss_allstep in self.ddim_sample_loop_progressive_inversion_pdeguide(
            obs,
            model,
            input,
            vel,
            shape,
            dh,
            omega,
            inversion_timestep=inversion_timestep,
            scale_factor=scale_factor,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final, image_all, pred_xstart, grad_all, dataloss_allstep, pdeloss_allstep = sample, image_all, pred_xstart, grad_all, dataloss_allstep, pdeloss_allstep
        return final["sample"], image_all, pred_xstart, grad_all, dataloss_allstep, pdeloss_allstep

    def ddim_sample_loop_progressive_inversion_pdeguide(
        self,
        obs,
        model,
        input,
        vel,
        shape,
        dh,
        omega,
        inversion_timestep=1,
        scale_factor=1.0,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image = noise
        else:
            image = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        denorm_vel = denormalizer_vel(vel)
        m0 = th.full_like(denorm_vel, 1 / (denorm_vel.min().item()**2))

        image_all = []
        pred_xstart = []
        grad_all = []
        dataloss_allstep = []
        pdeloss_allstep = []
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if i >= inversion_timestep:
                with th.no_grad():
                    out = self.ddim_sample_inversion_pdeguide(
                        model,
                        image,
                        input,
                        t,
                        vel,
                        dh,
                        omega,
                        m0=m0,
                        scale_factor=scale_factor, 
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                    )
            else:
                out = self.ddim_sample_inversion_pdeguide(
                        model,
                        image,
                        input,
                        t,
                        vel,
                        dh,
                        omega,
                        m0=m0,
                        scale_factor=scale_factor, 
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                )               
                data_loss = self.inversion_criterion(out["sample"][:, :, 0], obs[:, :, 0])
                pde_loss = out["pde_loss"]
                loss = data_loss + 0.0001*pde_loss
                loss.backward(retain_graph=True) 
                grad_all.append(th.autograd.grad(loss, vel, retain_graph=True)[0])
                dataloss_allstep.append(data_loss.item())
                pdeloss_allstep.append(pde_loss.item())

            if i % 5 == 0:
                image_all.append(out["sample"])
                pred_xstart.append(out["pred_xstart"])
            yield out, image_all, pred_xstart, grad_all, dataloss_allstep, pdeloss_allstep
            image = out["sample"]


    # calculate loss，KL divergence
    def _vb_terms_bpd(
        self, model, x_start, x_t, input, t, vel, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """

        # When calculating KL divergence, it is necessary to distinguish whether t-1 is equal to 0
        '''
         不等于0的情况下利用IDDPM文章的式(6)来计算L_(t-1)
        '''
        # The real x[0], x[t] and t are used to calculate the mean and variance of x[t-1]
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        # x[t], t and the predicted x[0] are used to calculate the mean and variance of x[t-1]
        out = self.p_mean_variance(
            model, x_t, input, t, vel, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        # KL divergence between p_theta and q distribution
        # normal_kl is used to calculate the KL divergence between two Gaussian distributions
        # Corresponding to the L_(t-1) loss function in Equation (6)
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        '''
        When L_0 is equal to 0, use Equation (5) of the IDDPM paper to calculate L_0
        '''
        # L_0 loss function is calculated using negative log-likelihood (discrete Gaussian log-likelihood)
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        # If it is equal to 0, the output returned is decoder_nll
        # If it is not equal to 0, the output returned is kl
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    # Selection of training loss
    def training_losses(self, model, x_start, input, t, vel, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of image gather.
        :param input: the [N x C x ...] tensor of inputocity model.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        # Sample x_t based on x_0 and any time t and noise
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        # # If loss is KL loss or RESCALED_KL
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                input=input,
                t=t,
                vel=vel,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            # If loss is RESCALED_KL, the calculated loss is multiplied by a corresponding weight, which is self.num_timesteps
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        # If loss is MSE loss or RESCALED_MSE
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:

            model_output = model(x_t, input, self._scale_timesteps(t), vel, **model_kwargs)

            # Determine whether the variance is learnable
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                # If variance is learnable, the predicted values ​​will include the variance
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                # This is to ensure that the predicted variance will not affect the predicted mean
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                # This function customizes the predicted value of the model, that is, the out returned by the model's prediction is frozen_out
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    input=input,
                    t=t,
                    vel=vel,
                    clip_denoised=False,
                )["output"]
                # If using LossType.RESCALED_MSE, we need to add terms["vb"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            # This refers to the output type of the network
            # The three key values ​​in the target list have different meanings: ModelMeanType.PREVIOUS_X, ModelMeanType.START_X, ModelMeanType.EPSILON
            # ModelMeanType.PREVIOUS_X represents the mean of the network prediction at the previous moment
            # ModelMeanType.START_X represents the network prediction x0
            # ModelMeanType.EPSILON represents the network prediction of noise
            # Finally, the dictionary selects the corresponding network output type according to the key value self.model_mean_type passed in
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            # If the variance is learnable, we need to add terms["vb"] loss here
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    # Prior KL divergence
    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

     # Calculate all the losses from time T to time 0
    def calc_bpd_loop(self, model, x_start, input, vel, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of image gather.
        :param input: the [N x C x ...] tensor of inputocity model.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    input=input,
                    t=t_batch,
                    vel=vel,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def laplace_operator(order=4, device='cuda'):
    if order == 2:
        laplace_kernel = th.tensor([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], dtype=th.float32)
        laplace_kernel = laplace_kernel.view(1, 1, 3, 3).to(device)
    elif order == 4:
        laplace_kernel = th.tensor([[0, 0, -1/12, 0, 0],
                                    [0, 0, 16/12, 0, 0],
                                    [-1/12, 16/12, -30/12*2, 16/12, -1/12],
                                    [0, 0, 16/12, 0, 0],
                                    [0, 0, -1/12, 0, 0]], dtype=th.float32)
        laplace_kernel = laplace_kernel.view(1, 1, 5, 5).to(device)
    return laplace_kernel