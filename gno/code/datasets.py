from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import random


def load_data(*, data_dir, batch_size, class_cond=False, deterministic=False):
    """
    Build an infinite generator that yields (du, u0, vel, kwargs) batches
    from a directory of .mat / .npy / .npz training files.

    Each yielded tuple contains:
        du   : float32 tensor of shape (B, 2, nz, nx) — scattered wavefield
               (channel 0: real part, channel 1: imaginary part).
        u0   : float32 tensor of shape (B, 2, nz, nx) — background wavefield
               (channel 0: real part, channel 1: imaginary part).
        vel  : float32 tensor of shape (B, 1, nz, nx) — velocity model,
               normalized to [-1, 1] via normalizer_vel().
        kwargs (dict): empty by default; contains {"y": class_label_tensor}
                       when class_cond=True.

    The generator loops over the dataset indefinitely, making it suitable for
    use directly inside a training loop without manual epoch management.

    Args:
        data_dir      (str)  : Root directory containing training .mat files
                               (searched recursively).
        batch_size    (int)  : Number of samples per yielded batch.
        class_cond    (bool) : If True, include a "y" key in the returned dict
                               carrying integer class labels. Raises an error if
                               labels are unavailable. Default: False.
        deterministic (bool) : If True, disable shuffling so batches are yielded
                               in a fixed order (useful for reproducible evaluation).
                               Default: False (shuffle enabled for training).

    Yields:
        tuple: (du, u0, vel, kwargs) — see above for shapes and dtypes.

    Raises:
        ValueError: If data_dir is empty or not specified.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    # Recursively collect all supported data files under data_dir
    all_files = _list_image_files_recursively(data_dir)

    # Wrap the file list in a PyTorch Dataset
    dataset = BasicDataset(
        all_files,
        class_cond=class_cond,
    )

    # Build a DataLoader; drop_last=True ensures every batch has exactly batch_size
    # samples, avoiding shape mismatches on the final (smaller) batch
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,  # Shuffle for training; fixed order for eval
        num_workers=1,              # Single worker avoids multiprocessing overhead
                                    # on HPC file systems (e.g., KAUST IBEX Lustre)
        drop_last=True,             # Discard the last incomplete batch
    )

    # Loop indefinitely over the DataLoader so the training loop never exhausts
    # the generator regardless of how many gradient steps are requested
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    """
    Recursively collect all supported data files under data_dir.

    Supported extensions: .mat, .npy, .npz.
    Files are collected in sorted order at each directory level to ensure
    a consistent, reproducible file list across runs.

    Args:
        data_dir (str): Root directory to search.

    Returns:
        list of str: Absolute paths to all discovered data files.
    """
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["mat", "npy", "npz"]:
            # Supported data file — add to the list
            results.append(full_path)
        elif bf.isdir(full_path):
            # Subdirectory — recurse into it
            results.extend(_list_image_files_recursively(full_path))
    return results


def normalizer_vel(x, dmin=1.5, dmax=4.5):
    """
    Linearly normalize a velocity model from physical units to [-1, 1].

    Applies the mapping:
        x_norm = 2 * (x - dmin) / (dmax - dmin) - 1

    so that dmin maps to -1 and dmax maps to +1, matching the expected
    input range of the diffusion model's noise schedule.

    Args:
        x    (np.ndarray): Velocity model in km/s.
        dmin (float)     : Minimum expected velocity (km/s). Default: 1.5.
        dmax (float)     : Maximum expected velocity (km/s). Default: 4.5.

    Returns:
        np.ndarray: Normalized velocity in [-1, 1].
    """
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0


def denormalizer_vel(x, dmin=1.5, dmax=4.5):
    """
    Invert normalizer_vel(): map a normalized velocity back to physical units.

    Applies the inverse mapping:
        x_phys = 0.5 * (x + 1) * (dmax - dmin) + dmin

    Used at inference time to convert the model's output back to km/s for
    evaluation and visualization.

    Args:
        x    (np.ndarray): Normalized velocity in [-1, 1].
        dmin (float)     : Minimum velocity used during normalization (km/s). Default: 1.5.
        dmax (float)     : Maximum velocity used during normalization (km/s). Default: 4.5.

    Returns:
        np.ndarray: Velocity model in km/s.
    """
    return 0.5 * (x + 1) * (dmax - dmin) + dmin


class BasicDataset(Dataset):
    """
    PyTorch Dataset for frequency-domain seismic wavefield data.

    Each sample is loaded from a .mat file containing one shot gather's worth
    of complex-valued scattered (du) and background (u0) wavefields, along with
    the corresponding velocity model (v). The fields are split into real and
    imaginary channels and returned as float32 NumPy arrays.

    Expected .mat file contents:
        du_real (nz, nx) : Real part of the scattered wavefield.
        du_imag (nz, nx) : Imaginary part of the scattered wavefield.
        u0_real (nz, nx) : Real part of the background (incident) wavefield.
        u0_imag (nz, nx) : Imaginary part of the background wavefield.
        v       (nz, nx) : P-wave velocity model in km/s.

    Args:
        paths      (list of str): Absolute paths to .mat training files.
        class_cond (bool)       : If True, class labels will be returned in the
                                  output dict (not yet implemented; reserved for
                                  future class-conditional training). Default: False.
    """

    def __init__(self, paths, class_cond=False):
        super().__init__()
        self.local_dataset = paths   # List of absolute file paths
        self.class_cond = class_cond

    def __len__(self):
        """Return the total number of training samples."""
        return len(self.local_dataset)

    def __getitem__(self, idx):
        """
        Load and return a single training sample by index.

        Workflow:
            1. Load the .mat file at self.local_dataset[idx].
            2. Extract real and imaginary parts of du and u0.
            3. Load and normalize the velocity model to [-1, 1].
            4. Stack real/imaginary channels: shape (2, nz, nx).
            5. Add a channel dimension to vel: shape (1, nz, nx).

        Args:
            idx (int): Index into self.local_dataset.

        Returns:
            tuple:
                du      (np.ndarray, float32, shape (2, nz, nx)): Scattered wavefield
                         [real, imag] — the diffusion model's training target.
                u0      (np.ndarray, float32, shape (2, nz, nx)): Background wavefield
                         [real, imag] — conditioning input to the UNet.
                vel     (np.ndarray, float32, shape (1, nz, nx)): Normalized velocity
                         model — conditioning input to the UNet.
                out_dict (dict): Empty by default; will carry class labels under
                                 key "y" when class_cond=True.
        """
        path = self.local_dataset[idx]

        # Load all fields from the .mat file
        dict = sio.loadmat(path)
        du_real = dict['du_real']   # Real part of scattered wavefield
        du_imag = dict['du_imag']   # Imaginary part of scattered wavefield
        u0_real = dict['u0_real']   # Real part of background wavefield
        u0_imag = dict['u0_imag']   # Imaginary part of background wavefield
        v = dict['v']               # Velocity model (km/s)

        # Normalize velocity from physical units [km/s] to [-1, 1]
        v = normalizer_vel(v)

        # Cast all arrays to float32 for PyTorch compatibility
        du_real = np.array(du_real, dtype=np.float32)
        du_imag = np.array(du_imag, dtype=np.float32)
        u0_real = np.array(u0_real, dtype=np.float32)
        u0_imag = np.array(u0_imag, dtype=np.float32)
        v       = np.array(v,       dtype=np.float32)

        # Stack real and imaginary parts along a new leading channel axis
        # Result shapes: (2, nz, nx) — channel 0: real, channel 1: imaginary
        du = np.stack((du_real, du_imag), axis=0)
        u0 = np.stack((u0_real, u0_imag), axis=0)

        # Placeholder for optional class labels (unused when class_cond=False)
        out_dict = {}

        # Add channel dimension to velocity: (nz, nx) → (1, nz, nx)
        return du, u0, np.expand_dims(v, axis=0), out_dict