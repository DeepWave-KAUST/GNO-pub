clear all;
close all;
nz = 101; nx = 101;
dx = 0.025; dz = 0.025;
h  = [dz dx];
src_z = 2;
npmlz = 60; npmlx = npmlz;
Nz = nz + 2*npmlz;
Nx = nx + 2*npmlx;
% Distance from source to each point in the model
r = @(zz,xx)(zz.^2+xx.^2).^0.5;

data_folder = '../../train/';
mkdir(data_folder)

load('../v_train.mat');
data_num = size(v_train, 1);
f_list = 4:10;
nshot = 5;
nsmooth = 5;
min_smooth = 2;
max_smooth = 20;
data_id = 1;
for f = f_list
    for imodel = 1:data_num
        v_nosmooth = squeeze(v_train(imodel, :, :));
        v_nosmooth = imresize(v_nosmooth, [nz nx], 'nearest');
        sorted_numbers = random_num(nsmooth - 1, min_smooth, max_smooth);
        for ismooth=1:nsmooth
            if ismooth == 1
                v = v_nosmooth;
            else
                v = imgaussfilt(v_nosmooth, sorted_numbers(ismooth-1));
            end

            omega = 2*pi*f;
            v = 0.001*v;
            vv = min(min(v));
            K = (omega./vv);

            %% ANALYTICAL Solution (Background wavefield)
            G_2D_analytic = @(zz,xx)0.25i * besselh(0,2,(K) .* r(zz,xx));

            %% Numerical results
            shot_loc = randperm(99, nshot) + 1;
            for src_x = shot_loc
                z  = [0:(nz-1)]'*h(1);
                x  = [0:(nx-1)]*h(2);
                sx = (src_x-1)*dx; 
                ns = length(sx);

                [zz,xx] = ndgrid(z,x);
                sx = repmat(sx,nx*nz,1);

                x1 = xx(:); 
                x_test = (repmat(x1,ns,1)); 
                z1 = zz(:);
                z_test = (repmat(z1,ns,1));
                sx_test = sx(:);
                v0 = ones(nx,nx)*min(min(v));

                v_e=extend2d(v,npmlz,npmlx,Nz,Nx);

                Ps1 = getP_H([nz,nx],npmlz,npmlx,src_z,src_x);
                Ps1 = Ps1'*12000;

                [o,d,n] = grid2odn(z,x);
                n=[n,1];

                nb = [npmlz  npmlx 0];
                n  = n + 2*nb;

                A = Helm2D((omega)./v_e(:),o,d,n,nb);
                U  = A\Ps1;

                du_real = zeros(nz*nx,1);
                du_imag = zeros(nz*nx,1);
  
                for is = 1:ns
                    U_2D = reshape(full(U(:,is)),[nz+2*npmlz,nx+2*npmlx]);
                    U_2d = U_2D(npmlz+1:end-npmlz,npmlx+1:end-npmlx);

                    xs = (src_x(is)-1)*dx;
                    zs = (src_z-1)*dz;

                    G_2D = (G_2D_analytic(zz - zs, xx - xs))*7.7;  

                    G_2D(src_z,src_x(is)) = (G_2D(src_z-1,src_x(is)) + G_2D(src_z+1,src_x(is)) + G_2D(src_z,src_x(is)-1) + G_2D(src_z,src_x(is)+1))/4;

                    dU_2d = U_2d-G_2D;

                    du_real = real(dU_2d);
                    du_imag = imag(dU_2d);

                    u0_real = real(G_2D);
                    u0_imag = imag(G_2D);
                end
                du_real(abs(du_real)>2)=0;
                save(strcat('../dataset/train/data', num2str(data_id),'.mat'), 'src_x', 'v', 'du_real', 'du_imag', 'u0_real', 'u0_imag')
                data_id = data_id + 1
            end
        end
    end
end
