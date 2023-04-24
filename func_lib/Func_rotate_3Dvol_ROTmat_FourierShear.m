% My_rotate_3Dvol_ROTmat_FourierShear
% Y. Yang, UCLA Physics & Astronomy
% First version date: 2018. 02. 05.

% output parameter: ROTvol: rotated 3D volume
% input parameters: Vol (original 3D volume)   
%                   MAT (3x3 rotation matrix)   
%                   rotStepInit (positive number for suggested rotation
%                   stepsize)

% My_rotate_3Dvol_ROTmat_FourierShear rotates a 3D volume using Fourier
% shear method as given in Welling et al., Graphical Models 68 (2006) 356-370

function RotVol = Func_rotate_3Dvol_ROTmat_FourierShear(initVol,MAT,rotStepInit,varargin)    
    
    if nargin > 3
        GPUyn = varargin{1};
    else
        GPUyn = 0;
    end

    % initilize size
    Nx = size(initVol,1);
    Ny = size(initVol,2);
    Nz = size(initVol,3);

    % obtain quaternion from matrix
    [x_ori,y_ori,z_ori,w_ori] = My_obtain_quaternion_from_matrix(MAT);
    
    % obtain rotation angle
    Ang = acosd(w_ori)*2; % acosd should always be positive
    if abs(Ang) < 1e-1
        fprintf(1,'Rotation Angle Too Small (%8.5f deg)\n',Ang);
    end
    
    Vec = [x_ori y_ori z_ori] / norm([x_ori y_ori z_ori]);

    numStep = floor(Ang / rotStepInit) + 1;
    rotStep = Ang / numStep;

    % rotation using the input rotStep
    w = cosd(rotStep/2);
    x = -sind(rotStep/2)*Vec(1); y = -sind(rotStep/2)*Vec(2); z = -sind(rotStep/2)*Vec(3);

    % check which shear decomposition shows minimum sum of absolute
    % components
    absSumar_forward = zeros(1,3);
    for i=1:3
        [~, ~, ~, ~, ~, absSum] = calculate_S_matrices_for3DFourierShearRot_perm_ver1(x,y,z,w,i);
        absSumar_forward(i) = absSum;
    end

    absSumar_backward = zeros(1,3);
    for i=1:3
        [~, ~, ~, ~, ~, absSum] = calculate_S_matrices_for3DFourierShearRot_perm_backward_ver1(x,y,z,w,i);
        absSumar_backward(i) = absSum;
    end

    [minV_forw, minI_forw]  = min(absSumar_forward);
    [minV_back, minI_back]  = min(absSumar_backward);

    % obtain shear coefficients for minimum sum of absolute decomposition
    if minV_forw < minV_back
        [~, ~, ~, ~, Coeffs, ~] = calculate_S_matrices_for3DFourierShearRot_perm_ver1(x,y,z,w, minI_forw);
    else
        [~, ~, ~, ~, Coeffs, ~] = calculate_S_matrices_for3DFourierShearRot_perm_backward_ver1(x,y,z,w,minI_back);
    end

    % initialize 3dgrid
    [nx, ny, nz] = ndgrid( (1:Nx) - round((Nx+1)/2), (1:Ny) - round((Ny+1)/2), (1:Nz) - round((Nz+1)/2));
    %[nx, ny, nz] = ndgrid(0:Nx-1, 0:Ny-1, 0:Nz-1);
    nx = ifftshift(nx);
    ny = ifftshift(ny); 
    nz = ifftshift(nz);
    
    if mod(Nx,2) == 0
        nx(round((Nx+1)/2),:,:) = 0;
        ny(round((Nx+1)/2),:,:) = 0;
        nz(round((Nx+1)/2),:,:) = 0;
    end
    if mod(Ny,2) == 0
        nx( :,round((Ny+1)/2),:) = 0;
        ny( :,round((Ny+1)/2),:) = 0;
        nz( :,round((Ny+1)/2),:) = 0;
    end
    if mod(Nz,2) == 0
        nx( :,:,round((Nz+1)/2)) = 0;
        ny( :,:,round((Nz+1)/2)) = 0;
        nz( :,:,round((Nz+1)/2)) = 0;
    end

    % apply shear transform of rotStep angle multiple times to get total Ang.
    SheerVol = initVol;
    SheerVol = ifftshift(SheerVol);
    
    if GPUyn
        SheerVol = gpuArray(single(SheerVol));
        Nx = gpuArray((Nx));
        Ny = gpuArray((Ny));
        Nz = gpuArray((Nz));
        nx = gpuArray(single(nx));
        ny = gpuArray(single(ny));
        nz = gpuArray(single(nz));
        Coeffs.a = gpuArray(Coeffs.a);
        Coeffs.b = gpuArray(Coeffs.b);
        Coeffs.c = gpuArray(Coeffs.c);
        Coeffs.d = gpuArray(Coeffs.d);
        Coeffs.e = gpuArray(Coeffs.e);
        Coeffs.f = gpuArray(Coeffs.f);
        Coeffs.g = gpuArray(Coeffs.g);
        Coeffs.h = gpuArray(Coeffs.h);
        minI_forw = gpuArray(minI_forw);
        minI_back = gpuArray(minI_back);
    end

    for i=1:numStep
        if minV_forw < minV_back
            SheerVol = SheerTransforms_for3Drot(SheerVol,Nx,Ny,Nz,nx,ny,nz,Coeffs.g,Coeffs.h,mod(minI_forw+1,3)+1);
            SheerVol = SheerTransforms_for3Drot(SheerVol,Nx,Ny,Nz,nx,ny,nz,Coeffs.e,Coeffs.f,mod(minI_forw,3)+1);
            SheerVol = SheerTransforms_for3Drot(SheerVol,Nx,Ny,Nz,nx,ny,nz,Coeffs.c,Coeffs.d,mod(minI_forw+2,3)+1);
            SheerVol = real(SheerTransforms_for3Drot(SheerVol,Nx,Ny,Nz,nx,ny,nz,Coeffs.a,Coeffs.b,mod(minI_forw+1,3)+1));    
        else
            SheerVol = SheerTransforms_for3Drot(SheerVol,Nx,Ny,Nz,nx,ny,nz,Coeffs.a,Coeffs.b,mod(minI_back+1,3)+1);
            SheerVol = SheerTransforms_for3Drot(SheerVol,Nx,Ny,Nz,nx,ny,nz,Coeffs.c,Coeffs.d,mod(minI_back+2,3)+1);
            SheerVol = SheerTransforms_for3Drot(SheerVol,Nx,Ny,Nz,nx,ny,nz,Coeffs.e,Coeffs.f,mod(minI_back,3)+1);
            SheerVol = real(SheerTransforms_for3Drot(SheerVol,Nx,Ny,Nz,nx,ny,nz,Coeffs.g,Coeffs.h,mod(minI_back+1,3)+1));              
        end
    end
    
    RotVol = fftshift(SheerVol);
    if GPUyn
        RotVol = gather(RotVol);
    end

    