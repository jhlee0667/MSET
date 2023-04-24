% SheerTransforms_for3Drot
% Y. Yang, UCLA Physics & Astronomy
% First version date: 2018. 02. 05.

% output parameter: SheerVol: 3D volume after shear transform
% input parameters: Vol: 3D input volume
%                   Nx,Ny,Nz: size of volume in 1,2,3rd dim
%                   nx,ny,nz: ndgrid grid matrices for nx,ny,nz
%                   a,b: shear components
%                   Dim: shear dimension

% SheerTransforms_for3Drot applies shear transform using given parameters
% to the given dimension.
% Reference: Welling et al., Graphical Models 68 (2006) 356-370

function SheerVol = SheerTransforms_for3Drot(Vol,Nx,Ny,Nz,nx,ny,nz,a,b,Dim)

    if Dim==1
        SheerVol = SheerX_for3Drot(Vol,Nx,Ny,Nz,nx,ny,nz,a,b);
    elseif Dim==2
        SheerVol = SheerY_for3Drot(Vol,Nx,Ny,Nz,nx,ny,nz,a,b);
    elseif Dim==3
        SheerVol = SheerZ_for3Drot(Vol,Nx,Ny,Nz,nx,ny,nz,a,b);
    end
    
end


function SheerVol = SheerX_for3Drot(Vol,Nx,Ny,Nz,nx,ny,nz,a,b)

    VolF = fft(fft(Vol,[],3),[],2);
    VolF = VolF.*exp(1i*2*pi*(Ny / Nx * a * ny / Ny + Nz / Nx * b * nz / Nz).*nx);
    SheerVol = ifft(ifft(VolF,[],3),[],2);
end

function SheerVol = SheerY_for3Drot(Vol,Nx,Ny,Nz,nx,ny,nz,a,b)

    VolF = fft(fft(Vol,[],3),[],1);
    VolF = VolF.*exp(1i*2*pi*(Nx / Ny * b * nx / Nx + Nz / Ny * a * nz / Nz).*ny);
    SheerVol = ifft(ifft(VolF,[],3),[],1);

end

function SheerVol = SheerZ_for3Drot(Vol,Nx,Ny,Nz,nx,ny,nz,a,b)

    VolF = fft(fft(Vol,[],2),[],1);
    VolF = VolF.*exp(1i*2*pi*(Nx / Nz * a * nx / Nx + Ny / Nz * b * ny / Ny).*nz);
    SheerVol = ifft(ifft(VolF,[],2),[],1);
end
