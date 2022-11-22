% J.Lee, KAIST (Korea), 2022 
% input 3D Vol, inverse rotation information { vec1, vec2, vec3, phi, theta, psi}

function RotVol = Func_inverse_rotating_vol_OV(OV_vol, vec1, vec2, vec3 ,phi, theta, psi, OV...
    ,use_cpp)

    if nargin <= 8
        use_cpp = 0;
    end

    [dimx, dimy, dimz] = size(OV_vol);
    
    ncy = round((dimy+1)/2); 
    ncx = round((dimx+1)/2); 
    ncz = round((dimz+1)/2);
    
    %calculate rotation matrix
    R1 = MatrixQuaternionRot(vec1,phi);
    R2 = MatrixQuaternionRot(vec2,theta);
    R3 = MatrixQuaternionRot(vec3,psi);
      
    R =(R1*R2*R3)';
    R = inv(R);

    if abs(theta) > 1
        RotVol = My_rotate_3Dvol_ROTmat_FourierShear_ver4(OV_vol,R,11);
    else
        RotVol = OV_vol;
    end
    
%     [Ry Rx Rz ] = meshgrid(((1:dimy) - ncy) / dimy, ((1:dimx) - ncx) /  dimx, ((1:dimz) - ncz) /dimz);
%     
%     %rotate coordinates
%     rotRCoords = R'*[Rx(:)';Ry(:)';Rz(:)'];
%     rotRx = rotRCoords(1,:);
%     rotRy = rotRCoords(2,:);
%     rotRz = rotRCoords(3,:);
%     
%     %reshape for interpolation
%     rotRx = reshape(rotRx,size(Rx));
%     rotRy = reshape(rotRy,size(Ry));
%     rotRz = reshape(rotRz,size(Rz));
%     
%     %rotation
%     if use_cpp ==1
%         RotVol = splinterp3(double(OV_vol),(rotRy+ncy/dimy)*dimy,(rotRx+ncx/dimx)*dimx,(rotRz+ncz/dimz)*dimz);
%     else
%         RotVol = interp3(Ry,Rx,Rz,OV_vol,rotRy,rotRx,rotRz,'linear');
%     end
%    
%     RotVol(isnan(RotVol)) = 0;

    if OV > 1
        RotVol=imresize3(RotVol,[size(RotVol,1)*1/OV, size(RotVol,2)*1/OV, size(RotVol,3)], 'box');
    end

end