% J.Lee, KAIST (Korea), 2022 
% input 3D Vol, inverse rotation information { vec1, vec2, vec3, phi, theta, psi}
% output inverse rotated 3D Vol

function RotVol = Func_inv_rot_3Dvol_FourierShear(Vol, vec1, vec2, vec3 ,phi, theta, psi )
    
    % Calculate rotation matrix
    R1 = MatrixQuaternionRot(vec1,phi);
    R2 = MatrixQuaternionRot(vec2,theta);
    R3 = MatrixQuaternionRot(vec3,psi);
      
    R =(R1*R2*R3)';
    R = inv(R);

    if abs(theta) > 0.5 || abs(phi) > 0.5 || abs(psi) > 0.5
        RotVol = Func_rotate_3Dvol_ROTmat_FourierShear(Vol,R,11);
    else
        RotVol = Vol;
    end

end