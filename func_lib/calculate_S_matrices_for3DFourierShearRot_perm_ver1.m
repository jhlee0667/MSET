% calculate_S_matrices_for3DFourierShearRot_perm
% Y. Yang, UCLA Physics & Astronomy
% First version date: 2018. 02. 05.

% output parameter: S_1, S_2, S_3, S_4: four decomposed shear matrices
%                   Coeffs: coefficients a-h for the shear matrices
%                   absSum: sum of absolute value of the Coeffs
% input parameters: X_inp,Y_inp,Z_inp,W : four quaternion elements for rotation  
%                   numPerm: integer for indicating how many circular
%                   permuatation of (xyz) to apply

% calculate_S_matrices_for3DFourierShearRot_perm calculates the forward
% transform shear matrices and coefficients for 3D rotation
% Reference: Welling et al., Graphical Models 68 (2006) 356-370

function [S_1, S_2, S_3, S_4, Coeffs, absSum] = calculate_S_matrices_for3DFourierShearRot_perm_ver1(X_inp,Y_inp,Z_inp,W,numPerm)
   
    PermXYZ = circshift([X_inp,Y_inp,Z_inp],[0 -1*numPerm]);
    X = PermXYZ(1);
    Y = PermXYZ(2);
    Z = PermXYZ(3);

    delX = 0;
    delY = 0;
    delZ = 0;
    
    Singular = 0;
    
    if abs(Y*Z-X*W) > 1e-5 && abs(X*Y-Z*W) > 1e-5

        a = (X^2+Y^2) / (Y*Z-X*W);
        b = (Y^2-Z^2) / (X*Y-Z*W) - 2*(Y*Z*(X^2+Y^2))/(X*Y-Z*W)/(Y*Z-X*W);
        c = 2*Y*Z / (X*Y-Z*W);
        d = 2*(X*W-Y*Z);
        e = 2*(X*Y-Z*W);
        f = -2*X*Y/(Y*Z-X*W);
        g = (X^2-Y^2) / (Y*Z-X*W) + 2* X*Y*(Y^2+Z^2) / (X*Y-Z*W) / (Y*Z-X*W);
        h = (-1+X^2+W^2) / (X*Y-Z*W);   
        
    elseif abs(Y)<1e-6    
        a = -X / W;
        b = Z/W;
        c = 0;
        d = 2*X*W;
        e = -2*Z*W;
        f = 0;
        g = -X/W;
        h = Z/W;
        
    else
        Singular = 1;
    end
    
    if Singular
        S_1 =[];
        S_2 =[];
        S_3 =[];
        S_4 =[];
        Coeffs = [];
        absSum = inf;
        
    else        
        if mod(numPerm,3) == 0
            S_1 = S_y_MAT(a,b,delY);
            S_2 = S_z_MAT(c,d,delZ);
            S_3 = S_x_MAT(e,f,delX);
            S_4 = S_y_MAT(g,h,delY);
        elseif mod(numPerm,3) == 1
            S_1 = S_z_MAT(a,b,delZ);
            S_2 = S_x_MAT(c,d,delX);
            S_3 = S_y_MAT(e,f,delY);
            S_4 = S_z_MAT(g,h,delZ);
        elseif mod(numPerm,3) == 2
            S_1 = S_x_MAT(a,b,delX);
            S_2 = S_y_MAT(c,d,delY);
            S_3 = S_z_MAT(e,f,delZ);
            S_4 = S_x_MAT(g,h,delX);
        end    

        Coeffs.a = a;
        Coeffs.b = b;
        Coeffs.c = c;
        Coeffs.d = d;
        Coeffs.e = e;
        Coeffs.f = f;
        Coeffs.g = g;
        Coeffs.h = h;

        absSum = sum(abs([a b c d e f g h]));
    
    end

end


function S_x = S_x_MAT(a,b,delX)

S_x = [1 a b delX;
       0 1 0 0;
       0 0 1 0;
       0 0 0 1];
end

function S_y = S_y_MAT(a,b,delY)

S_y = [1 0 0 0;
       b 1 a delY;
       0 0 1 0;
       0 0 0 1];
end

function S_z = S_z_MAT(a,b,delZ)
S_z = [1 0 0 0;
       0 1 0 0;
       a b 1 delZ;
       0 0 0 1];
end