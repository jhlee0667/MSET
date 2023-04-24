% My_obtain_quaternion_from_matrix
% Y. Yang, UCLA Physics & Astronomy
% First version date: 2018. 02. 05.

% output parameter: [qx, qy, qz, qw]: quaternion elements
% input parameters: MAT (3x3 SO(3) matrix)

% My_obtain_quaternion_from_matrix calculates quaternion elements from a 3D
% rotation matrix

function [qx, qy, qz, qw] = My_obtain_quaternion_from_matrix(MAT)

Tr = MAT(1,1)+MAT(2,2)+MAT(3,3);

if (Tr > 0)
  S = sqrt(Tr+1) * 2; % S=4*qw 
  qw = 0.25 * S;
  qx = (MAT(3,2) - MAT(2,3)) / S;
  qy = (MAT(1,3) - MAT(3,1)) / S; 
  qz = (MAT(2,1) - MAT(1,2)) / S; 
elseif ((MAT(1,1) > MAT(2,2))&&(MAT(1,1) > MAT(3,3)))
  S = sqrt(1.0 + MAT(1,1) - MAT(2,2) - MAT(3,3)) * 2; % S=4*qx 
  qw = (MAT(3,2) - MAT(2,3)) / S;
  qx = 0.25 * S;
  qy = (MAT(1,2) + MAT(2,1)) / S; 
  qz = (MAT(1,3) + MAT(3,1)) / S; 
elseif ( MAT(2,2) >MAT(3,3))
  S = sqrt(1.0 +  MAT(2,2) -  MAT(1,1) - MAT(3,3)) * 2;% // S=4*qy
  qw = (MAT(1,3) - MAT(3,1)) / S;
  qx = (MAT(1,2) + MAT(2,1)) / S; 
  qy = 0.25 * S;
  qz = (MAT(2,3) + MAT(3,2)) / S; 
else
  S = sqrt(1.0 + MAT(3,3) - MAT(1,1) - MAT(2,2)) * 2; %// S=4*qz
  qw = (MAT(2,1) - MAT(1,2)) / S;
  qx = (MAT(1,3) + MAT(3,1)) / S;
  qy = (MAT(2,3) + MAT(3,2)) / S;
  qz = 0.25 * S;
end