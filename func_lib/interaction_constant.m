function [sigma] = interaction_constant(E0)
%Calculates the electron interaction constant sigma for a given electron
%accelerating voltage
    m = 9.109383*10^-31;
    e = 1.602177*10^-19;
    c =  299792458;
    h = 6.62607*10^-34;
    lambda = electron_wavelength(E0);
    sigma = (2*pi/lambda/E0)*(m*c^2+e*E0)/(2*m*c^2+e*E0);
end

