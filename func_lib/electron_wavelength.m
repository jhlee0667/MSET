function [lambda] = electron_wavelength(E0)
%For energy E0 in eV give the wavelength in Angstrom
    m = 9.109383*10^-31;
    e = 1.602177*10^-19;
    c =  299792458;
    h = 6.62607*10^-34;
    lambda = h/sqrt(2*m*e*E0) ...
        /sqrt(1 + e*E0/2/m/c^2) * 10^10; % wavelength in A
end

