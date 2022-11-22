% Author: J.Lee, KAIST (Korea), 2022.
% Y.Yang, Multi-Dimensional Atomic Imaging Lab
% Multislice simulation package


function [STEM_data] = STEP02_FORWARD_4D(STEM_data) 
    
    % make slices of 2D projected potentials
    %OV=STEM_data.oversampling;
    %thickness = STEM_data.slice_thickness*OV;
    %volsize = size(STEM_data.rec,3)*OV;
    %thickness = STEM_data.slice_thickness;
    %volsize = size(STEM_data.rec,3);

    RotVol = STEM_data.RVol;
    %potz = double(zeros(size(RotVol,1),size(RotVol,2),ceil(volsize/thickness)));

    %for k=1:ceil(volsize/thickness)
    %    if k < ceil(volsize/thickness)
    %        potz(:,:,k)=sum(RotVol(:,:,thickness*(k-1)+1:thickness*k),3);
    %    else
    %        potz(:,:,k)=sum(RotVol(:,:,thickness*(k-1)+1:volsize),3);
    %    end
    %end
    
    % Calculate transmission function
    STEM_data.potz = RotVol * STEM_data.potential_pixelsize;
    STEM_data.trans = single(exp(1i*STEM_data.sigma*STEM_data.potz));


    % index
    x1 = STEM_data.row;
    y1 = STEM_data.col;
    
    % Calculate incident beam wave function
    %wave_f = single(Func_generate_probe_wave_scan(STEM_data.probe_wfn,x1,y1,STEM_data.probe_step_size/STEM_data.potential_pixelsize));
    wave_f = single(Func_generate_probe_wave_scan_sub_pixel(STEM_data.probe_wfn,x1,y1));

    
    % Forward propagation
    for a2 = 1:STEM_data.numPlanes
        wave_f = fftshift(fftn((ifftshift(wave_f.*STEM_data.trans(:,:,a2))))).*STEM_data.prop;   
        wave_f = fftshift((ifftn(ifftshift(wave_f))));
        STEM_data.save_wave_f(:,:,a2) = wave_f;
    end
    wave_f = fftshift(fftn((ifftshift(wave_f))));

    %STEM_data.ADF_image(x1,y1) = sum(abs(wave_f).^2.*STEM_data.detector_mask,[1,2]);
    
    STEM_data.full_4D_data{x1,y1} = abs(wave_f).^2;
    
    % load measured 4D data
    measured_4D_data = STEM_data.measured_4D_data{x1,y1}; 
    
    % calculate residual vector
    STEM_data.resi_vec = wave_f.*(1-(measured_4D_data).^(0.5)./(STEM_data.full_4D_data{x1,y1}+10^(-30)).^(0.5));
    %STEM_data.resi_vec = exp(1i*angle(wave_f)).*(abs(wave_f)-(measured_4D_data).^(0.5));

    % estimate error
    tmp_error = (measured_4D_data.^(0.5)-STEM_data.full_4D_data{x1,y1}.^(0.5)).^2;
    STEM_data.error(STEM_data.Nth_angle) = STEM_data.error(STEM_data.Nth_angle) + mean(abs(tmp_error(:)))/STEM_data.N_scan_x/STEM_data.N_scan_y;
    %STEM_data.raw_error{x1,y1} = abs(STEM_data.measured_4D_data{x1,y1}.^(0.5)-STEM_data.full_4D_data{x1,y1}.^(0.5));

    %STEM_data.error_array(x1,y1) = mean(abs(tmp_error(:)));
end

