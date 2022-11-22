function probe_wf = Func_generate_probe_wave_scan_sub_pixel(probe_wf0,x1,y1)
    
    cent=round((size(probe_wf0)+1)/2);   
    
    scx=cent(1);
    scy=cent(2);

    dx = x1-scx;
    dy = y1-scy;

    ny = size(probe_wf0,1);
    nx = size(probe_wf0,2);
    
    [Y, X] = meshgrid(-nx/2:nx/2-1,-ny/2:ny/2-1);
    %F = fftshift(ifftn(ifftshift(probe_wf0)));
    %Pfactor = exp(2*pi*i*(dx*X/nx + dy*Y/ny));
    %probe_wf = fftshift(fftn(ifftshift(F.*Pfactor)));


    F = fftn(probe_wf0);
    Pfactor = single(exp(-2*pi*1i*(dx*X/nx + dy*Y/ny)));
    probe_wf = ifftn((F.*ifftshift(Pfactor)));


end