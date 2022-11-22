% J.Lee, KAIST, 2022

function OV_vol = Func_real_intepolation_OV_vector(Vol,OV_vector)

    volsize=length(Vol);

    OVx = OV_vector(1);
    OVy = OV_vector(2);

    OVz = OV_vector(3);
    %
    OV_low_limit_x = round((OVx-1)/2);
    OV_high_limit_x = round(OVx/2)-1;
    OV_low_limit_y = round((OVy-1)/2);
    OV_high_limit_y = round(OVy/2)-1;
    OV_low_limit_z = round((OVz-1)/2);
    OV_high_limit_z = round(OVz/2)-1;

    %
    [X,Y,Z] = meshgrid(1:volsize,1:volsize,1:volsize);
    OV_list_x = 1-OV_low_limit_x/OVx:1/OVx:volsize+OV_high_limit_x/OVx;
    OV_list_y = 1-OV_low_limit_y/OVy:1/OVy:volsize+OV_high_limit_y/OVy;
    OV_list_z = 1-OV_low_limit_z/OVz:1/OVz:volsize+OV_high_limit_z/OVz;

    [Xq,Yq,Zq] = meshgrid(OV_list_x, OV_list_y, OV_list_z);
    
    %OV_vol = interp3(X,Y,Z,Vol,Xq,Yq,Zq,'linear');
    %OV_vol = interp3(X,Y,Z,Vol,Xq,Yq,Zq,'cubic');
    OV_vol = splinterp3(double(Vol),Xq,Yq,Zq);
    OV_vol(isnan(OV_vol))=0;
    

end