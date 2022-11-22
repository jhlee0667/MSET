%J.Lee, KAIST, 2021

function range=Func_raster_range(center,mm,pp,size,step)

if(nargin<5)
   step = 1;
end

if center-mm<1
    range=[mod(center-mm-1,size):size 1:center+pp];
elseif center+pp>size
    range=[center-mm:size 1:mod(center+pp,size)];
else
    range=center-mm:step:center+pp;
end
    
end