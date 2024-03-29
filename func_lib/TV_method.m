% Author: J.Lee, KAIST (Korea), 2022.
% Y.Yang, Multi-Dimensional Atomic Imaging Lab
% NCAET simulation package

function [output, error] = TV_method(input, lambda, Niter)

    if length(size(input)) == 1
        Ndims = 1;
        tau = 1/4;
    elseif length(size(input)) == 2
        Ndims = 2;
        tau = 1/8;
    elseif length(size(input)) == 3
        Ndims = 3;
        tau = 1/12;
    end
    
    P = zeros([size(input),length(size(input))]);

    for i = 1:Niter
        temp_term = tau .* TV_del(TV_div(P, Ndims)-input/lambda, Ndims);
        %P = (P + temp_term)./(1 + abs(temp_term));
        P = TV_project(P + temp_term, Ndims);
    end

    output = input - lambda*TV_div(P, Ndims);
    error = lambda * TV_error(output, Ndims);
end


%%
function P_out=TV_project(P_in, Ndims)
    if Ndims==1
        P_out=P_in./sqrt(max(1,abs(P_in).^2));
    elseif Ndims==2
        P_out=P_in./sqrt(max(1,abs(P_in(:,:,1)).^2 + abs(P_in(:,:,2)).^2));
    elseif Ndims==3
        P_out=P_in./sqrt(max(1,abs(P_in(:,:,:,1)).^2 + abs(P_in(:,:,:,2)).^2 + abs(P_in(:,:,:,3)).^2));
    end
end


function P_out=TV_div(P_in, Ndims)
    if Ndims==1
        P_out=P_in-circshift(P_in,1);
    elseif Ndims==2
        P_out=P_in(:,:,1) + P_in(:,:,2);
        P_out=P_out-circshift(P_in(:,:,1),[1 0]);
        P_out=P_out-circshift(P_in(:,:,2),[0 1]);
    elseif Ndims==3
        P_out=P_in(:,:,:,1) + P_in(:,:,:,2) + P_in(:,:,:,3);
        P_out=P_out-circshift(P_in(:,:,:,1),[1 0 0]);
        P_out=P_out-circshift(P_in(:,:,:,2),[0 1 0]);        
        P_out=P_out-circshift(P_in(:,:,:,3),[0 0 1]); 
    end
end

function P_out=TV_del(P_in, Ndims)
    if Ndims==1
        P_out=-P_in+circshift(P_in,-1);
    elseif Ndims==2
        P_out(:,:,1)=-P_in+circshift(P_in,[-1 0]);
        P_out(:,:,2)=-P_in+circshift(P_in,[0 -1]);
    elseif Ndims==3
        P_out(:,:,:,1)=-P_in+circshift(P_in,[-1 0 0]);
        P_out(:,:,:,2)=-P_in+circshift(P_in,[0 -1 0]);        
        P_out(:,:,:,3)=-P_in+circshift(P_in,[0 0 -1]); 
    end
end

function error=TV_error(P_in, Ndims)
    P_out=TV_del(P_in, Ndims);
    if Ndims==1
        error=sum(sqrt(abs(P_out(:)).^2));

    elseif Ndims==2
        error=sum(sqrt(abs(P_out(:,:,1)).^2+abs(P_out(:,:,2)).^2),"all");

    elseif Ndims==3
        error=sum(sqrt(abs(P_out(:,:,:,1)).^2+abs(P_out(:,:,:,2)).^2+abs(P_out(:,:,:,3)).^2),"all");

    end
end

