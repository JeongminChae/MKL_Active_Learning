

function [theta , er_sel, kernel_loss, chosen_index,Consensus_cons]=omkl_rs_final(y,x,params, theta, kernel_loss, D, frequent_index,nei,theta_all)
N=params.N;  ker_list=params.ker_list;   beta=params.beta; delta=params.delta;
eta_g=params.eta_g;sigma=params.sigma; L=params.L;
lambda=params.lambda;
eta_l=params.eta_l;


n_ker=length(sigma);


   for i=1:n_ker
             
        theta0=theta(:,i);
        kx0=[sin(D{i}*x); cos(D{i}*x)];
        fx(i)=kx0'*theta0; %combination of kernel functions
        
        
        for n=1:length(nei)
            theta_nei0 =theta_all(:,:,nei(n));
            n_kx0=[sin(D{i}*x); cos(D{i}*x)];
            n_fx(n)=n_kx0'*theta_nei0;   
            %n_fx_nodes{nei(n)}=n_fx;
        end
        
        
        er_temp(i,1)=(y-fx(i))^2; %LOSS_TEMP --> theta
        grad=-2*(y-fx(i))*kx0; %GRADIENT
        theta_temp= theta0 - (eta_l)*grad - 2*lambda*(eta_l)*theta0; %OGD
        theta0=theta_temp;
        theta(:,i)=theta0;
        
        dis_all=0;
        for n=1:length(nei)
            dis=((fx(i)-n_fx(n)).^2);
            dis_all=dis_all+dis;
        end
        
        Consensus_cons=dis_all;
        
        
        
   end
    
   
    
    %PROBABILITY DISTRIBUTION
    

    prob = kernel_loss/sum(kernel_loss);
    
    
 
    
    %CHOOSE ONE INDEX ACCORDING TO THE ABOVE PROBABILITY
   
    temp_a = rand;
    
    for tt=1:n_ker
        if sum(prob(1:tt)) > temp_a
            chosen_index = tt;
            break;
        end  
    end
    
   
    
   % [temp_value, chosen_index] = max(prob);
    
 
    
     f_hat_select = fx(frequent_index); %COMMON_MODE (CHOSEN AT THE CENTRAL SERVER)
         
     kernel_loss = kernel_loss.*exp(-eta_g*er_temp);
     
     
    
    
    if max(kernel_loss)>1e12
        kernel_loss=kernel_loss/sum(kernel_loss);
    end
    


    %CURRENT LOSS
    er_sel= (y-f_hat_select)^2;


    
end