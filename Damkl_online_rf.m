

function [theta_nodes,er_hat_nodes,Consensus_all_nodes,Eff_all_nodes,erm_mean_nodes1,erm_mean_nodes2,erm_mean_nodes3,erm_mean_nodes4,erm_mean_nodes5]=Damkl_online_rf(y,X,params,Connectivity)
N=params.N; T=params.T; ker_list=params.ker_list;   beta=params.beta; delta=params.delta;
sigma=params.sigma; L=params.L; n_ker=length(sigma);
lambda=params.lambda; n_nodes = params.n_nodes;

%parameter
eta=params.eta; 
eta_g=params.eta_g; 
rho=params.rho;
lambda=params.lambda;
M=params.M;
niter =params.niter;
eta_c=params.eta_c;


w_nodes = cell(n_nodes,n_ker); 
lambda_nodes = cell(n_nodes,n_ker);
er_nodes = cell(n_nodes,1);
er_hat_nodes = cell(n_nodes,1);
erm_nodes=cell(n_nodes,1);
erm_all_nodes=cell(n_nodes,1);
Eff_all_nodes=cell(n_nodes,1);
Consensus_all_nodes=cell(n_nodes,1);
for i=1:n_nodes
    erm_all_nodes{i}=zeros(1,ceil((size(X,2)/n_nodes))-1);
    Eff_all_nodes{i}=zeros(1,ceil((size(X,2)/n_nodes))-1);
    Consensus_all_nodes{i}=zeros(1,ceil((size(X,2)/n_nodes))-1);
end
erm_mean_nodes=cell(n_nodes,1);
fx_nodes = cell(n_nodes,1);
theta_nodes = cell(n_nodes,1);

%% Neighbor

Nei = cell(n_nodes,1);
for n = 1:n_nodes
   Nei{n} = find(Connectivity(n,:)==1);
end

n_data = ceil((size(X,2)/n_nodes))-1; % n_data per node




%% Distribute data to each node

B_X = cell(n_nodes,1);
B_y = cell(n_nodes,1);

for i =1:n_nodes
    B_X{i}=zeros(size(X,1),n_data);
    B_y{i}=zeros(n_data,1);
end



for d = 1:n_data
for i =1:n_nodes
    B_X_temp = B_X{i};
    B_y_temp = B_y{i};
    B_X_temp(:,d) = X(:,1);
    B_y_temp(d) = y(1);
    X(:,1)=[];
    y(1)=[];
    B_X{i} = B_X_temp;
    B_y{i} = B_y_temp;
end
end
    
    
n_fx_nodes=cell(n_nodes,n_nodes);
q_nodes=cell(n_nodes);
Consensus_cons_nodes=cell(n_nodes);
n_f_hat_nodes=cell(n_nodes,n_nodes);



for iter = 1:niter

   
    
    

W_int=zeros(2*L,1);
lambda_int = zeros(2*L,1);
n_ker=length(sigma);
D=cell(n_nodes,n_ker);
theta_ini=params.theta_ini;
for i=1:n_nodes
    theta_nodes{i}=theta_ini;
end

%% Random vector generation

for j = 1:n_nodes
    for i=1:n_ker
     D{j,i}=sigma(i)*randn(L,N);
    end
end

C=ceil(L/N)-1;
resd=L-C*N;
for j = 1:n_nodes
    for i=1:n_ker
        for c=1:C+1
            G=randn(N,N);
            [Q,R]=qr(G);
            if c<C+1
                D_temp(c*N-N+1:c*N,:)=sigma(i)*Q;
            else
                D_temp(c*N-N+1:c*N-N+resd,:)=sigma(i)*Q(1:resd,:);     
            end
        end
        v1=chi2rnd(ones(L,1));
     D{j,i}=diag(sqrt(v1))*D_temp;
    end
end


    
f_hat_nodes=cell(n_nodes,1);
for i=1:n_nodes
    f_hat_nodes{i}=zeros(n_data,1);
end

for i=1:n_nodes
    for j=1:n_ker
        pre_w_nodes{i,j}=W_int;
    end
end

%% Efficiency check
Eff_nodes = cell(n_nodes,1);
eta_c_new_nodes = cell(n_nodes,1);
for i=1:n_nodes
    Eff_nodes{i}=zeros(1,n_data);
    eta_c_new_nodes{i}=eta_c;
end
    



%% Algorithm

for t=1:n_data
    

        
    
    for i = 1:n_nodes
        
        X_temp = B_X{i};
        y_temp = B_y{i};
        nei=Nei{i};
        Eff_node = Eff_nodes{i};
        q=q_nodes{i};
        fx=fx_nodes{i};
        eta_c_new=eta_c_new_nodes{i};
        
        x = X_temp(:,t);
        
        if t>0.1*n_data
            value_c = Confidence_condition(q,fx'); 
        else
            value_c=100;
        end
        

            
        
        if value_c<100 && Eff_node([t-1])==1  
            if sum(Eff_node([t-2:t-1]))==1
                eta_c_new=eta_c;
                eta_c_new = eta_c_new*(0.5);
            else
                eta_c_new = eta_c_new*(0.5);
            end
            
            eta_c_new_nodes{i}=eta_c_new;
            
            if value_c < eta_c_new
                Eff_node(t)=1;
            end
        else
            if value_c < eta_c
                Eff_node(t)=1;
            end
        end
            
        
        for j = 1:n_ker
            if t==1
                w0 = W_int;
                lambda = lambda_int;
            else
                w0 = w_nodes{i,j};
                lambda = lambda_nodes{i,j};
            end
            
            if Eff_node(t)==1
                w_nodes{i,j}=w0;
                er_temp(j,1)=0;
                
                for n=1:length(nei)
                    n_fx=n_fx_nodes{i,nei(n)};
                    if t==1
                        n_w0 = W_int;
                    else
                        n_w0 = w_nodes{nei(n),j};
                    end
                    n_kx0=[sin(D{nei(n),j}*x); cos(D{nei(n),j}*x)];
                    n_fx(j)=n_kx0'*n_w0;   
                    n_fx_nodes{i,nei(n)}=n_fx;
                end
                
                            
              
                
                
                nei = Nei{i};
                pre_w_sum = 0;
                for n = 1:length(nei)
                    pre_w_sum = pre_w_sum + pre_w_nodes{nei(n),j};
                end
                
                Inv= inv(((eta +rho*numel(Nei{i}))*eye(size(kx0,1))));
             
                w_temp = Inv*(eta*w0 - lambda + rho*(((numel(Nei{i})*w0)+pre_w_sum)/2)); 
                w0=w_temp;
                w_nodes{i,j}=w0;
                
                
            else
                
                kx0=[sin(D{i,j}*x); cos(D{i,j}*x)];
                fx(j)=kx0'*w0;
            
            
                for n=1:length(nei)
                    n_fx=n_fx_nodes{i,nei(n)};
                    if t==1
                        n_w0 = W_int;
                    else
                        n_w0 = w_nodes{nei(n),j};
                    end
                    n_kx0=[sin(D{nei(n),j}*x); cos(D{nei(n),j}*x)];
                    n_fx(j)=n_kx0'*n_w0;   
                    n_fx_nodes{i,nei(n)}=n_fx;
                end
                
                

            
           
            
                % theta update
                er_temp(j,1)=(y_temp(t)-fx(j))^2;
                nei = Nei{i};
                pre_w_sum = 0;

                for n = 1:length(nei)
                    pre_w_sum = pre_w_sum + pre_w_nodes{nei(n),j};
                end
                    
                Inv= inv(2*kx0*kx0' + ((eta +rho*numel(Nei{i}))*eye(size(kx0,1))));
           
                w_temp = Inv*(2*y_temp(t)*kx0 + eta*w0 - lambda + rho*(((numel(Nei{i})*w0)+pre_w_sum)/2));                                  
                w0=w_temp;
                w_nodes{i,j}=w0;  
            
            end
                  
    
            er_nodes{i}=er_temp;
            fx_nodes{i}=fx;
            Eff_nodes{i}=Eff_node;
        
        end
    end
    
    pre_w_nodes=w_nodes;
        
    %% Lambda update
    for i = 1:n_nodes
        for j=1:n_ker
            if t==1
                lambda = lambda_int;
            else
                lambda = lambda_nodes{i,j};
            end
                
            w0 = w_nodes{i,j};
            nei = Nei{i};
            nei_w_sum = 0;
            for n = 1:length(nei)
                nei_w_sum = nei_w_sum + w_nodes{nei(n),j};
            end

            % Regression -lambda update
       
            lambda_temp = lambda + (rho/2)*((length(nei)*w0)-nei_w_sum);
            lambda_nodes{i,j} = lambda_temp;

        end
    end
    
        
    %% Receive weight from neighbors and get f_hat
    for i=1:n_nodes
        nei = Nei{i};
        if t==1
            theta=theta_ini;
        else
            theta=theta_nodes{i};
        end
        
        fx=fx_nodes{i};
        f_hat=f_hat_nodes{i};
        theta_nei=1;
        
        for n=1:length(nei)
            theta_nei=theta_nei.*theta_nodes{nei(n)};
        end
        
        theta_up=theta.*theta_nei;
        q=(theta_up/sum(theta_up));
        f_hat(t)=fx*q;
        q_nodes{i}=q;
        f_hat_nodes{i}=f_hat;
    end
    
    
        
    
    %% Consensus
    for i=1:n_nodes
        nei = Nei{i};
     
        f_hat=f_hat_nodes{i};
        dis_all=0;
        Consensus_cons=Consensus_cons_nodes{i};
        for n=1:length(nei)
            n_f_hat=n_f_hat_nodes{i,nei(n)};
            n_fx=n_fx_nodes{i,nei(n)};
            q=q_nodes{nei(n)};

            n_f_hat(t)=n_fx*q;            
            n_f_hat_nodes{i,nei(n)}=n_f_hat;
            dis=((f_hat(t)-n_f_hat(t)));
            dis_all=dis_all+dis;
        end
        
        Consensus_cons(t)=(dis_all/length(nei)).^2;
        Consensus_cons_nodes{i}=Consensus_cons;
    end
    
    
    
    
    
    
   
    %% Wight update with the loss of my node
    for i=1:n_nodes
        theta=theta_nodes{i};
        er_temp=er_nodes{i};
        theta = theta.*exp(-eta_g*er_temp);
        theta;
        theta_nodes{i}=theta;
    end
    
    
    

    
    for i=1:n_nodes
        y_temp = B_y{i};
        f_hat = f_hat_nodes{i};
        er = er_hat_nodes{i};
        erm = erm_nodes{i};
    
        
        if t==1
            er(1)=1;
        else
            er(t)=(y_temp(t)-f_hat(t))^2;
        end
        

        erm(t)= mean(er);
        
        erm(t)
        
        er_hat_nodes{i}=er;
        erm_nodes{i}=erm;
    end
    

    
end

for i=1:n_nodes
    Eff_node = Eff_nodes{i};
    Eff_all = Eff_all_nodes{i};
    Eff_all=Eff_all+Eff_node;
    Eff_all_nodes{i}=Eff_all;
end


for i=1:n_nodes
    erm = erm_nodes{i};
    erm_all=erm_all_nodes{i};
    erm_all = erm_all + erm;
    erm_all_nodes{i}=erm_all;
end

for i=1:n_nodes
    Consensus = Consensus_cons_nodes{i};
    Consensus_all=Consensus_all_nodes{i};
    Consensus_all = Consensus_all + Consensus;
    Consensus_all_nodes{i}=Consensus_all;
end
  
              
   
for i=1:n_nodes
    erm_all=erm_all_nodes{i};
    erm_mean = erm_all/iter; 
    erm_mean_nodes{i} = erm_mean;
    
    erm_mean_nodes1=erm_mean_nodes{1};
    erm_mean_nodes2=erm_mean_nodes{2};
    erm_mean_nodes3=erm_mean_nodes{3};
    erm_mean_nodes4=erm_mean_nodes{4};
    erm_mean_nodes5=erm_mean_nodes{5};


end



end
        