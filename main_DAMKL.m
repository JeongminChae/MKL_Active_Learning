
%% 2021 International Conference Machine Learning -- Distributed Active Learning with Multiple Kernels (AMKL)
clear all;close all;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Construct connectivity pattern
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_nodes = 5; 

% Network1 -- Should manually set by researcher 
A = zeros(n_nodes,n_nodes);
%A(1,2)=1;
A(1,3)=1;
A(1,4)=1;
A(1,5)=1;
%A(2,1)=1;
A(2,3)=1;
A(2,4)=1;
A(2,5)=1;
A(3,1)=1;
A(3,2)=1;
A(3,4)=1;
A(3,5)=1;
A(4,1)=1;
A(4,2)=1;
A(4,3)=1;
%A(4,5)=1;
A(5,1)=1;
A(5,2)=1;
A(5,3)=1;
%A(5,4)=1;

%%%%%%%%%%%%%%%%%
%% Load data %%%
%%%%%%%%%%%%%%%%%

% Time-series data --S&P, NASDAQ, AQI

R=readmatrix("Temperature.csv");
%R=readmatrix("NASDAQ.csv");
%R=readmatrix("S&P.csv");
%R=readmatrix("AQI_small.csv");
y=R(:,1);


% Time-series data generation
y = (y-min(y))/max(y);
T = length(y);
p = 5;
X = [];
for i = 1:T-p+1       
    X = [X y(i:i+p-1)];
end
temp_y = y(p+1:T,:);
y = temp_y;
N = size(X,1);
n_data = ceil((size(X,2)/n_nodes))-1; 





%%%%%%%%%%%%%%%%%
%% Get neighbors
%%%%%%%%%%%%%%%%%

Nei=cell(n_nodes,1);
for i=1:n_nodes
    Nei{i}=find(A(i,:));
end




%%%%%%%%%%%%%%%%%%
%% Simulation part
%%%%%%%%%%%%%%%%%%


c1=0;
cs1=10.^c1;




for i=1:length(c1)

    all_step=1/sqrt(T);
    all_beta=0.5;
    all_lambda=0.01;
    all_cplx=50;
    


    sigma = zeros(1,17);
    for j = 1:17
        sigma(j) = 10^((j-9)/2); % Create Kernel dictionary
    end

        
params=struct;
params.N=N; params.T=T;  params.ker_list={'rbf'}; n_ker=size(params.ker_list,2);
params.beta=all_beta; params.delta=0.1;   params.eta=all_step; params.w_ini=0*ones(1,n_ker); 
params.lambda=all_lambda;   params.sigma= sigma; params.L=all_cplx;  params.theta_ini=ones(length(params.sigma),1)/length(params.sigma);
params.S=2; params.n_nodes = n_nodes; params.n_data = n_data; params.sigma_REF_DOKL=1000;
params.eta_g = 1/sqrt(100); % Weight update 
params.eta_g_a = 1/sqrt(10*T);
params.eta = sqrt(100); 
params.rho= sqrt(10000);
params.eta_l= 1/sqrt(10*T);
params.niter=5;
params.M=1;
params.eta_c=0.005;
params.eta_c_a=0.005;


 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DOMKL with random feature approximation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    [theta_nodes_o,er_hat_nodes_o,Consensus_cons_nodes_o,erm_mean_nodes1_o,erm_mean_nodes2_o,erm_mean_nodes3_o,erm_mean_nodes4_o,erm_mean_nodes5_o]=Domkl_online_rf(y,X,params,A);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DAMKL with random feature approximation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
   
    [theta_nodes_a,er_hat_nodes_a,Consensus_cons_nodes_a,Eff_all_nodes_a,erm_mean_nodes1_a,erm_mean_nodes2_a,erm_mean_nodes3_a,erm_mean_nodes4_a,erm_mean_nodes5_a]=Damkl_online_rf(y,X,params,A);


end




%%%%%%%%%%%%%%%
%% Print results
%%%%%%%%%%%%%%%


fprintf('Domkl_MSE_mean=%f\n', (erm_mean_nodes1_o(end)+erm_mean_nodes2_o(end)+erm_mean_nodes3_o(end)+erm_mean_nodes4_o(end)+erm_mean_nodes5_o(end))/n_nodes);
fprintf('Damkl_MSE_mean=%f\n', (erm_mean_nodes1_a(end)+erm_mean_nodes2_a(end)+erm_mean_nodes3_a(end)+erm_mean_nodes4_a(end)+erm_mean_nodes5_a(end))/n_nodes);



% Consensus 
Consensus_cons_tot=0;
Consensus_cons_tot_a=0;

for i=1:n_nodes
    Consensus_cons_o=Consensus_cons_nodes_o{i};
    Consensus_cons_a=Consensus_cons_nodes_a{i};
    Consensus_cons_tot= Consensus_cons_tot +Consensus_cons_o;
    Consensus_cons_tot_a= Consensus_cons_tot_a +Consensus_cons_a;
end


fprintf('CV_mean_Domkl=%f\n', sum(Consensus_cons_tot)/(((n_data)*(params.niter)*(n_nodes))));
fprintf('CV_mean_Damkl=%f\n', sum(Consensus_cons_tot_a)/((n_data)*(params.niter)*(n_nodes)));

    
Eff_tot =0;
for i=1:n_nodes
    Eff_all=Eff_all_nodes_a{i};
    Eff_tot = Eff_tot + Eff_all;
end
Eff_mean=Eff_tot/((params.niter)*(n_nodes));    
fprintf('Efficiency_mean_Damkl=%f\n', (n_data-sum(Eff_mean))/n_data);

