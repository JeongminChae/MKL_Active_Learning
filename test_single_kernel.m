clear all;
%close all;
clc

load TwitterData
%load TomData
%load AirData

%R = readmatrix("navalplantmaintenance.csv");
%X = R(1:11934,1:16);
%X = X';
%y = R(1:11934,18);


%KERNEL GENERATION


for i=1:17
    
    kernel_temp(i) = 10^((i-9)/2);  
end


T=length(y);
N=size(X,1);
 
c1=1;%[-1:3];
 c1=0;
 cs1=10.^c1;
%cs1=1;

for i=1:length(c1)

    all_step=1/sqrt(T);%cs1(i)/sqrt(T);
    all_beta=0.5;
    % all_lambda=st*T;
    all_lambda=0.01;
    all_cplx=50;
    
 %%%%%%%%%%%%%%%%%%
%% Online Single Kernels
%%%%%%%%%%%%%%%%%%%%%
        
params2=struct;
params2.N=N; params2.T=T;  params2.ker_list={'poly'}; n_ker=size(params2.ker_list,2);
params2.beta=all_beta; params2.delta=0.1;   params2.eta=all_step; params2.w_ini=zeros(1,n_ker); params2.theta_ini=ones(n_ker,1)/n_ker;
params2.lambda=all_lambda;        params2.sigma=[1];  params2.p=[2];

tic      
    [f_rbf, theta,er_rbf,erm_rbf]=mkl_online(y,X,params2);
t_rbf=toc;  
E_rbf(i)=erm_rbf(end);



params3=struct;
params3.N=N; params3.T=T;  params3.ker_list={'poly'}; n_ker=size(params3.ker_list,2);
params3.beta=all_beta; params3.delta=0.1;   params3.eta=all_step; params3.w_ini=zeros(1,n_ker); params3.theta_ini=ones(n_ker,1)/n_ker;
params3.lambda=all_lambda;         params3.sigma=[ 10];  params3.p=[3];

tic;      
    [f_rbf10, theta_rbf10,er_rbf10,erm_rbf10]=mkl_online(y,X,params3);        
t_lin=toc;
E_rbf10(i)=erm_rbf10(end);


params4=struct;
params4.N=N; params4.T=T;  params4.ker_list={'lin'}; n_ker=size(params4.ker_list,2);
params4.beta=all_beta; params4.delta=0.1;   params4.eta=all_step; params4.w_ini=zeros(1,n_ker); params4.theta_ini=ones(n_ker,1)/n_ker;
params4.lambda=all_lambda;   params4.sigma=[0.1];       params4.p=[0];

tic;      
    [f_rbf01, theta_rbf01,er_rbf01,erm_rbf01]=mkl_online(y,X,params4);     
t_poly=toc;      
E_rbf01(i)=erm_rbf01(end);






end




    fprintf('RBF01=%f\n',  min(E_rbf01));
    fprintf('RBF1=%f\n',  min(E_rbf));
    fprintf('RBF10=%f\n',  min(E_rbf10));
   % fprintf('OMKL=%f\n',  min(E_mkl));
   % fprintf('OMKL-Avg=%f\n',  min(E_mkl_avg));
%    fprintf('AdaMKL=%f\n',  min(E_adap));
   % fprintf('OMKL-B=%f\n', min(E_mkl_bg));
   % fprintf('Raker=%f\n', min(E_raker));
   % fprintf('AdaRaker=%f\n\n', min(E_adaraker));

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot figures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
figure(1)
semilogy(1:T,erm_rbf,'-',1:T,erm_rbf10,'-',1:T,erm_rbf01,'-','linewidth',2)
legend('POLY2','POLY3','LINEAR')
% 
% figure(3)
% tt=[ t_adap t_mkl t_rbf t_lin t_poly t_adaprf t_rf];
% tt=tt/max(tt);
% %xtlab={'AdapMKL';'OMKL';'RBF';'Lin';'Poly';'AdapRF';'RF'};
% xtlab={'AdapMKL';'OMKL';'RBF1';'RBF10';'RBF01';'AdapRF';'RF'};
% bar(1:7,tt, 'BarWidth',0.7);     
% 
% set(gca,'XTickLabel',xtlab)
% 
% 
% 
% % 
% % figure(4)
% % semilogy(1:T,erm_adap,'-',1:T,erm_mkl,'-',1:T,erm_rbf,'-',1:T,erm_rbf10,'-',1:T,erm_rbf01,'-',1:T,erm_rf,'-',1:T,erm_rf_mkl,'-','linewidth',2)
% % legend('AdapMKL','OMKL','RBF1','RBF10','RBF01','AdapRF')
% 
% % 
% figure(5)
% semilogy(1:T,erm_adap,'-',1:T,erm_mkl,'-',1:T,erm_mkl_bg,'-',1:T,erm_rbf,'-',1:T,erm_rbf10,'-',1:T,erm_rbf01,'-',1:T,erm_rf,'--',1:T,erm_rf_mkl,'--','linewidth',2)
% legend('AdaMKL','OMKL','OMKL-B','KL-RBF(\sigma^2=1)','KL-RBF(\sigma^2=10)','KL-RBF(\sigma^2=0.1)','AdaRaker','Raker')
% 
% 
% 
% 
% figure(8)
% tt=[t_adap t_mkl t_mkl_bg t_rbf t_lin t_poly t_adaprf t_rf  ];
% tt=tt/max(tt);
% %xtlab={'AdapMKL';'OMKL';'RBF';'Lin';'Poly';'AdapRF';'RF'};
% xtlab={'AdaMKL';'OMKL';'OMKL-B';'RBF1';'RBF10';'RBF01';'AdaRaker';'Raker'};
% bar(1:8,tt, 'BarWidth',0.6);      
% 
% set(gca,'XTickLabel',xtlab)  
% % mean(y.^2)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Strongly adaptive OMKL with random feature approximation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
%params5=struct;
%params5.N=N; params5.T=T;  params5.ker_list={'rbf'}; n_ker=size(params5.ker_list,2);
%params5.beta=all_beta; params5.delta=0.1;   params5.eta=all_step; params5.w_ini=0*ones(1,n_ker); 
%params5.lambda=all_lambda;   params5.sigma=[ 0.1 1 10]; params5.L=all_cplx;  params5.theta_ini=ones(length(params5.sigma),1)/length(params5.sigma);
%params5.S=2;


   % tic;
  %  [f_rf,theta_rf,w_hat,er_rf,erm_rf]=mkl_adap_online_rf(y,X,params5);        
  %  t_adaprf=toc;
  %  E_adaraker(i)=erm_rf(end);
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% OMKL with random feature approximation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

  %  tic;
  %  [f_mkl_rf, theta_omkl_rf,er_rf_mkl,erm_rf_mkl]=mkl_online_rf(y,X,params5);
  %  t_rf=toc;        
        
   %E_raker(i)=erm_rf_mkl(end);
