%%

% Build up some statistics of the samples

latent_mean = settings.latent_mean;
% inds2impute = settings.inds2impute;
inds2impute = ~settings.inds_y;

sampleEvery = settings.storeEvery;
var_mean = zeros(p,p,N);
var_var = zeros(p,p,N);
var_u = zeros(p,p,N);
var_l = zeros(p,p,N);
mu_mean = zeros(p,N);
mu_var = zeros(p,N);
mu_u = zeros(p,N);
mu_l = zeros(p,N);

cov_true = true_params.cov_true;
mu_true = true_params.mu;

for tt=1:N
    theta_zeta_tt = zeros(p,k,(Niter-Nburn)/sampleEvery);
    var_tt = zeros(p,p,(Niter-Nburn)/sampleEvery);
    mu_tt = zeros(p,(Niter-Nburn)/sampleEvery);
    m = 1;
    for nn=Nburn+1:sampleEvery:Niter
        n = nn+saveEvery-1;
        if rem(n,saveEvery)==0 & n<=Niter
            filename = [saveDir '/BNP_covreg_statsiter' num2str(n) 'trial' num2str(trial) '.mat'];
            load(filename)
            store_count = 1;
        end
        theta_zeta_tt(:,:,m) = Stats(store_count).theta*Stats(store_count).zeta(:,:,tt);
        var_tt(:,:,m) = Stats(store_count).theta*Stats(store_count).zeta(:,:,tt)*Stats(store_count).zeta(:,:,tt)'*Stats(store_count).theta'...
            + diag(1./Stats(store_count).invSig_vec);
        mu_tt(:,m) = Stats(store_count).theta*Stats(store_count).zeta(:,:,tt)*Stats(store_count).psi(:,tt);
        
        m = m + 1;
        store_count = store_count + 1;
    end
    
    var_mean(:,:,tt) = mean(var_tt,3);
    var_var(:,:,tt) = var(var_tt,0,3);
    
    mu_mean(:,tt) = mean(mu_tt,2);
    mu_var(:,tt) = var(mu_tt,0,2);
    
    for pp=1:p
        for jj=pp:p
            [var_u(pp,jj,tt) var_l(pp,jj,tt)] = calculate_hpd(var_tt(pp,jj,:),0.95);
        end
        if latent_mean
            [mu_u(pp,tt) mu_l(pp,tt)] = calculate_hpd(mu_tt(pp,:),0.95);
        end
    end
    
    if ~rem(tt,100)
        display(num2str(tt))
    end
    
end



%%
LineWidth = 1.5;
fs = 20;

% Plot of posterior mean of covariance function
figure;
for pp=1:p
    for jj=pp:p
        plot(squeeze(var_mean(pp,jj,:)),'LineWidth',LineWidth); hold all;
    end
end
xlim([0 N]); ylabel('Variance/Covariance','FontSize',fs); xlabel('Time','FontSize',fs)
set(gca,'FontSize',16);
ylim([-2 2])
title('BNP Covariance Regression','Fontsize',20)


% Plot of true covariance function
figure;
for pp=1:p
    for jj=pp:p
        plot(squeeze(true_params.cov_true(pp,jj,:)),'LineWidth',LineWidth); hold all;
    end
end
xlim([0 N]); ylabel('Variance/Covariance','FontSize',fs); xlabel('Time','FontSize',fs)
set(gca,'FontSize',16);
ylim([-2 2])
title('Truth','FontSize',20)


%%
LineWidth = 1.5;

% Plot of posterior mean of mean function
figure;
for pp=1:p
    plot(mu_mean(pp,:),'LineWidth',LineWidth); hold all;
end
xlim([0 N]); ylabel('Mean','FontSize',fs); xlabel('Time','FontSize',fs)
set(gca,'FontSize',16)
title('BNP Covariance Regression','Fontsize',20)

% Plot of true mean function
figure;
for pp=1:p
    plot(true_params.mu(pp,:),'LineWidth',LineWidth); hold all;
end
xlim([0 N]); ylabel('Mean','FontSize',fs); xlabel('Time','FontSize',fs)
set(gca,'FontSize',16)
title('Truth','FontSize',20);
ylim([-3 3])


%%

fs = 20;

Sig_vec = true_params.Sig_vec;
theta = true_params.theta;

ind_start = 1;
ind_end = nmc; 
inds = [1:N];

if strcmp(reg_flag,'full')
    % Plot of Sigma
    figure;
    hl = boxplot(1./invSig_hist(:,[ind_start:ind_end])','notch','on','widths',0.75,'outliersize',2);
    for ih=1:size(hl,1)
        set(hl(ih,:),'LineWidth',2);
    end
    hold on; scatter([1:p],Sig_vec,75,'go','filled'); hold off; 
    ylabel('\Sigma_{0,p}','FontSize',20); xlabel('p','FontSize',20);
    set(gca(),'FontSize',16)
end

% Plot of covariance function hpd regions
cov_true = true_params.cov_true;
figure;
for pp=1:p
    for jj=1:pp      
        plot(squeeze(var_l(pp,jj,:)),'--','LineWidth',2); hold on;
        plot(squeeze(var_u(pp,jj,:)),'--','LineWidth',2);
        plot(squeeze(var_mean(pp,jj,:)),'LineWidth',2); plot(squeeze(cov_true(pp,jj,:)),'r','LineWidth',2);  hold off;
        xlim([0 N]); axis tight; axis square; %ylim([min(var_l(:)) max(var_u(:))]);
        box on;
        set(gca(),'XTickLabel','','XTick',zeros(1,0),'YTickLabel','','YTick',zeros(1,0),'DataAspectRatio',[10 max(abs(cov_true(pp,jj,:))) 1]);
        title(['p: ' num2str(pp) ',' num2str(jj)]);
        waitforbuttonpress;        
    end
end

% Plot of mean function hpd regions
mu_true = true_params.mu;
figure;
for pp=1:p    
    plot(mu_l(pp,:),'--','LineWidth',2); hold on;
    plot(mu_u(pp,:),'--','LineWidth',2);
    plot(mu_mean(pp,:),'LineWidth',2); plot(mu_true(pp,:),'r','LineWidth',2);  hold off;
    xlim([0 N]); axis tight; axis square; %ylim([min(var_l(:)) max(var_u(:))]);
    box on;
    set(gca(),'XTickLabel','','XTick',zeros(1,0),'YTickLabel','','YTick',zeros(1,0),'DataAspectRatio',[10 max(abs(cov_true(pp,jj,:))) 1]);
    title(['p: ' num2str(pp) ',' num2str(jj)]);
    waitforbuttonpress;
end