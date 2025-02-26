clear all
clc
close all

% add the executor part of the code here
incompleteData=[-0.4490   -1.7064       NaN   -0.6148   -1.5419
                                0.1487   -0.9518    0.2673    0.0989   -0.5542
                               -0.1798   -1.2944   -0.3253       NaN   -0.3821
                                    NaN   -1.6605   -0.4925   -0.3972   -1.1671
                               -0.0854       NaN   -0.0069   -0.4036   -1.5815
                                0.1132   -2.3902    0.0413   -0.4059   -1.8628
                               -0.8854   -1.2259   -1.0160   -0.1432   -0.3691
                                0.1812   -1.5490    0.1469    0.1501   -0.7279
                               -0.5348   -1.5431   -0.6250       NaN   -0.7101
                               -0.3532   -1.2494   -0.3006   -0.5558   -1.2585
                                0.0729   -1.9973    0.0781   -0.1438   -1.2745
                               -0.4013   -1.0881   -0.5758   -0.0776   -0.3776
                               -0.7348   -0.6456   -0.9468   -0.3309   -0.2594
                               -0.3911   -0.7800   -0.5576    0.0045   -0.1129
                                0.2652   -1.2685    0.3553    0.0129   -0.7856
                               -0.3205   -2.3581       NaN   -0.1445   -1.2682
                               -0.4158       NaN       NaN   -0.3779   -0.8805
                               -0.8292   -1.3634   -1.2016   -0.4153   -0.6297
                                0.2795   -1.4995       NaN   -0.2296   -1.2453
                                   NaN   -0.9969    0.2460    0.1050   -0.5895];

    [estimatedBlock] = pca_missing_data_estimator(incompleteData)


function [estimatedBlock] =pca_missing_data_estimator(Incomplete_Data)
%%% This function receives the pca model, a new observation for which some
%%% values are missing (denoted by nan) and provides the estimation for the missing columns of data

%% put nan for all missing data
    Incomplete_Data(~isfinite(Incomplete_Data)) = NaN;

%% Seperating Complete rows and Incompleterows
            
    incomplet_idx = find(any(isnan(Incomplete_Data), 2));
    incomBlock=Incomplete_Data(incomplet_idx,:);

    CompleteBlock=Incomplete_Data;
    CompleteBlock(incomplet_idx,:)=[];

%% create PCA using the complete block
    Num_com=Num_Com_determination(CompleteBlock);
    pca_model=pca_nipals(CompleteBlock,Num_com,0.95);
    
    estimatedBlock=zeros(size(incomBlock));
    
    for i=1:size(incomBlock,1)
        new_obs=incomBlock(i,:);
        [~,available_col]=find(isnan(new_obs(1,:))==false);
    
        new_obs=scaler_pca(pca_model,new_obs);
        P=pca_model.P;
    
        x_new=new_obs(:,available_col);
        p_new=P(available_col,:);
        t_new=((x_new*p_new)/(p_new'*p_new));

         x_estimated=t_new*P';
                
        estimatedBlock(i,:)=unscaler_pca(pca_model,x_estimated);
        estimatedBlock(i,available_col)=incomBlock(i,available_col);
    end
              
    end
    
    function mypca=pca_nipals(data,Num_com,alfa)
    
    %%% receives data (in original format), the number of required
    %%% components, alfa and return a PCA model including P, T, Rsquared, 
    %%% x_hat, t_squared,SPE, tsquared_lim and SPE_lim
            
            %% pre-processing
            data_origin=data;
            [~,var_rank]=max(var(data_origin));
            Num_obs=size(data,1);
            Cx=mean(data);
            Sx=std(data)+1e-16;
            data=(data-Cx)./Sx;
            X=data;
            P=zeros(size(X,2),Num_com);
            T=zeros(size(X,1),Num_com);
            SPE=zeros(size(T));
            Rsquare=zeros(1,Num_com);
            SPE_lim=zeros(1,Num_com); 
            tsquared=zeros(Num_obs,Num_com);
            T2_lim=zeros(1,Num_com);
            ellipse_radius=zeros(1,Num_com);
            covered_var=zeros(1,Num_com);
    
    
            
            %% Nipals Algorithem
            
            for i=1:Num_com
            
                b=var_rank;
                t1=X(:,b);
                
                while true
         
                    P1=(t1'*X)/(t1'*t1);
                    P1=P1./norm(P1);
                    tnew=((P1*X')/(P1*P1'))'; 
    
                    told=t1;
                    t1=tnew;
                    
                    E=tnew -told;
                    E=E'*E;
                    if E<1e-15
                        break
                    end
                end
                
                xhat=t1*P1;
                Enew=X-xhat;
                X=Enew;
             
                P(:,i)=P1;
                T(:,i)=t1;
                covered_var(i)=var(t1);
    
                % SPE
                [SPE(:,i),SPE_lim(i),Rsquare(i)]=SPE_calculation(T, P,data,alfa);
    
                %T2
                [tsquared(:,i), T2_lim(i),ellipse_radius(i)]=T2_calculations(T(:,1:i),i,Num_obs,alfa);
            
            end
    
           
    %% Function output
            mypca.P=P;
            mypca.T=T;
            mypca.Rsquare=Rsquare;
            mypca.covered_var=covered_var;
            mypca.X_hat=T*P';
            mypca.tsquared=tsquared;
            mypca.T2_lim=T2_lim;
            mypca.ellipse_radius=ellipse_radius;
            mypca.SPE_x=SPE;
            mypca.SPE_lim_x=SPE_lim;
            mypca.x_scaling=[Cx;Sx];
            mypca.Xtrain_normal=data_origin;
            mypca.Xtrain_scaled=data;
            mypca.alfa=alfa;
    end
    
    function scaled_point=scaler_pca(pca_model,un_scaled_point)
    %%% receive unscaled point and pca model and scales the point
    
            Cx=pca_model.x_scaling(1,:);
            Sx=pca_model.x_scaling(2,:);
    
            scaled_point=(un_scaled_point-Cx)./Sx;
    end
    
    function un_scaled_point=unscaler_pca(pca_model,scaled_point)
    %%% receive scaled point and pca model and unscales the point
    
            Cx=pca_model.x_scaling(1,:);
            Sx=pca_model.x_scaling(2,:);
    
            un_scaled_point=(scaled_point.*Sx)+Cx;        
    end

    function [spe,spe_lim,Rsquare]=SPE_calculation(score, loading,Original_block,alfa)

%%% receive score,loading, original block (scaled format) and alfa, and calculate the Error
%%% and SPE and the SPE_lim as well as Rsquared

            X_hat=score*loading';
            Error=Original_block-X_hat;
            spe=sum(Error.*Error,2);
            m=mean(spe);
            v=var(spe);
            spe_lim=v/(2*m)*chi2inv(alfa,2*m^2/v);

            %Rsquared
            Rsquare=(1-var(Error)/var(Original_block));
             
    end

    function [tsquared, T2_lim,ellipse_radius]=T2_calculations(T,Num_com,Num_obs,alfa)

%%% recieve Score Matrix, the current applied number of components,num of
%%% observations and alfa and return all Hotelling T2 related calculations
%%% including tsquared, T2_lim and ellipse_radius
            tsquared=sum((T./std(T)).^2,2);
            T2_lim=(Num_com*(Num_obs^2-1))/(Num_obs*(Num_obs-Num_com))*finv(alfa,Num_com,(Num_obs-Num_com));
            ellipse_radius=(T2_lim*std(T(:,Num_com))^2)^0.5;

end
function [Num_com]=Num_Com_determination(X)

%%% recieve the blocks (not scaled and not centered format) and calculate the true rank for both blocks based on
%%% the rule covered var<1
%%% Num component cannot be more than (Num_Obs-1)

            Num_Obs=size(X,1);
            max_num_com=min(Num_Obs-1,size(X,2));
            mypca=pca_nipals(X,max_num_com,0.95);
            var_covered=mypca.covered_var;

            Num_com=sum(var_covered>1);
            Num_com=min(Num_com,Num_Obs-1);
end