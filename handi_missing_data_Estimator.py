import numpy as np
from MyPcaClass import MyPca as pca
import diffrencepy as dis

def estimate_missing_data(Z:np.ndarray=None,comple_data:np.ndarray=None):
        '''
        This function receives one block of data in which each observation is in one row
         data includes all rows without missing data and wih missing data
        '''
        def is_number(x):
            try:
              return np.isfinite(x)  # Returns True for finite numbers
            except:
                return False  # Returns False for non-numeric types
        def miss_finder(data):
            mask=np.ones_like(data)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if not is_number(data[i,j]):
                        data[i,j]=np.nan
                        mask[i,j]=0
            completeRows=np.where(np.sum(mask,axis=1)==data.shape[1])[0]
            incompletRows=np.where((np.sum(mask,axis=1)<data.shape[1]) & (np.sum(mask,axis=1)>1))[0]

            return completeRows, incompletRows

        completeRows,incompletelRows=miss_finder(Z)
        # pca development over complete rows
        X_complete=Z[completeRows,:]
        pca_model=pca()
        pca_model.train(X_complete)

        # Using PCA for Data estimator
        X_incomplete=Z[incompletelRows,:]
        Estimated_block=np.zeros_like(X_incomplete)
        Estimation_quality=np.zeros((X_incomplete.shape[0],1))
        for i in range(X_incomplete.shape[0]):
           x_new=X_incomplete[i,:].reshape(1,X_incomplete.shape[1])
           available_col = np.where(~np.isnan(x_new).any(axis=0))[0]
           no_avable_col = np.where(np.isnan(x_new).any(axis=0))[0]
           # scaling  x_new
           C_scaling=pca_model.x_scaling[0,available_col]
           S_scaling=pca_model.x_scaling[1,available_col]
           X_new_scaled=(x_new[0,available_col]-C_scaling)/S_scaling.reshape(1,-1)
           
           P_new=pca_model.P[available_col,:]
           t_new=(X_new_scaled @ P_new) @ np.linalg.inv(P_new.T @ P_new)
           
           x_hat=t_new @ pca_model.P.T
           Estimated_block[i,:]=pca_model.unscaler(x_hat).reshape(1,-1)
           if comple_data is not None:
               actual=comple_data[incompletelRows[i],no_avable_col].reshape(1,-1)
               estimated=Estimated_block[i,no_avable_col].reshape(1,-1)
               Estimation_quality[i]=dis.clossness_metric(actual,estimated,pca_model.Xtrain_normal[:,no_avable_col])
        return Estimated_block,Estimation_quality,incompletelRows
    
