#%%
import numpy as np
from MyPcaClass import MyPca as pca

def estimate_missing_data(Z:np.ndarray=None):
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
    Estimated_block=X_incomplete.copy()
    for i in range(X_incomplete.shape[0]):
        x_new=X_incomplete[i,:].reshape(1,X_incomplete.shape[1])
        available_col = np.where(~np.isnan(x_new).any(axis=0))[0]
        no_avable_col = np.where(np.isnan(x_new).any(axis=0))[0]
        # scaling  x_new
        C_scaling=pca_model.x_scaling[0,available_col]
        S_scaling=pca_model.x_scaling[1,available_col]
        X_new_scaled=(x_new[0,available_col]-C_scaling)/S_scaling.reshape(1,-1)
        
        P_new=pca_model.P[available_col,:]
        t_new = (X_new_scaled @ P_new) @ np.linalg.inv(P_new.T @ P_new)
        x_hat=t_new @ pca_model.P.T
        estimatedrow=pca_model.unscaler(x_hat).reshape(1,-1)
        Estimated_block[i,no_avable_col]=estimatedrow[0,no_avable_col]
    return Estimated_block

# Generating data
nan=np.nan
IncompletData=np.array([[-0.44898664, -1.70641315,         nan, -0.61481122, -1.54186358],
       [ 0.14868248, -0.95182193,  0.26727939,  0.09894418, -0.55422052],
       [-0.1798493 , -1.29440995, -0.32527414,         nan, -0.38205467],
       [        nan, -1.66052776, -0.49252195, -0.39719669, -1.16710771],
       [-0.08538955,         nan, -0.00694832, -0.40364363, -1.58145218],
       [ 0.11321242, -2.39018704,  0.04125567, -0.4059341 , -1.86275122],
       [-0.88536128, -1.22590664, -1.01597992, -0.14316028, -0.369122  ],
       [ 0.1811789 , -1.54900803,  0.146894  ,  0.15009641, -0.72790364],
       [-0.53476372, -1.54313308, -0.62499737,         nan, -0.71010732],
       [-0.35322975, -1.24944183, -0.30060002, -0.55576616, -1.25846825],
       [ 0.07291691, -1.99729142,  0.0780735 , -0.14377211, -1.27453381],
       [-0.40126038, -1.08807783, -0.57577629, -0.0776227 , -0.37762036],
       [-0.73483643, -0.64561676, -0.94676946, -0.33094222, -0.25938656],
       [-0.39107563, -0.78003942, -0.55761634,  0.00451722, -0.11291934],
       [ 0.26519618, -1.2684861 ,  0.3552548 ,  0.01289193, -0.7855855 ],
       [-0.32050487, -2.35814405,         nan, -0.14445954, -1.26817184],
       [-0.41584169,         nan,         nan, -0.37790992, -0.88047615],
       [-0.82915371, -1.36340072, -1.20160091, -0.41525505, -0.62972113],
       [ 0.27949563, -1.49953679,         nan, -0.22962866, -1.24533141],
       [        nan, -0.9969084 ,  0.24598719,  0.10496481, -0.58950246]])

# %%
estimated_block=estimate_missing_data(IncompletData)
print(estimated_block)

