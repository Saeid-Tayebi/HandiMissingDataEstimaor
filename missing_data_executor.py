# %% Handi missing data executor
import numpy as np
import handi_missing_data_Estimator
import pandas as pd

# Generating data
Num_observation=30
Ninput=5
Noutput=3
Num_com = 2             # Number of PLS components (=Number of X Variables)
alpha = 0.95            # Confidence limit (=0.95)
scores_plt=np.array([1,2])

X =np.random.rand(Num_observation,Ninput)
Beta=np.random.rand(Ninput,Noutput) * 2 -1 #np.array([3,2,1])
CompleteData=(X @ Beta)



def randomly_replace_values(data, percentage=10, replacement_value=np.nan):
    flat_data = data.flatten()
    num_elements_to_replace = int(len(flat_data) * (percentage / 100))
    indices_to_replace = np.random.choice(len(flat_data), num_elements_to_replace, replace=False)
    flat_data[indices_to_replace] = replacement_value
    return flat_data.reshape(data.shape)
IncompletData=randomly_replace_values(CompleteData, percentage=10, replacement_value=np.nan)

#estimated_block,estimation_accuracy,incompleteRows=MissEstimatorPCA.MissEstimator(y_with_missing,Y)
estimated_block,estimation_accuracy,incompleteRows=handi_missing_data_Estimator.estimate_missing_data(IncompletData,CompleteData)

ResultData=np.hstack((CompleteData[incompleteRows],estimated_block,estimation_accuracy))
ResultDF=pd.DataFrame(ResultData)
print(ResultDF)
print('Estimated Accuracy')
print(estimation_accuracy)

# %%
