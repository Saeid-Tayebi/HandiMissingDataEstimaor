# Handi Missing Data Estimator

This repository contains a handy tool for **missing data estimation** using the **Principal Component Analysis (PCA)** approach. The method has also been integrated into the **PCA-NIPALS** codes, making it even easier to use within that framework.

## Overview

The tool is designed to estimate missing values in datasets where some rows have incomplete data. It uses the complete columns of the data to train a PCA model and then applies this model to estimate the missing values for each row. 

### Key Features:
- **Data Imputation using PCA**: The code estimates missing values by learning a PCA model from the complete data and imputing the missing values based on the principal components.
- **Integrated with PCA-NIPALS**: This method has already been added to the **PCA-NIPALS** codes, providing a seamless way to handle missing data as part of the PCA modeling process.
- **Accuracy Evaluation**: If the true values of the data are available, the tool can also compute the accuracy of its estimates by comparing the imputed values with the actual values.
  
### Predictive Modeling Hint

This method can even be used as a **predictive model**. To do this, you can add the output (`Y`) variables to the input (`X`) to create a new block of data. For future data points where only `X` values are available, you can append them to this block and set the `Y` columns as `NaN`. The PCA model will then predict the missing `Y` values based on the relationships it learned from the complete data.

## How It Works

1. You provide a block of data with missing values.
2. The code trains a PCA model using the complete columns in the dataset.
3. The PCA model is then used to estimate the missing values for each row.
4. If you provide the original data (with no missing values), the code evaluates its accuracy in estimating those missing values.

### Example of Usage

Below is an example of how to use the tool in your code:

```python
# %% Handi missing data executor
import numpy as np
import handi_missing_data_Estimator
import pandas as pd

# Generating synthetic data
Num_observation = 30
Ninput = 5
Noutput = 3
Num_com = 2             # Number of PLS components (=Number of X Variables)
alpha = 0.95            # Confidence limit (=0.95)
scores_plt = np.array([1, 2])

X = np.random.rand(Num_observation, Ninput)
Beta = np.random.rand(Ninput, Noutput) * 2 - 1
CompleteData = X @ Beta

# Function to randomly replace values with NaN
def randomly_replace_values(data, percentage=10, replacement_value=np.nan):
    flat_data = data.flatten()
    num_elements_to_replace = int(len(flat_data) * (percentage / 100))
    indices_to_replace = np.random.choice(len(flat_data), num_elements_to_replace, replace=False)
    flat_data[indices_to_replace] = replacement_value
    return flat_data.reshape(data.shape)

# Introducing missing values in 10% of the data
IncompletData = randomly_replace_values(CompleteData, percentage=10, replacement_value=np.nan)

# Estimating missing values using the Handi Missing Data Estimator
estimated_block, estimation_accuracy, incompleteRows = handi_missing_data_Estimator.estimate_missing_data(IncompletData, CompleteData)

# Combining results into a DataFrame for easy viewing
ResultData = np.hstack((CompleteData[incompleteRows], estimated_block, estimation_accuracy))
ResultDF = pd.DataFrame(ResultData)
print(ResultDF)
print('Estimated Accuracy:', estimation_accuracy)
# %%
```

### Running the Code

1. Place the **`handi_missing_data_Estimator.py`** in the same directory or import it from your module path.
2. Modify the input data to match your use case or generate synthetic data like in the example.
3. Run the provided script to estimate the missing values and check the accuracy.
