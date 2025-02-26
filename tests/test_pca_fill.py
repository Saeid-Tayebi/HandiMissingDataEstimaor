import numpy as np
from PCAFill.lib.pca import PcaClass as pca, pcaeval
from PCAFill.pca_fill import pca_fill


Num_observation = 30
xvar = 4
yvar = 3
n_component = yvar+1             # Number of PLS components (=Number of X Variables)

# Calibration Dataset
X = np.random.rand(Num_observation, xvar)
Beta = np.random.rand(xvar, yvar) * 2 - 1  # np.array([3,2,1])
Y = (X @ Beta)


def test_missing_data():
    incom_data = Y.copy()
    n = np.size(incom_data)
    portion_of_missed_data = np.random.uniform(0.1, 0.3)
    nanidx = np.random.choice(
        range(0, n), size=int(np.round(portion_of_missed_data * n)), replace=False)
    incom_data.flat[nanidx] = np.nan
    incom_data[-1, :] = np.nan

    compl_data = pca_fill(data=incom_data)

    not_a_nan_idx = np.where(~np.isnan(incom_data.reshape(-1)))[0]

    # Ensure no NaN values remain
    assert np.size(np.where(np.isnan(compl_data))[0]) == 0

    # Ensure known values are unchanged
    assert np.allclose(incom_data.flat[not_a_nan_idx], compl_data.flat[not_a_nan_idx], rtol=1e-3)

    # Test if completely empty rows are handled correctly
    assert np.all(compl_data[-1, :] == 0)
