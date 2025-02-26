import numpy as np
from .lib.pca import PcaClass as pca


def pca_fill(data: np.ndarray):
    """receives data in which there are some rows have missing values,
    then using the complete rows it develops a PCA model and using that pca model, it estimates the missing values


    Args:
        data (np.ndarray): data with nan values

    Returns:
        complete data: data with missing data filled
    """
    data = data.copy()
    incomplete_rows = np.where(np.isnan(data).any(axis=1))[0]

    complete_part = np.delete(data, incomplete_rows, axis=0)
    incomplete_part = data[incomplete_rows].reshape(-1, data.shape[1])

    pca_over_complete_data = pca().fit(complete_part)

    estimated_part = pca_over_complete_data.MissEstimator(incom_data=incomplete_part)

    data[incomplete_rows] = estimated_part

    return data
