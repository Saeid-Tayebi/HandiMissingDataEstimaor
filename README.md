# PCAFill: Missing Data Estimation Using PCA

**PCAFill** is a Python package for estimating missing data in datasets using **Principal Component Analysis (PCA)**. It provides a robust and efficient way to handle missing values by leveraging the relationships within the available data. The package is now available for installation from the [Releases section](https://github.com/Saeid-Tayebi/HandiMissingDataEstimaor/releases/tag/first_release) of this repository.

In addition to the Python implementation, this project includes a **MATLAB version** for users who prefer working in MATLAB. Both implementations provide the same functionality and can be used interchangeably.

---

## Key Features

- **PCA-Based Imputation**: Estimates missing values using a PCA model trained on the complete rows of the dataset.
- **Cross-Platform Support**: Available for both **Python** and **MATLAB**.
- **Easy Integration**: Simple APIs for imputing missing values in NumPy arrays (Python) and matrices (MATLAB).
- **Tested Scenarios**: Includes a `test` folder with comprehensive test cases to ensure reliability and correctness.

---

## Installation

### Python

You can install the Python package directly from the [Releases section](https://github.com/Saeid-Tayebi/HandiMissingDataEstimaor/releases/tag/first_release) of this repository. Download the `.whl` or `.tar.gz` file and install it using `pip`:

```bash
pip install path_to_downloaded_file
```

Alternatively, you can clone the repository and install it locally:

```bash
git clone https://github.com/Saeid-Tayebi/HandiMissingDataEstimaor.git
cd HandiMissingDataEstimator
pip install .
```

### MATLAB

The MATLAB implementation is provided in the `matlab` folder of this repository. Simply add the `matlab` folder to your MATLAB path, and you’re ready to use the functions.

---

## Usage

### Python

The main function `pca_fill` takes a NumPy array with missing values (represented as `NaN`) and returns the array with missing values estimated using PCA.

```python
import numpy as np
from PCAFill.pca_fill import pca_fill

# Example data with missing values
data = np.array([
    [1, 2, np.nan, 4],
    [5, np.nan, 7, 8],
    [9, 10, 11, 12]
])

# Estimate missing values
filled_data = pca_fill(data)
print(filled_data)
```

### MATLAB

The MATLAB implementation provides the same functionality. Use the `pca_missing_data_estimator` function to estimate missing values in a matrix.

```matlab
% Example data with missing values
incompleteData = [
    1, 2, NaN, 4;
    5, NaN, 7, 8;
    9, 10, 11, 12
];

% Estimate missing values
estimatedData = pca_missing_data_estimator(incompleteData);
disp(estimatedData);
```

---

## Testing

### Python

The package includes a `test` folder with a comprehensive test suite to ensure the correctness of the imputation algorithm. You can run the tests using `pytest`:

```bash
pytest test/test.py
```

The test cases cover scenarios such as:

- Ensuring no `NaN` values remain after imputation.
- Verifying that known values are unchanged.
- Handling completely empty rows correctly.

### MATLAB

The MATLAB implementation can be tested using the example scripts provided in the `matlab` folder. Simply run the scripts to verify the functionality.

---

## How It Works

1. **Identify Missing Rows**: The algorithm identifies rows with missing values.
2. **Train PCA Model**: A PCA model is trained on the complete rows of the dataset.
3. **Estimate Missing Values**: The trained PCA model is used to estimate missing values for incomplete rows.
4. **Return Complete Data**: The dataset with estimated values is returned.

---

## Folder Structure

```
HandiMissingDataEstimator/
├── PCAFill/                  # Python package
│   ├── lib/                  # PCA implementation
│   ├── pca_fill.py           # Main imputation function

├── matlab/                   # MATLAB implementation
│   ├── pca_missing_data_estimator.m  # Main MATLAB function

├── test/                     # Test cases
│   ├── test_pca_fill.py      # Python tests

├── usage_example.py          # Python usage example
└── README.md                 # This file
```

---

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License.
