import numpy as np
import pytest
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

incom_data = Y.copy()
n = np.size(incom_data)
portion_of_missed_data = np.random.uniform(0.1, 0.3)
nanidx = np.random.choice(
    range(0, n), size=int(np.round(portion_of_missed_data * n)), replace=False)
incom_data.flat[nanidx] = np.nan
incom_data[-1, :] = np.nan

compl_data = pca_fill(data=incom_data)


print('Original Data')
print(Y)
print('-------')

print('incomplete Data')
print(incom_data)
print('-------')

print('completed Data')
print(compl_data)
print('-------')


# Generating data
nan = np.nan
IncompletData = np.array([[-0.44898664, -1.70641315,         nan, -0.61481122, -1.54186358],
                         [0.14868248, -0.95182193,  0.26727939,  0.09894418, -0.55422052],
                          [-0.1798493, -1.29440995, -0.32527414,         nan, -0.38205467],
                          [nan, -1.66052776, -0.49252195, -0.39719669, -1.16710771],
                          [-0.08538955,         nan, -0.00694832, -0.40364363, -1.58145218],
                          [0.11321242, -2.39018704,  0.04125567, -0.4059341, -1.86275122],
                          [-0.88536128, -1.22590664, -1.01597992, -0.14316028, -0.369122],
                          [0.1811789, -1.54900803,  0.146894,  0.15009641, -0.72790364],
                          [-0.53476372, -1.54313308, -0.62499737,         nan, -0.71010732],
                          [-0.35322975, -1.24944183, -0.30060002, -0.55576616, -1.25846825],
                          [0.07291691, -1.99729142,  0.0780735, -0.14377211, -1.27453381],
                          [-0.40126038, -1.08807783, -0.57577629, -0.0776227, -0.37762036],
                          [-0.73483643, -0.64561676, -0.94676946, -0.33094222, -0.25938656],
                          [-0.39107563, -0.78003942, -0.55761634,  0.00451722, -0.11291934],
                          [0.26519618, -1.2684861,  0.3552548,  0.01289193, -0.7855855],
                          [-0.32050487, -2.35814405,         nan, -0.14445954, -1.26817184],
                          [-0.41584169,         nan,         nan, -0.37790992, -0.88047615],
                          [-0.82915371, -1.36340072, -1.20160091, -0.41525505, -0.62972113],
                          [0.27949563, -1.49953679,         nan, -0.22962866, -1.24533141],
                          [nan, -0.9969084,  0.24598719,  0.10496481, -0.58950246]])

# %%
estimated_block = pca_fill(IncompletData)
print(estimated_block)
