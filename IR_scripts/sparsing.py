#This code is added to the dataset_loading.py file at line 108

### Added Code
print("Setting values to 0")
percent_set_to_nan = 0.5
n_zeros = int(percent_set_to_nan * X.size)
mask = np.ones_like(X)
# randomly set n_zeros values in the mask to zero
np.random.seed(42)
mask.ravel()[np.random.choice(X.size, n_zeros, replace=False)] = 0
X = X * mask
### End of added code