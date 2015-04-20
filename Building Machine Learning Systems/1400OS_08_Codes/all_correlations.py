import numpy as np

# This is the version in the book:
def all_correlations(bait, target):
    '''
    corrs = all_correlations(bait, target)

    corrs[i] is the correlation between bait and target[i]
    '''
    return np.array(
            [np.corrcoef(bait, c)[0,1]
                for c in target])

# This is a faster, but harder to read, implementation:
def all_correlations(y, X):
    '''
    Cs = all_correlations(y, X)

    Cs[i] = np.corrcoef(y, X[i])[0,1]
    '''
    X = np.asanyarray(X, float)
    y = np.asanyarray(y, float)
    xy = np.dot(X, y)
    y_ = y.mean()
    ys_ = y.std()
    x_ = X.mean(1)
    xs_ = X.std(1)
    n = float(len(y))
    ys_ += 1e-5 # Handle zeros in ys
    xs_ += 1e-5 # Handle zeros in x

    return (xy - x_*y_*n)/n/xs_/ys_

