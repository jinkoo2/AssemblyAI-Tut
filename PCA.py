import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean # in princicple, this is not needed for cov calculation, cause the function takes the mean out itself.

        # covariance, function needs samples as comlumns
        cov = np.cov(X.T)

        print('cov.shape', cov.shape)

        # eigen vectors and values
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        eigenvectors = eigenvectors.T # output is in column vectors. so, convert to row vectors.

        print('eigen vectors', eigenvectors)
        print('eigen values', eigenvalues)

        # sort eigenvectors in dec order
        idxs = np.argsort(eigenvalues)[::-1] # find indices to sort in dec based on the eigenvalues

        print(' indices to sorted in dec=',idxs)

        eigenvalues = eigenvalues[idxs] # sort eigen values
        eigenvectors = eigenvectors[idxs] #soft eigen vectors

        print('eigen vectors after sort', eigenvectors)
        print('eigen values after sort', eigenvalues)

        # only keep n_components of the eigen vectors
        self.components = eigenvectors[:self.n_components] 
        print('eigen vectors of priciple components', self.components)

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)

# Testing
if __name__ ==  "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets

    data = datasets.load_iris()
    X = data.data
    y = data.target

    # project the data onto the 2 primary principle components
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print(X.shape)
    print(X_projected.shape)

    x1 = X_projected[:,0]
    x2 = X_projected[:,1]

    plt.scatter(x1,x2, c=y, edgecolors="none", alpha=0.8, cmap=plt.cm.get_cmap('viridis',3))
    plt.xlabel('comp 1')
    plt.ylabel('comp 2')
    plt.colorbar()
    plt.show()





    
        