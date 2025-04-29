import numpy as np

# Given a weight matrix and list of singular value indices, 
# perform SVD on W and remove singular vectors corresponding to sigma indices. 
# Return the updated, reduced weight matrix.
def weights_svd(W, sigma_indices):
    # Perform SVD on W.
    U, S, VT = np.linalg.svd(W) # Left singular vectors (U), singular values (S), right singular vectors (VT.T)

    # Construct singular value matrix (Sigma)
    Sigma = np.zeros((W.shape[0], W.shape[1]))
    np.fill_diagonal(Sigma, S)

    # Remove singular vectors corresponding to sigma indices.
    U_reduced = np.delete(U, sigma_indices)
    VT_reduced = np.delete(VT, sigma_indices)
    Sigma_reduced = np.delete(Sigma, sigma_indices)

    # Construct reduced weight matrix
    W_reduced = np.dot(U_reduced, np.dot(Sigma_reduced, VT_reduced))


    # Return reduced weight matrix
    return W_reduced

    