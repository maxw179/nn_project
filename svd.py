import numpy as np

# Given a weight matrix and list of singular value indices, 
# perform SVD on W and set singular vectors corresponding to sigma indicest to 0. 
# Return the updated, reduced weight matrix.
def weights_svd(W, sigma_indices):
    # Perform SVD on W.
    U, S, VT = np.linalg.svd(W) # Left singular vectors (U), singular values (S), right singular vectors (VT.T)

    # Construct singular value matrix (Sigma)
    Sigma = np.zeros((W.shape[0], W.shape[1]))
    np.fill_diagonal(Sigma, S)

    # Set singular vectors corresponding to sigma indices to 0.
    U[:, sigma_indices] = 0
    S[sigma_indices] = 0
    VT[sigma_indices, :] = 0

    # Construct singular value matrix (Sigma)
    Sigma = np.zeros((W.shape[0], W.shape[1]))
    np.fill_diagonal(Sigma, S)

    # Construct reduced weight matrix
    W_reduced = np.dot(U, np.dot(Sigma, VT))


    # Return reduced weight matrix
    return W_reduced

    