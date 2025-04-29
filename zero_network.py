'''
This function takes a neural network, as defined by Tensorflow's 
get weights function, zero-s the nth eigenvalue, and returns the new weights 
'''
# This notebook loads Max's neural net 
# and eigendecomposes the graph laplacian

# import needed libraries
import numpy as np
import scipy.linalg as linalg
from load_params import *


layer_sizes = np.array([784, 128, 64, 28, 10])
num_neurons = np.sum(layer_sizes) # = 1014
blocks = np.r_[0, np.cumsum(layer_sizes)]   # e.g. [0, 784, 912, 976, 1004, 1014]

#zero_eigenvals is the eigenvalue list
def zero_net(zero_eigenvals = []):
    A, D, L = get_laplacian()
    # Compute eigenvalues and eigenvectors of the Laplacian
    eigenvalues, eigenvectors = linalg.eigh(L)
    #argsort by the absolute value of the eigenvalue (swapped so biggest come first)
    sorted_eigenvalue_indices = np.argsort(-np.abs(eigenvalues))
    #figure out which ones we want to 0
    zerod_indices = sorted_eigenvalue_indices[zero_eigenvals]
    #zero them
    eigenvalues[zerod_indices] = 0

    # Reconstruct the Laplacian after modifying eigenvalues
    L_reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    new_model = reconstruct_weights(L_reconstructed)

    return A, L, L_reconstructed, new_model

def reconstruct_weights(L_reconstructed):
    weights_recons = []
    for i in range(4):
        #get the rows for each block
        first_row = blocks[i]
        last_row = blocks[i+1] 
        #and the columns for each block
        first_col = blocks[i+1]
        last_col = blocks[i+2] 

        #get reconstructed weights from the reconstructed Laplacian
        W_recons = L_reconstructed[first_row : last_row , first_col : last_col] 
        weights_recons.append(-W_recons)
    
    model = []
    for i in range(4):
        model.append(weights_recons[i])
        model.append(bias[i])
    
    return model

def get_laplacian():

    #construct the adjacency matrix
    A = np.zeros(shape = (num_neurons, num_neurons))

    # get the locations of each block
    blocks = np.r_[0, np.cumsum(layer_sizes)]   # e.g. [0, 784, 912, 976, 1004, 1014]

    # initialize adjacency
    A = np.zeros((num_neurons, num_neurons))

    # just load the weight matrices into the blocks 
    for i in range(4):
        #first just load the upper triangular part of the adjacency matrix

        #get the rows for each block
        first_row = blocks[i]
        last_row = blocks[i+1] 
        #and the columns for each block
        first_col = blocks[i+1]
        last_col = blocks[i+2] 

        # now assign the weight matrix to that block
        A[first_row : last_row , first_col : last_col] = weights[i]

        # mirror to make A symmetric
        A[first_col : last_col , first_row : last_row] = weights[i].T

    #create degree matrix
    D = np.diag(A.sum(axis=1))

    #calculate laplacian
    L = D - A
    return A, D, L
    



