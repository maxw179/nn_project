# this file takes max's weights and biases
# and loads them into a list called params

# i recommend importing this using
# from load_params import *
# at the start of your files

import numpy as np

#load back the data
dir = "nn_training/"

#make a list where each entry is a matrix of weights or biases
params = []
for i in range(8):
    #for even i, type_string == "weights", for odd i type_string == "biases"
    type_string = 'weights' * (i%2 == 0) + 'biases' * (i%2 == 1)
    #params will have elements weights_0, biases_0, weights_1, biases_1, weights_2, biases_2, weights_3, biases_3
    params.append(np.load(dir +"nn_" + type_string + "_layer_" + str(int(i/2)) + ".npy", ))

#create a list of weight matrices
weights = [params[2*i] for i in range(4)]

#and a list of biases
bias = [params[2*i + 1] for i in range(4)]