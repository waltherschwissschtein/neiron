import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import sys
import os
def sigmoid(x):
    return 1/(1 + np.exp(-x))
def obr_sigmoid(x):
    return (1/(1 + np.exp(-x)))*(1-1/(1 + np.exp(-x)))
    
def init_weights(kvo_neir_1_sloa,kvo_neir_2_sloa):
    weights = np.random.normal(0.0, 1, (kvo_neir_1_sloa,kvo_neir_2_sloa))
    return weights
def pr_raspr(weight,znach):
    input=np.dot(weight,znach)       
    input=sigmoid(input)
    return input
a=init_weights(3,3)
x=np.array([2,4,8])
x=pr_raspr(a,x)
a=init_weights(1,3)
x=pr_raspr(a,x)


