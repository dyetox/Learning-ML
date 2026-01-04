import numpy as np

def ReLU(Z):
    """
    Implements ReLU function
    
    Z: Output of the linear layer
    """
    return np.maximum(0, Z)

def linear_forward(A_prev, W, b):
    """
    Implements the linear part of forward propagation
    
     A_prev: activations from previous layer
     W: weights matrix, numpy array of shape
     b: bias vector, numpy array of shape
     cache: a tuple containing A_prev, W, and b; stored for backpropagation
    """
    Z = np.dot (W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache


def compute_cost(AL, Y):
    """
    Implementation of the binary cross-entropy function.
    
    AL: probability vector corresponding to label predictions
    Y: true label vector 
    """
    m = Y.shape[1]
    J = (-1/m)* np.sum((Y*np.log(AL)) + (1-Y)*np.log(1-AL))
    J = np.squeeze(J)
    return J

def linear_backward(dZ, cache):
    """
    Implements function that calculates gradients for a single linear layer (Z= WA +b)
    
    dZ: Gradient of cost relative to the linear output
    cache: tuple coming from linear forward propagation

    Returns
    dA_prev: Gradient of the cost relative to the activation
    dW: Gradient of cost relative to W
    db: Gradient of cost relative to b
    """
    A_prev, W, b = cache

    m = A_prev.shape[1]
    
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def relu_backward(dA, cache):
    """
    Implementing ReLU for backpropagation
    
    """
    Z = cache
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0
    
    return dZ



if __name__ == "__main__":
    print("Hello World")
    









