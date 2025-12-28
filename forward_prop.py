import numpy as np

def ReLU(Z):
    return np.maximum(0, Z)

def linear_forward(A_prev, W, b):
    Z = np.dot (W, A_prev) + b
    return Z

if __name__ == "__main__":

    np.random.seed(1)
    A_prev_mock = np.random.randn(3, 2) # Input Data (3 features, 2 examples)
    W_mock = np.random.randn(1, 3)      # Weights (1 neuron, 3 inputs)
    b_mock = np.random.randn(1, 1)      # Bias

    print("--- STARTING TEST BENCH ---")

    Z_output = linear_forward(A_prev_mock, W_mock, b_mock)
    A_output = ReLU(Z_output)
    Z_expected = [[3.26295337, -1.23429987]]

    print(f"Calculated Z: {Z_output}")
    print(f"Calculated A: {A_output}")

    if Z_output is not None and np.allclose(Z_output, Z_expected):
        print("Linear Step: PASSED")
    else:
        print("Linear Step: FAILED")


    









