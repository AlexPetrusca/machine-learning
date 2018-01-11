import numpy as np
import time


def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

before = time.time()

np.random.seed(1)
synapse0 = 2 * np.random.random((3, 4)) - 1
synapse1 = 2 * np.random.random((4, 1)) - 1
for j in xrange(60000):
    l0 = X  # feed forward inputs
    l1 = sigmoid(l0.dot(synapse0))  # feed forward inputs through l1 via synapse0
    l2 = sigmoid(l1.dot(synapse1))  # feed forward result through l2 via synapse1

    l2_error = y - l2  # calculate overall error: (output) - (expected output)
    l2_delta = l2_error * sigmoid(l2, True)  # get delta value for overall error [applies to synapse1]

    l1_error = l2_delta.dot(synapse1.T)  # backpropogate error in synapse1 to figure out the error in synapse0
    l1_delta = l1_error * sigmoid(l1, True)  # get delta value for synapse0 error

    synapse1 += l1.T.dot(l2_delta)  # adjust synapse1 weights: error passed though l1
    synapse0 += l0.T.dot(l1_delta)  # adjust synapse0 weights: error passed though l0

after = time.time()

print after-before

print "Output After Training:"
print l2