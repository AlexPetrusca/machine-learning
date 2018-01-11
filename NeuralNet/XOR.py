import time

from NeuralNet import NeuralNet
import numpy as np

net = NeuralNet(2, [4], 1)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

Y = np.array([[0],
              [1],
              [1],
              [0]])

before = time.time()
for j in xrange(60000):
    net.feed_forward(X)
    net.back_propagate(Y)
after = time.time()

print after-before

print "Output After Training:"
print net.get_last_output()
print
print net.get_weights()


