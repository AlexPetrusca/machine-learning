import numpy as np


def bias_data(inputs):
    ones = np.full((inputs.shape[0], 1), 1)
    biased = np.concatenate((inputs, ones), axis=1)
    return biased


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


class NeuralNet:
    def __init__(self, i, h, o):
        self._i = i
        self._h = h
        self._o = o

        self.layers = list()
        for idx in range(len(h) + 2):
            self.layers.append(np.array(np.zeros(1)))

        np.random.seed(1)
        self.synapses = list()
        self.synapses.append(2 * np.random.random((i + 1, h[0])) - 1)
        for idx in range(len(h) - 1):
            self.synapses.append(2 * np.random.random((h[idx], h[idx + 1])) - 1)
        self.synapses.append(2 * np.random.random((h[-1], o)) - 1)

    def feed_forward(self, inputs):
        self.layers[0] = bias_data(inputs)
        for idx in range(1, len(self.layers)):
            self.layers[idx] = sigmoid(np.dot(self.layers[idx - 1], self.synapses[idx - 1]))
        return self.layers[-1]

    def back_propagate(self, expected):
        backprops = list()

        total_error = expected - self.layers[-1]
        total_delta = total_error * sigmoid(self.layers[-1], deriv=True)
        backprops.append(total_delta)

        for idx in range(1, len(self.synapses) + 1):
            error = backprops[idx - 1].dot(self.synapses[-idx].T)
            delta = error * sigmoid(self.layers[-(idx + 1)], deriv=True)
            backprops.append(delta)

        for idx in range(1, len(self.synapses) + 1):
            self.synapses[-idx] += self.layers[-(idx + 1)].T.dot(backprops[idx - 1])

        return self.get_weights()

    def reset(self):
        np.random.seed(1)
        self.synapses = list()
        self.synapses.append(2 * np.random.random((self._i + 1, self._h[0])) - 1)
        for idx in range(len(self._h) - 1):
            self.synapses.append(2 * np.random.random((self._h[idx], self._h[idx + 1])) - 1)
        self.synapses.append(2 * np.random.random((self._h[-1], self._o)) - 1)

    def get_weights(self):
        return np.array(self.synapses)

    def get_last_input(self):
        return self.layers[0]

    def get_last_output(self):
        return self.layers[-1]
