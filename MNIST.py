import numpy as np
import mnist_loader as mnist

training_data, validation_data, test_data = mnist.load_data_wrapper()

allInputs = np.squeeze(list(map(lambda x: x[0].T, training_data)))
allAnswers = np.squeeze(list(map(lambda x: x[1].T, training_data)))
inputs = np.split(allInputs, 5000)
answers = np.split(allAnswers, 5000)

allTestInputs = np.squeeze(list(map(lambda x: x[0].T, test_data)))
allTestAnswers = np.squeeze(list(map(lambda x: x[1].T, test_data)))
testInputs = np.split(allTestInputs, 1000)
testAnswers = np.split(allTestAnswers, 1000)

print "Neural Started\n"


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


np.random.seed(1)
synapse0 = np.random.randn(784, 74)
synapse1 = np.random.randn(74, 10)

print "Training:"
for i in xrange(11):
    print "Cycle %d" % (i+1)
    for j in xrange(len(inputs)):
        layer0 = inputs[j]
        layer1 = sigmoid(np.dot(layer0, synapse0))
        layer2 = sigmoid(np.dot(layer1, synapse1))

        layer2_error = answers[j] - layer2
        layer2_delta = layer2_error * sigmoid(layer2, True)

        layer1_error = layer2_delta.dot(synapse1.T)
        layer1_delta = layer1_error * sigmoid(layer1, True)

        synapse1 += layer1.T.dot(layer2_delta)
        synapse0 += layer0.T.dot(layer1_delta)
print "DONE"


def decipher(output):
    return np.argmax(output)


print "\nTests:"
test_cycles = 1000
numCorrect = 0.0
numWrong = 0.0
for i in xrange(test_cycles):
    t = testInputs[i]
    a = testAnswers[i]
    for j in xrange(len(t)):
        layer0 = [t[j]]
        layer1 = sigmoid(np.dot(layer0, synapse0))
        layer2 = sigmoid(np.dot(layer1, synapse1))
        output = np.squeeze(layer2)

        # print "expected: %d" % (a[j]) + "\tactual: %d" % (decipher(output))
        if a[j] == decipher(output):
            numCorrect += 1
        else:
            numWrong += 1
    # print ""

print "DONE\n"
print "Total Number of Tests: %d" % (test_cycles * len(testInputs[0]))
print "Number Correct: %d" % (numCorrect)
print "Number Incorrect: %d" % (numWrong)
