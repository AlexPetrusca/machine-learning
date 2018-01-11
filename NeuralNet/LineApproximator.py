from NeuralNet import NeuralNet
import numpy as np
from matplotlib import pyplot as plt


def f(x):
    return 2 * x

net = NeuralNet(2, [40], 1)
learnRange = 20
for i in range(100000):
    x = (learnRange * np.random.rand(1) - learnRange/2).squeeze()
    y = (learnRange * np.random.rand(1) - learnRange/2).squeeze()
    fx = f(x)
    expected = 1 if (y > fx) else 0

    net.feed_forward(np.array([[x, y]]))
    net.back_propagate(np.array([[expected]]))

print net.feed_forward(np.array([[5, 2]]))   # 0
print net.feed_forward(np.array([[5, 13]]))  # 1
print net.feed_forward(np.array([[5, 14]]))  # 1

fig1 = plt.figure()
ax1 = fig1.add_subplot(212)
ax2 = fig1.add_subplot(211)
for i in range(1, 100):
    ax1.scatter(i, net.feed_forward(np.array([[learnRange*i, 2*learnRange*i + 1]])), s=120, color='blue')

np.random.seed(1)
for i in range(1, 100):
    x = (100 * np.random.rand(1) - 50).squeeze()
    y = (100 * np.random.rand(1) - 50).squeeze()
    result = net.feed_forward(np.array([[x, y]]))
    if result > 0.5:
        ax2.scatter(x, y, s=10, color='green')
    else:
        ax2.scatter(x, y, s=10, color="red")

ax2.plot([-30, 30], [-60, 60])

print
print net.feed_forward(np.array([[20, 40.1]]))  # 1
print net.feed_forward(np.array([[20, 39.9]]))  # 0

print
print net.feed_forward(np.array([[40, 80.1]]))  # 1
print net.feed_forward(np.array([[40, 79.9]]))  # 0

print
print net.feed_forward(np.array([[60, 120.1]]))  # 1
print net.feed_forward(np.array([[60, 119.9]]))  # 0

print
print net.feed_forward(np.array([[80, 160.1]]))  # 1
print net.feed_forward(np.array([[80, 159.9]]))  # 0

print
print net.feed_forward(np.array([[100, 200.1]]))  # 1
print net.feed_forward(np.array([[100, 199.9]]))  # 0

plt.show()
