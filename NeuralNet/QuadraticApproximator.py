from NeuralNet import NeuralNet
import numpy as np
from matplotlib import pyplot as plt


def f(x):
    return x**2


net = NeuralNet(2, [80, 40], 1)
net2 = NeuralNet(2, [80], 1)
learnRange = 50
np.random.seed(2)
for i in range(100000):
    x = (learnRange * np.random.rand(1) - learnRange/2).squeeze()
    y = (learnRange * np.random.rand(1) - learnRange/2).squeeze()
    fx = f(x)
    expected = 1 if (y > fx) else 0

    net.feed_forward(np.array([[x, y]]))
    net.back_propagate(np.array([[expected]]))

print
print net.feed_forward(np.array([[2, 20]]))  # 1
print net.feed_forward(np.array([[2, 4.3]]))  # 1

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

x = np.linspace(-4, 4, 1000)
y = f(x)
ax1.plot(x, y)

for i in range(1, 1000):
    x = (32 * np.random.rand(1) - 16).squeeze()
    y = (32 * np.random.rand(1) - 16).squeeze()
    result = net.feed_forward(np.array([[x, y]]))
    if result > 0.007:
        ax1.scatter(x, y, s=10, color='green')
    else:
        ax1.scatter(x, y, s=10, color="red")

learnRange2 = 3
for i in range(100000):
    x = (learnRange2 * np.random.rand(1) - learnRange2/2).squeeze()
    y = (learnRange2 * np.random.rand(1) - learnRange2/2).squeeze()
    fx = f(x)
    expected = 1 if (y > fx) else 0

    net2.feed_forward(np.array([[x, y]]))
    net2.back_propagate(np.array([[expected]]))

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

x = np.linspace(-1, 1, 1000)
y = f(x)
ax2.plot(x, y)

for i in range(1, 1000):
    x = (2 * np.random.rand(1) - 1).squeeze()
    y = (2 * np.random.rand(1) - 1).squeeze()
    result = net2.feed_forward(np.array([[x, y]]))
    if result > 0.7:
        ax2.scatter(x, y, s=10, color='green')
    else:
        ax2.scatter(x, y, s=10, color="red")

plt.show()
