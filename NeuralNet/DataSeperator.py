from NeuralNet import NeuralNet
import numpy as np
from matplotlib import pyplot as plt, patches


def onclick(event):
    global points
    global expected
    if event.button == 1 or event.button == 3:
        plt.scatter(event.xdata, event.ydata, s=10, color="green" if(event.button == 1) else "red")
        if len(points) == 0:
            points = [np.append(points, [event.xdata, event.ydata])]
            expected = [np.append(expected, [1 if(event.button == 1) else 0])]
        else:
            points = np.vstack((points, [event.xdata, event.ydata]))
            expected = np.vstack((expected, [1 if(event.button == 1) else 0]))
    elif event.button == 2 and not is_training[0]:
        is_training[0] = True

net = NeuralNet(2, [10, 10, 10], 1)
points = []
expected = []
is_training = [False]

fig = plt.figure()
ax = fig.add_subplot(111)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.xlim(-50, 50)
plt.ylim(-50, 50)

plt.ion()

while not is_training[0]:
    plt.pause(0.01)

for idx in range(100000):
    net.feed_forward(points)
    net.back_propagate(expected)
    if idx % 10000 == 0:
        print "%.0f%% done" % (float(idx)/10000 * 1000)

print "done"

for x in range(100):
    for y in range(100):
        result = net.feed_forward(np.array([[x-50 + 1, y-50 + 1]]))
        patch = patches.Rectangle((x-50, y-50), 1, 1, alpha=0.2, color="green" if (result > 0.25) else "red")
        ax.add_patch(patch)

plt.pause(1000)
