import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# fm = matplotlib.font_manager
# fm._get_fontconfig_fonts.cache_clear()

font = {'family': 'times',
        'size': 16}

matplotlib.rc('font', **font)

# 5000 parts


# l = 5e-3
histo2 = np.array(
    [0, 43891, 38928, 32782, 26676, 21325, 17169, 13696, 11590, 9797, 8569, 7403, 6435, 5610, 4825, 4253, 3535, 3078, 2692, 2294, 2028, 1715, 1382, 1198, 1003,
     879, 744, 639, 509, 452, 392, 321, 250, 233, 167, 153, 110, 90, 76, 65, 48, 40, 39, 29, 26, 16, 15, 4, 6, 6, 4, 1, 3, 2, 1, 2, 0, 2, 0, 0])

# l = 1e-3
histo1 = np.array(
    [0, 115655, 116578, 117759, 119900, 122134, 127847, 132757, 139701, 146887, 155529, 164933, 171974, 175798, 176143, 169992, 158588, 142365, 123198, 101872,
     81391, 61693, 45399, 31809, 21287, 13844, 8607, 5321, 3151, 1773, 944, 475, 221, 144, 57, 31, 14, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0])

# l = 1e-2
histo3 = np.array(
    [0, 28088, 22960, 18558, 15427, 13685, 12136, 11235, 10044, 9523, 8989, 8337, 7721, 7164, 6823, 6420, 5878, 5731, 5216, 4970, 4675, 4426, 4119, 3793, 3560,
     3416, 3091,
     2779, 2763, 2560, 2341, 2265, 2104, 1958, 1804, 1695, 1580, 1450, 1315, 1169, 1095, 1083, 987, 927, 844, 709, 696, 690, 578, 530, 491, 476, 437, 389, 351,
     348, 289, 285, 244, 240])


def addPlot(histo, color, label):
    h = histo / sum(histo) * 100
    debut = 0
    first = h[1]
    for i in range(len(h)):
        if h[i] > first / 4:
            debut = i

    xs = np.array([i for i in range(len(h))])

    def getys(histo, debut):
        N = histo[debut]
        ntot = sum(histo[debut:])
        x = 1 - N / ntot
        return N * x ** (xs - debut)

    plt.ylim(0, first)

    plt.xlabel("|j-i|")
    plt.ylabel("% de collision")
    plt.plot(xs, h, "+", color=color, label=label, markersize=9)
    plt.plot(xs, getys(h, debut), "-", color=color, linewidth=1.8)


plt.grid()
addPlot(histo1, "red", "ls=1 mm")
addPlot(histo2, "blue", "ls=5 mm")
addPlot(histo3, "black", "ls=10 mm")

plt.legend()
plt.show()
