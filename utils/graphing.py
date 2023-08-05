
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
# 279.91357421875
# 99.11660766601562

def plot_file(f_name, label_str, title):
    f = open(f_name, 'r')
    content = f.read()
    data = content.split('\n')
    del data[-1]
    data = np.array([float(x) for x in data])
    data_idx = list(range(len(data)))

    plt.plot(data_idx, data, 'r-', label=label_str)
    plt.legend()
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


if __name__ == "__main__":
    plot_file('./gnn.txt', 'PyG temporal', 'TGCN Loss')
