import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot(filename):
    methods = ['RSE', 'ARSE', 'T-linkage', 'RCMSA', 'RPA']

    with open(filename, 'r') as csvfile:
        labels = [line[0] for line in csv.reader(csvfile, delimiter=',')]
        labels = filter(lambda e: len(e) > 0, labels)
        labels = filter(lambda e: e != 'Mean' and e != 'Median' and e != 'STD',
                        labels)

    mat = np.genfromtxt(filename, delimiter=',')
    data = mat[2:21, 5:]
    stats = mat[21:24, 5:]

    idx = np.arange(data.shape[0])
    width = 0.15
    total_width = data.shape[1] * width

    colors = sns.color_palette("Set1", n_colors=5)

    sns.set_style("whitegrid")

    plt.figure(figsize=(20, 5))
    ax = plt.axes()
    for j in range(data.shape[1]):
        bars = ax.bar(idx + j * width - total_width / 2, data[:, j], width,
                      linewidth=0,
                      color=colors[j])
        bars.set_label(methods[j])

    for j in range(stats.shape[1]):
        ax.bar(data.shape[0] + j * width - total_width / 2, stats[0, j], width,
               yerr=stats[1, j], ecolor='#5c5c5c', linewidth=0, color=colors[j])

    for j in range(stats.shape[1]):
        ax.bar(data.shape[0] + 1 + j * width - total_width / 2, stats[2, j],
               width,
               linewidth=0, color=colors[j])

    idx = np.arange(data.shape[0] + 2)
    _, labels = plt.xticks(idx + width,
                           labels + [u'mean (\u00B1STD)', 'median'],
                           rotation=45, horizontalalignment='right',
                           fontsize='16')
    labels[-1].set_weight('bold')
    labels[-2].set_weight('bold')
    _, labels = plt.yticks()
    for l in labels:
        l.set_fontsize(16)

    plt.legend(ncol=data.shape[1], fontsize='16')

    plt.ylabel('Misclassification error (%)', fontsize='16')

    plt.xlim(-width - total_width / 2,
             data.shape[0] + 1 + width + total_width / 2)
    plt.ylim(-1, (np.floor_divide(np.max(data), 10) + 2) * 10)
    plt.tight_layout()
    
    plt.savefig(filename[:-4] + '.pdf')


if __name__ == '__main__':
    plot('fundamental.csv')
    plot('homography.csv')
    plt.show()
