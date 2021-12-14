import matplotlib.pyplot as plt
import numpy as np
# after the training loop returns, we can plot the data
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
def plot_test(test_results,path):
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}

    plt.rc('font', **font)

    fig, ax = plt.subplots(ncols=3, nrows=1,figsize=(40, 10))
    # Defining custom 'xlim' and 'ylim' values.
    # custom_xlim = (0, 100)
    custom_ylim = (0, 1.1)
    # Setting the values for all axes.
    plt.setp(ax, ylim=custom_ylim)

    ax[0].plot(test_results[0], 'r', label='Test loss')
    ax[1].plot(test_results[1], 'g', label='Test meanIOU')
    ax[2].plot(test_results[2], 'b', label='Test pixelAcc')

    for i in ax:
        i.legend()
        i.grid(True)

    plt.savefig(path)
    plt.clf()

def plot(retval,N,running_m,path):
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}

    plt.rc('font', **font)
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(40, 20))
    # Defining custom 'xlim' and 'ylim' values.
    # custom_xlim = (0, 100)
    custom_ylim = (0, 1.1)
    # Setting the values for all axes.
    plt.setp(ax, ylim=custom_ylim)
    # N = 1000
    if running_m:
        ax[0][0].plot(running_mean(retval[0], N), 'r.', label='training loss')
        ax[1][0].plot(running_mean(retval[1], N), 'r.', label='validation loss')
        ax[0][1].plot(running_mean(retval[2], N), 'g.', label='meanIOU training')
        ax[1][1].plot(running_mean(retval[4], N), 'g.', label='meanIOU validation')
        ax[0][2].plot(running_mean(retval[3], N), 'b.', label='pixelAcc  training')
        ax[1][2].plot(running_mean(retval[5], N), 'b.', label='pixelAcc validation')
    else:
        ax[0][0].plot(retval[0], 'r', label='training loss')
        ax[1][0].plot(retval[1], 'r', label='validation loss')
        ax[0][1].plot(retval[2], 'g', label='meanIOU training')
        ax[1][1].plot(retval[4], 'g', label='meanIOU validation')
        ax[0][2].plot(retval[3], 'b', label='pixelAcc  training')
        ax[1][2].plot(retval[5], 'b', label='pixelAcc validation')
    for i in ax:
        for j in i:
            j.legend()
            j.grid(True)
    plt.ylim((0, 1.1))
    plt.savefig(path)
    plt.clf()

