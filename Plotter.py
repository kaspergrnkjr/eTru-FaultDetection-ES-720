import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import getpass


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation= 0, ha="center",
             rotation_mode="anchor")

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    font = {'weight': 'bold',
            'size': 4.8}
    matplotlib.rc('font', **font)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, 0 if data[i,j] < 0.01 else valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts



def confusionMatrixCombiner(firstStage, secondStage, info):
    confusionMatrix = firstStage + secondStage
    confusionMatrix = np.array(confusionMatrix)
    #Positive = Fault
    TrueNegative = confusionMatrix[0][0]/np.sum(confusionMatrix[0][:])
    FalsePositive = np.sum(confusionMatrix[0][1:])/np.sum(confusionMatrix[0][:])
    FalseNegative = np.sum(confusionMatrix[1:,0])/np.sum(confusionMatrix[1:,:])
    TruePositive = np.sum(confusionMatrix[1:,1:])/np.sum(confusionMatrix[1:,:])
    print('TrueNegative' , TrueNegative)
    print('FalsePositive' , FalsePositive)
    print('FalseNegative' , FalseNegative)
    print('TruePositive' , TruePositive)
    for i in range(21):
        confusionMatrix[i, :] = confusionMatrix[i, :] / np.sum(confusionMatrix[i, :])
    fig, ax = plt.subplots()
    idx = np.array(range(0, 21))
    ax.set_xlabel('Predicted label', fontsize=15)
    ax.set_ylabel('True label', fontsize=15)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(2.50)
    solverID = info[11]
    if solverID == 'NoneType' or solverID == 'none':
        solverID = "SVM"
    computer = getpass.getuser()
    print(computer)
    if computer == 'User':
        basepath = 'Basepath to save confusion matrix'

    if solverID != "MLP":
        dimensionID = info[12]
        dimensionID2 = info[13]
    else:
        dimensionID = 'all'
        dimensionID2 = '-'
        solverID = 'SVM'
    if solverID == "LDA":
        plt.title("LDA-SVM", fontsize=25)
    elif solverID == "PCA":
        plt.title("PCA-SVM", fontsize=25)
    elif solverID == "MLP":
        plt.title("MLP", fontsize=25)
    else:
        plt.title("SVM", fontsize=25)

    avgAccuracy = np.trace(confusionMatrix)/np.sum(confusionMatrix)

    plt.savefig(basepath + solverID + '_DimensionsFirstStage_' + str(dimensionID)+ '_DimensionsSecondStage_' + str(dimensionID2) + '_RM3_' + str(info[5]) + '_BINARY_' + str(
        info[2]) + '_SWAPDATA_' + str(info[0]) + '_USETEST_' + str(info[1]) + '_TWOSTAGE_' + str(
        info[3]) + '_FRACTION_' + str(info[4]) + '_ACCURACY_' + str(avgAccuracy) + '.pdf', format="pdf", dpi=1000)
    print(avgAccuracy)
    plt.show()
