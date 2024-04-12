import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import RocCurveDisplay, roc_curve


def plot_ROC_PRauc_CM_stem(y_true, y_pred, y_proba, pos_label):
    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=pos_label)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)

    prec, recall, _ = precision_recall_curve(y_true, y_proba, pos_label=pos_label)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)

    cm = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(cm)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    roc_display.plot(ax=ax1, color="red")
    pr_display.plot(ax=ax2)
    cm_display.plot(ax=ax3)
    plot_stemplot(y_true, y_proba, ax=ax4)

    plt.show()


def plot_stemplot(y_true, y_proba, ax=None):
    y_df = pd.DataFrame(y_true)
    y_df["score"] = y_proba
    y_df = y_df.sort_values(by="score", ascending=False)
    y_df = y_df.reset_index().reset_index()

    y_df_response = y_df[y_df[y_true.name] == 1]
    y_df_non_response = y_df[y_df[y_true.name] == 0]

    if ax is None:
        (markers1, stemlines1, baseline1) = plt.stem(y_df_response['index'], y_df_response["score"], bottom=0.5)
        (markers2, stemlines2, baseline2) = plt.stem(y_df_non_response['index'], y_df_non_response["score"], bottom=0.5)
        plt.setp(markers1, marker='o', markersize=3, markeredgecolor="red", markeredgewidth=2)
        plt.setp(markers2, marker='o', markersize=3, markeredgecolor="blue", markeredgewidth=2)
        plt.setp(stemlines1, linestyle="-", color="red", linewidth=2)
        plt.ylabel('Prediction scores')
    else:
        (markers1, stemlines1, baseline1) = ax.stem(y_df_response['index'], y_df_response["score"], bottom=0.5)
        (markers2, stemlines2, baseline2) = ax.stem(y_df_non_response['index'], y_df_non_response["score"], bottom=0.5)
        plt.setp(markers1, marker='o', markersize=3, markeredgecolor="red", markeredgewidth=2)
        plt.setp(markers2, marker='o', markersize=3, markeredgecolor="blue", markeredgewidth=2)
        plt.setp(stemlines1, linestyle="-", color="red", linewidth=2)
        plt.ylabel('Prediction scores')


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.

    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
