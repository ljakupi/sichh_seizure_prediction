import os
import itertools
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

np.random.seed(1)
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm


def plot_cv_splits(cv_fold, X, Y, PREICTAL_DURATION, PATIENT, cv_fold_name):

    # StratifiedKFold plot
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_cv_indices(cv_fold, X, Y, ax, 5)
    ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))], ['Testing set', 'Training set'], loc=(1.02, .8))
    fig.subplots_adjust(right=.7)

    DIR = './images/cv/' + 'chb{:02d}'.format(PATIENT)
    os.makedirs(DIR, exist_ok=True)  # create any parent directory that does not exist
    fig.savefig(DIR + '/' + str(PREICTAL_DURATION) + '_' + cv_fold_name + '.png')  # Save the full figure...




def plot_cv_indices(cv, X, y, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class']
    ax.set(yticks=np.arange(n_splits+1) + .5, yticklabels=yticklabels,
           xlabel='Samples', ylabel="CV iteration",
           ylim=[n_splits+1.2, -.2], xlim=[0, len(y)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=14)
    return ax



def plot_roc_auc(tprs, mean_fpr, aucs, MODEL, PATIENT, plt):
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    DIR = './images/cv/' + 'chb{:02d}'.format(PATIENT)
    os.makedirs(DIR, exist_ok=True)  # create any parent directory that does not exist
    plt.savefig(DIR + '/' + MODEL + '_ROC_AUC.png')


#Evaluation of Model - Confusion Matrix Plot
def plot_confusion_matrix(cm, MODEL, PATIENT, plt, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    DIR = './images/cv/' + 'chb{:02d}'.format(PATIENT)
    os.makedirs(DIR, exist_ok=True)  # create any parent directory that does not exist
    plt.savefig(DIR + '/' + MODEL + '_confusion_matrix.png')

