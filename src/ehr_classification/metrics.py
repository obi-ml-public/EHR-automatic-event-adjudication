import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr, spearmanr


# binary classification metrics
def binary_classification_auroc(x, y):
    return roc_auc_score(np.argmax(x, axis=-1), np.argmax(y, axis=-1))


def binary_classification_auprc(x, y):
    return average_precision_score(x, y)


def binary_classification_f1(x, y):
    return precision_recall_fscore_support(np.argmax(x, axis=-1), np.argmax(y, axis=-1), pos_label=1, average='binary')[2]


# multi-class classification metrics
def prec_rec_conf_multitask(x, y, print_conf_mat=True):
    x_ = np.argmax(x, axis=-1)
    y_ = np.argmax(y, axis=-1)

    # TODO zero division warning hides confusion matrix
    if print_conf_mat:
        print(confusion_matrix(x_, y_))

    return precision_recall_fscore_support(x_, y_, average='macro')[2]


def multiclass_classification_f1score(x, y):
    return prec_rec_conf_multitask(x, y)


def multiclass_classification_auroc(x, y):
    return roc_auc_score(np.argmax(x, axis=-1), y, average='macro', multi_class='ovr')


def multiclass_classification_auprc(x, y):
    return average_precision_score(x, y, average='macro')


# multitask regression metrics
def multitask_spearman(x, y):
    return np.mean([spearmanr(x[:, i], y[:, i])[0] for i in range(x.shape[1])])


def multitask_pearson(x, y):
    return np.mean([pearsonr(x[:, i], y[:, i])[0] for i in range(x.shape[1])])
