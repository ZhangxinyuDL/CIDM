import numpy as np
import logging
from numba import jit

def calc_CSI(pred, true, threshold):
    if pred.shape[0] != true.shape[0]:
        raise Exception('y_pred and y_true shape not match!')

    pred_labels = np.ones_like(pred)
    pred_labels[pred < threshold] = 0

    true_labels = np.ones_like(true)
    true_labels[true < threshold] = 0

    # compute the CSI
    CSI, POD, FAR, TP, FP, FN, TN, acc, CI_acc, NCI_acc = get_CSI(pred_labels, true_labels)

    logging.info('CSI %5.3f, POD %5.3f, FAR  %5.3f, threshold %5.3f' % (CSI, POD, FAR, threshold))
    logging.info(' TP, FP, FN, TN: %d, %d, %d, %d' % (TP, FP, FN, TN))
    logging.info(' accuracy: %5.3f ' % acc)
    logging.info(' CI_accuracy: %5.3f ' % CI_acc)
    logging.info(' NCI_accuracy: %5.3f ' % NCI_acc)

    return CSI, POD, FAR


def get_CSI(dec_labels, true_labels):
    '''calculate the CSI, POD, FAR and return them'''

    # number of samples
    num = dec_labels.shape[0]

    # compute TP, FP, FN, TN
    TP = float(np.sum(true_labels[dec_labels == true_labels]))
    FP = float(np.sum(dec_labels == 1) - TP)
    FN = float(np.sum(true_labels == 1) - TP)
    TN = float(np.sum(true_labels[dec_labels == true_labels] == 0))

    # compute CSI, POD, FAR
    if TP + FP + FN == 0:
        print('There is no CI')
        CSI = 0
    else:
        CSI = TP / (TP + FP + FN)

    if TP + FN == 0:
        print('There is no CI')
        POD = 0
        CI_acc = 0
    else:
        POD = TP / (TP + FN)
        CI_acc = TP / (TP + FN)

    if FP + TP == 0:
        FAR = 0
    else:
        FAR = FP / (FP + TP)

    # compute CI, NCI accuracy
    acc = (TP + TN) / (TP + TN + FP + FN)
    # CI_acc = TP / ( TP + FN )
    NCI_acc = TN / (TN + FP)

    return [CSI, POD, FAR, TP, FP, FN, TN, acc, CI_acc, NCI_acc]



