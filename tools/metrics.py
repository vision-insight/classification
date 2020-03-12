import os
import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score


def overall_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
    #if not isinstance(y_true, np.ndarray):
    #    y_true = np.asarray(y_true)
    #if not isinstance(y_pred, np.ndarray):
    #    y_pred = np.asarray(y_pred)
    #return np.mean(y_true == y_pred, dtype=np.float32)

def mean_accuracy(y_true, y_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)

    unique_label = np.unique(y_true)
    per_class_acc = []
    for label in unique_label:
        index = np.where(y_true == label)
        per_class_acc.append(np.mean(y_true[index] == y_pred[index]))
    return np.mean(per_class_acc, dtype=np.float32)


def mean_prec(y_true, y_pred):
    return precision_score(y_true, y_pred, average = "macro")

    #if not isinstance(y_true, np.ndarray):
    #    y_true = np.asarray(y_true)
    #if not isinstance(y_pred, np.ndarray):
    #    y_pred = np.asarray(y_pred)

    #unique_label = np.unique(y_true)
    #per_class_prec = []
    #for label in unique_label:
    #    index = np.where(y_pred == label)
    #    if len(index[0]) == 0:
    #        per_class_prec.append(0)
    #        continue
    #    per_class_prec.append(np.mean(np.where(y_true[index] == label)
    #return np.mean(per_class_prec, dtype = np.float32)

def overall_prec(y_true, y_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)

    return np.mean(y_true == y_pred, dtype=np.float32)

#print(mean_prec(label_list, pred_label_list))

def mean_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average = "macro")
    #if not isinstance(y_true, np.ndarray):
    #    y_true = np.asarray(y_true)
    #if not isinstance(y_pred, np.ndarray):
    #    y_pred = np.asarray(y_pred)
    #unique_label = np.unique(y_true)
    #per_class_recall = []
    #for label in unique_label:
    #    index = (y_true == label) * (y_pred == label)
    #    
    #    true_label_count = y_true.tolist().count(label)
    #    per_class_recall.append(sum(index)/true_label_count)

    #return np.mean(per_class_recall, dtype = np.float32)

