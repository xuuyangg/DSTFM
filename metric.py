# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import confusion_matrix
def get_TP(target, prediction, threshold):
    '''
    compute the  number of true positive

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    assert (target.shape == prediction.shape)

    target = 1 - np.clip(target, threshold, 0) / threshold
    prediction = 1 - np.clip(prediction, threshold, 0) / threshold

    tp_array = np.logical_and(target, prediction) * 1.0
    tp = np.sum(tp_array)

    return tp


def get_FP(target, prediction, threshold):
    '''
    compute the  number of false positive

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    assert (target.shape == prediction.shape)

    target = np.clip(target, threshold, 0) / threshold
    prediction = 1 - np.clip(prediction, threshold, 0) / threshold

    fp_array = np.logical_and(target, prediction) * 1.0
    fp = np.sum(fp_array)

    return fp


def get_FN(target, prediction, threshold):
    '''
    compute the  number of false negtive

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    assert (target.shape == prediction.shape)

    target = 1 - np.clip(target, threshold, 0) / threshold
    prediction = np.clip(prediction, threshold, 0) / threshold

    fn_array = np.logical_and(target, prediction) * 1.0
    fn = np.sum(fn_array)

    return fn


def get_TN(target, prediction, threshold):
    '''
    compute the  number of true negative

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    assert (target.shape == prediction.shape)

    target = np.clip(target, threshold, 0) / threshold
    prediction = np.clip(prediction, threshold, 0) / threshold

    tn_array = np.logical_and(target, prediction) * 1.0
    tn = np.sum(tn_array)

    return tn


def get_recall(target, prediction, threshold):
    '''
    compute the recall rate

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    tp = get_TP(target, prediction, threshold)
    fn = get_FN(target, prediction, threshold)
    print('tp={0}'.format(tp))
    print('fn={0}'.format(fn))
    if tp + fn <= 0.0:
        recall = tp / (tp + fn + 1e-9)
    else:
        recall = tp / (tp + fn)
    return recall


def get_precision(target, prediction, threshold):
    '''
    compute the  precision rate

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    tp = get_TP(target, prediction, threshold)
    fp = get_FP(target, prediction, threshold)
    print('tp={0}'.format(tp))
    print('fp={0}'.format(fp))
    if tp + fp <= 0.0:
        precision = tp / (tp + fp + 1e-9)
    else:
        precision = tp / (tp + fp)
    return precision


def get_F1(target, prediction, threshold):
    '''
    compute the  F1 score

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    recall = get_recall(target, prediction, threshold)
    print(recall)
    precision = get_precision(target, prediction, threshold)
    print(precision)
    if precision == 0.0 or recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1


def get_accuracy(target, prediction, threshold):
    '''
    compute the accuracy rate

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    tp = get_TP(target, prediction, threshold)
    tn = get_TN(target, prediction, threshold)

    accuracy = (tp + tn) / target.size

    return accuracy


def get_relative_error(target, prediction):
    '''
    compute the  relative_error

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    '''

    assert (target.shape == prediction.shape)

    return np.mean(np.nan_to_num(np.abs(target - prediction) / np.maximum(target, prediction)))


def get_abs_error(target, prediction):
    '''
    compute the  absolute_error

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    '''

    assert (target.shape == prediction.shape)

    return np.mean(np.abs(target - prediction))


def get_nde(target, prediction):
    '''
    compute the  normalized disaggregation error

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    '''

    return np.sum((target - prediction) ** 2) / np.sum((target ** 2))


def get_sae(target, prediction, sample_second):
    '''
    compute the signal aggregate error
    sae = |\hat(r)-r|/r where r is the ground truth total energy;
    \hat(r) is the predicted total energy.
    '''
    r = np.sum(target * sample_second * 1.0 / 3600.0)
    rhat = np.sum(prediction * sample_second * 1.0 / 3600.0)

    return np.abs(r - rhat) / np.abs(r)

def get_sae_delta(target, prediction, step):
    r, rhat = [], []
    for i in range(0, target.shape[0], step):
        if i+step >= target.shape[0]:
            break
        r.append(np.sum(target[i:i+step]))
        rhat.append(np.sum(prediction[i:i+step]))
    r = np.array(r)
    rhat = np.array(rhat)
    
    return np.mean(np.abs(r-rhat)/step)
def get_macroF1(target, prediction, state_num):
    #TP True positive 正类预测为正类
    #FP False positive 负类预测为正类
    #TN True negative 正类预测为负类
    #FN False negative 负类预测为负类
    #p=TP/(TP+FP)
    #R=TP/(TP+FN)
    #F1=2*P*R/(P+R)
    epsilon = 1e-8
    macroF1=0.
    F1=np.zeros(state_num)
    for i in range(state_num):
        TP = epsilon
        FP = epsilon
        FN = epsilon
        TN = epsilon
        P = epsilon
        R = epsilon
        for j in range(len(target)):
            if target[j]==i and prediction[j]==i:
                TP+=1
            if target[j]==i and prediction[j]!=i:
                TN+=1
            if target[j]!=i and prediction[j]==i:
                FP+=1
            else:
                FN+=1
        P=TP/(TP+FP)
        R=TP/(TP+FN)
        F1[i]=(2*P*R)/(P+R)
    macroF1=np.mean(F1)
    return macroF1



appliance_thresholds = {
    'kettle': 2000,
    'fridge': 50,
    'washingmachine': 50,
    'washing_machine': 50,
    'microwave': 300,
    'dishwasher': 20,
}

def acc_precision_recall_f1_score(status,status_pred):
    assert status.shape == status_pred.shape
    
    if type(status)!=np.ndarray:
        status = status.detach().cpu().numpy().squeeze()   
    if type(status_pred)!=np.ndarray: 
        status_pred = status_pred.detach().cpu().numpy().squeeze()
    

    status      = status.reshape(status.shape[0], -1)
    status_pred = status_pred.reshape(status_pred.shape[0],-1)
    accs, precisions, recalls, f1_scores = [], [], [], []


    for i in range(status.shape[0]):
        tn, fp, fn, tp = confusion_matrix(status[i, :], status_pred[i, :], labels=[0, 1]).ravel()
        acc            = (tn + tp) / (tn + fp + fn + tp)
        precision      = tp / np.max((tp + fp, 1e-9))
        recall         = tp / np.max((tp + fn, 1e-9))
        f1_score       = 2 * (precision * recall) / np.max((precision + recall, 1e-9))

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return np.array(accs), np.array(precisions), np.array(recalls), np.array(f1_scores)

def get_F1(target, prediction, appliance):
    # Get the threshold for the appliance
    threshold = appliance_thresholds.get(appliance)
    if threshold is None:
        raise ValueError(f"Appliance '{appliance}' not found in thresholds.")
    
    # Convert target and prediction to binary classifications
    target_ = (target > threshold).astype(int)
    pred_ = (prediction > threshold).astype(int)

    _, _, _, f1 = acc_precision_recall_f1_score(target_, pred_)
    return f1