#!/usr/bin/env python
import mxnet as mx
import numpy as np
import math

def LogLoss(label, pred):
	label =label.T.reshape((-1,))
	loss = 0.
	min_num = 1e-13
	max_num = 1 - min_num
	for i in range(pred.shape[0]):
		p = min(max(pred[i][0],min_num), max_num)
		if label[i] > 0.5:
			loss += -math.log(p)
		else:
			loss += -math.log(1-p)
			
	return loss / label.shape[0]

def AUC(label, pred):
	num = len(pred)
	i_sorted = sorted(range(num), key = lambda i : pred[i][0], reverse = True)
	auc_temp = 0.0
	tp = 0.0
	tp_pre = 0.0 
	fp = 0.0
	fp_pre = 0.0
	last_value = pred[i_sorted[0]]
	for i in range(num):
	    if label[i_sorted[i]] > 0.1:
	        tp+=1
	    else:
	        fp+=1
        if last_value != pred[i_sorted[i]]:
            auc_temp += ( tp + tp_pre ) * ( fp - fp_pre)
            tp_pre = tp
            fp_pre = fp
            last_value = pred[i_sorted[i]]
	auc_temp += ( tp + tp_pre ) * ( fp -fp_pre )
	return auc_temp / (2.0 * tp * fp)
