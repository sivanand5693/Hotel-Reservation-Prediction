# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:28:10 2021

@author: sivan
"""

import numpy as np

#define accuracy measure(reservation_status = predicted reservation status)
def accuracy_score(row):
    if row[0]==row[1]:
        val=1
    else:
        val=0
    return val

#creating function to obtain ROC parameters
def calcRates(y_vals,probs):
    
    #convert to arrays
    y_vals = np.array(y_vals)
    probs = np.array(probs)

    # sort the indexs by their probabilities
    index = np.argsort(probs, kind="quicksort")[::-1]
    probs = probs[index]
    y_vals = y_vals[index]

    #Grab indices with distinct values
    d_index = np.where(np.diff(probs))[0]
    t_holds = np.r_[d_index, y_vals.size - 1]

    # sum up with true positives       
    tps = np.cumsum(y_vals)[t_holds]
    tpr = tps/tps[-1]
    #calculate the false positive
    fps = 1 + t_holds - tps
    fpr = fps/fps[-1]
    return fpr, tpr
