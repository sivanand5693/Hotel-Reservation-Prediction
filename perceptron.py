# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 06:50:39 2021

@author: sivan
"""

#import Data_preprocessing as dat
import helper_functions as helper

from sklearn.linear_model import Perceptron
import numpy as np
import random
import pandas as pd

seed = 0
random.seed(seed)
np.random.seed(seed)

epochs=5
accuracy_list=[]

hotel_train = pd.read_csv('.\data\hotel_train.csv')
hotel_test = pd.read_csv('.\data\hotel_test.csv')

# training

#no regularization (penalty), max_iter is #epochs
per = Perceptron(penalty=None, fit_intercept=True,max_iter=1000, tol=1e-3, shuffle=True,
                 eta0=1, random_state=seed)
per_l1=Perceptron(penalty='l1', fit_intercept=True,max_iter=1000, tol=1e-3, shuffle=True,
                 eta0=1, random_state=seed)
per_l2=Perceptron(penalty='l2', fit_intercept=True,max_iter=1000, tol=1e-3, shuffle=True,
                 eta0=1, random_state=seed)
per_elas=Perceptron(penalty='elasticnet', fit_intercept=True,max_iter=1000, tol=1e-3, shuffle=True,
                 eta0=1, random_state=seed)

def perceptron_algo(train,test,response='reservation_status'):
    X_train = train.drop([response], axis=1)
    Y_train = train[response]
    X_test = test.drop([response], axis=1)
    Y_test = test[response]
        
    #fitting perceptron 
    per.fit(X_train,Y_train)
    per_l1.fit(X_train,Y_train)
    per_l2.fit(X_train,Y_train)
    per_elas.fit(X_train,Y_train)

    print("\n No Regularization")
    print("Iterations to converge: %d" %(per.n_iter_)) #attributes
    print("Training Accuracy: %0.3f%%" %(100*per.score(X_train,Y_train))) #using score function -> accuracy
    print(" Testing Accuracy: %0.3f%%" %(100*per.score(X_test,Y_test))) #testing data
    
    print("\n L1 Regularization")
    print("Iterations to converge: %d" %(per_l1.n_iter_))
    print("Training Accuracy: %0.3f%%" %(100*per_l1.score(X_train,Y_train))) 
    print(" Testing Accuracy: %0.3f%%" %(100*per_l1.score(X_test,Y_test))) 
     
    print("\n L2 Regularization")
    print("Iterations to converge: %d" %(per_l2.n_iter_))
    print("Training Accuracy: %0.3f%%" %(100*per_l2.score(X_train,Y_train))) 
    print(" Testing Accuracy: %0.3f%%" %(100*per_l2.score(X_test,Y_test)))
    
    print("\n Elasticnet Regularization")
    print("Iterations to converge: %d" %(per_elas.n_iter_))
    print("Training Accuracy: %0.3f%%" %(100*per_elas.score(X_train,Y_train))) 
    print(" Testing Accuracy: %0.3f%%" %(100*per_elas.score(X_test,Y_test)))    


perceptron_algo(hotel_train,hotel_test)

