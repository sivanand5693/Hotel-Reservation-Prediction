# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 06:50:39 2021

@author: sivan
"""

#import Data_preprocessing as dat
import helper_functions as helper
import hyperparameters as par

from sklearn.linear_model import Perceptron
import numpy as np
import random
import pandas as pd

seed = 0
random.seed(seed)
np.random.seed(seed)


hotel_train = pd.read_csv('.\data\hotel_train.csv')
hotel_test = pd.read_csv('.\data\hotel_test.csv')


######################### sklearn Perceptron ################################

#no regularization (penalty), max_iter is #epochs
per = Perceptron(penalty=None, fit_intercept=True,max_iter=1000, tol=1e-3, shuffle=True,
                 eta0=1, random_state=seed)
per_l1=Perceptron(penalty='l1', fit_intercept=True,max_iter=1000, tol=1e-3, shuffle=True,
                 eta0=1, random_state=seed)
per_l2=Perceptron(penalty='l2', fit_intercept=True,max_iter=1000, tol=1e-3, shuffle=True,
                 eta0=1, random_state=seed)
per_elas=Perceptron(penalty='elasticnet', fit_intercept=True,max_iter=1000, tol=1e-3, shuffle=True,
                 eta0=1, random_state=seed)

def perceptron_algo(train,test,response='binary_response'):
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

###############################################################################



########################## Perceptron ################################

accuracy_list=[]

class perceptron_implement():
    def __init__(self,train,test,response='binary_response'):
        self.train=train
        self.test=test
        self.response=response
        self.nsize = train.shape[0]
        self.npar = train.shape[1] - 1 #contains response
        self.weights = [0.0] * self.npar
        self.bias = 0.0                 
        
        
    def training(self):
        X_train = self.train.drop([self.response], axis=1)
        Y_train = self.train[self.response]
        
        
        for epoch in range(par.epochs):
            update_counter=0
            for n in range(self.nsize):
            
                decision_boundary = Y_train.iloc[n] * (X_train.iloc[n].dot(self.weights) + self.bias)
            
                if(decision_boundary<=0.0):
                    update_counter+=1
                    self.weights += Y_train.iloc[0]*X_train.iloc[0]
                    self.bias += Y_train.iloc[0]
            print("Epoch: %d Accuracy: %0.3f%%" %(epoch+1, 100*update_counter/self.nsize))   
        
        #print(update_counter)


#percept=perceptron_implement(hotel_train,hotel_test)
#percept.training()

###############################################################################