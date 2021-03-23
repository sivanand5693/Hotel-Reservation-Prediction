# -*- coding: utf-8 -*-


################### to import from another file ########################

#import file_paths as path #run file_paths.py first

#print(path.blah)
#########################################################################





import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import hyperparameters as par   #user created

seed = 0
random.seed(seed)
np.random.seed(seed)


#loading data file
hotel=pd.read_csv('C:\Purdue\Spring 2020\CS 578\Project\hotel\Final Submission\Hotel-Reservation-Prediction\hotel_bookings.csv')
#hotel = pd.read_csv('hotel_bookings.csv')

#details about dataset

print(hotel.shape)
print(hotel.columns)
print(hotel.head())

######################## Preprocessing Begins ##########################
hotel_upd = hotel[hotel['adults']!=0]

#creating new feature
def f(row):
    if row['reserved_room_type'] == row['assigned_room_type']:
        val = 1
    else:
        val = 0
    return val

hotel_upd['reserved_eq_assigned_roomtype'] = hotel_upd.apply(f, axis=1)

def f2(row):
    if row['adr']<=0:
        val = 0
    else:
        val = row['adr']
    return val
    
hotel_upd['adr'] = hotel_upd.apply(f2, axis=1)

#creating new column for Checkout 
def f3(row):
    if row['reservation_status']=='Check-Out':
        val = 1
    else:
        val = 0
    return val

hotel_upd['CheckOut'] = hotel_upd.apply(f3, axis=1)

#removing NA values
def f4(row):
    if row['children']=='NA':
        val = 0
    else:
        val = row['adr']
    return val
    
hotel_upd['children'] = hotel_upd.apply(f4, axis=1)

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

#feature selection

#mutual information

col_cat = ['hotel','arrival_date_month','meal','market_segment','distribution_channel','deposit_type','customer_type','reserved_eq_assigned_roomtype']
info_list=[]
join_coly = 'reservation_status'

#marginal prob of response
marproby = hotel_upd.groupby(['reservation_status']).size().reset_index().rename(columns={0:'marcounty'})
marproby['marproby']= marproby['marcounty']/len(hotel_upd)


for i in col_cat:
        
    #marginal prob of predictor
    marprobx = hotel_upd.groupby([i]).size().reset_index().rename(columns={0:'marcountx'})
    marprobx['marprobx']= marprobx['marcountx']/len(hotel_upd)

    #joint probabilities 
    joinprob = hotel_upd.groupby([i,'reservation_status']).size().reset_index().rename(columns={0:'joincount'})
    joinprob['joinprob']= joinprob['joincount']/len(hotel_upd)

    #merging on column of interest
    join_colx = joinprob.columns[0]

    totalprob = pd.merge(joinprob,marprobx,on=join_colx,how='left')
    totalprob = pd.merge(totalprob,marproby,on=join_coly,how='left')

    #information calculation
    totalprob['info'] = totalprob['joinprob']*(np.log(totalprob['joinprob']/(totalprob['marprobx']*totalprob['marproby'])))
    info_list.append(sum(totalprob['info']))

col_cat=pd.DataFrame(col_cat)
info_list=pd.DataFrame(info_list)
frames = [col_cat, info_list]
mutualinfo = pd.concat(frames, axis=1)
mutualinfo.columns = ('categorical_feature','mutual_info')

##checking categories###
#hotel_upd['deposit_type'].value_counts()


#dummy variables columns
hotel_upd[['market_Aviation','market_Complntry','market_Corprt','market_Direct','mar_Groups','market_OffTA','market_OnTA','market_Undef']]=pd.get_dummies(hotel_upd['market_segment'])
hotel_upd[['distr_Corprt','distr_Direct','distr_GDS','distr_TATO','distr_Undef']]=pd.get_dummies(hotel_upd['distribution_channel'])
hotel_upd[['No_Deposit','Non_Refund','Refundable']]=pd.get_dummies(hotel_upd['deposit_type'])


#removing irrelevent columns and those with mutual information <0.01
hotel_upd=hotel_upd.drop(['hotel','is_canceled','arrival_date_year','arrival_date_month','arrival_date_week_number','arrival_date_day_of_month','meal','country','reserved_room_type','assigned_room_type','agent','company','customer_type','reservation_status_date','CheckOut'],axis=1)

#removing categorical columns and last dummy category for each
hotel_upd=hotel_upd.drop(['market_segment','market_Undef',
                          'distribution_channel','distr_Undef',
                          'deposit_type','Refundable'],axis=1)


########### Dataset of different sizes ######

if(par.binary_classification==True):
    hotel_upd = hotel_upd[hotel_upd['reservation_status'] != 'No-Show']


hotel_sample = hotel_upd.sample(n=par.total_size,random_state=seed,replace=False)
#hotel_sample = hotel_upd.sample(n=10000,random_state=seed,replace=False)

#train & test
hotel_train, hotel_test = train_test_split(hotel_sample, test_size=par.percentage_test, random_state=seed)
#hotel_train, hotel_test = train_test_split(hotel_sample, test_size=0.2, random_state=seed)


#count of response variable
print(hotel_train['reservation_status'].value_counts())
print(hotel_test['reservation_status'].value_counts())


