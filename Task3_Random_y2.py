# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 20:47:25 2019

@author: terry
"""

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from matplotlib.font_manager import FontProperties
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

set1 = np.load('normal_train.npy', allow_pickle=True)

set3 = set1.tolist()
X = set3['data_matrix']

Y1 = set3['components']
Y2 = set3['dist_type']


#for i in range(0, 22500):
#    X[i] = X[i][:990]

X_tr = []
plt_cm_type0, plt_cm_type1, plt_cm_type2 = [], [], []
cluster = 50
CUT = 50
#for CUT in range(1,cluster+1):
X_var, X_tr = [],[]
for i in range(0, 22500):
    X_mean_1 = np.mean(X[i])
    X_tr.append(X_mean_1)
    
    X_var_1 = np.var(X[i])
    X_tr.append(X_var_1)
    
    
    L = np.int(len(X[i])/CUT)
    for j in range(0,CUT):
        X_mean = np.mean(X[i][j*L:(j+1)*L])
        X_tr.append(X_mean)
#        X_max = np.max(X[i][j*L:(j+1)*L])
#        X_tr.append(X_max)
#        X_min = np.min(X[i][j*L:(j+1)*L])
#        X_dif = X_max - X_min
#        X_tr.append(X_dif)
        X_med = np.percentile(X[i][j*L:(j+1)*L],[10,25,50,75,90])
        X_tr.append(X_med[0])
        X_tr.append(X_med[1])
        X_tr.append(X_med[2])
        X_tr.append(X_med[3])
        X_tr.append(X_med[4])
       
#        X_tr.append(X_min)
        X_var = np.var(X[i][j*L:(j+1)*L])
        X_tr.append(X_var)
    ###        
X_tr = np.array(X_tr)
X_tr = X_tr.reshape(-1,CUT*7 + 2)
X_tr = DataFrame(X_tr)
Y1 = pd.DataFrame(Y1)

classifier = RandomForestClassifier(n_estimators=200, max_depth=40,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_tr, Y1 ,test_size = 0.2, random_state = 0)
classifier.fit(X_train, y_train)
#joblib.dump(classifier, "RandomF_150_30y1.m")
y_pre = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pre)
#cm_type0 = cm[0,0]
#cm_type1 = cm[1,1]
#cm_type2 = cm[2,2]
#print(cm_type2)
#    plt_cm_type0.append(cm_type0)
#    plt_cm_type1.append(cm_type1)
#    plt_cm_type2.append(cm_type2)

#plt.plot(plt_cm_type0)
#plt.show()
#
#plt.plot(plt_cm_type1)
#plt.show()
#
#plt.plot(plt_cm_type2)
#plt.show()
    ####
#    X_mean2 = np.mean(X[i][L:2*L])
#    X_mean3 = np.mean(X[i][2*L:3*L])
#    X_mean4 = np.mean(X[i][3*L:4*L])
#    X_mean5 = np.mean(X[i][4*L:5*L])
#    X_mean6 = np.mean(X[i][5*L:6*L])
#    X_mean7 = np.mean(X[i][6*L:7*L])
#    X_mean8 = np.mean(X[i][7*L:8*L])
#    X_mean9 = np.mean(X[i][8*L:9*L])
#    X_mean10 = np.mean(X[i][9*L:10*L])
#    X_mean11 = np.mean(X[i][10*L:len(X[i])])
#    X_tr.append(X_mean1)
#    X_tr.append(X_mean2)
#    X_tr.append(X_mean3)
#    X_tr.append(X_mean4)
#    X_tr.append(X_mean5)
#    X_tr.append(X_mean6)
#    X_tr.append(X_mean7)
#    X_tr.append(X_mean8)
#    X_tr.append(X_mean9)
#    X_tr.append(X_mean10)
#    X_tr.append(X_mean11)
    
    
#    X_var1 = np.var(X[i][0:L])
#    X_var2 = np.var(X[i][L:2*L])
#    X_var3 = np.var(X[i][2*L:3*L])
#    X_var4 = np.var(X[i][3*L:4*L])
#    X_var5 = np.var(X[i][4*L:5*L])
#    X_var6 = np.var(X[i][5*L:6*L])
#    X_var7 = np.var(X[i][6*L:7*L])
#    X_var8 = np.var(X[i][7*L:8*L])
#    X_var9 = np.var(X[i][8*L:9*L])
#    X_var10 = np.var(X[i][9*L:len(X[i])])
#    X_tr.append(X_var1)
#    X_tr.append(X_var2)
#    X_tr.append(X_var3)
#    X_tr.append(X_var4)
#    X_tr.append(X_var5)
#    X_tr.append(X_var6)
#    X_tr.append(X_var7)
#    X_tr.append(X_var8)
#    X_tr.append(X_var9)
#    X_tr.append(X_var10)


#X_tr = np.array(X_tr)
#X_tr = X_tr.reshape(-1,CUT*2)
#X_tr = DataFrame(X_tr)
###p_col_x = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20']
###X_tr.columns = p_col_x
#Y2 = pd.DataFrame(Y2)
###p_col_y = ['Y2']
###Y2.columns = p_col_y
###data = X_tr
###data['Y2'] = Y2.values
##3z = np.abs(stats.zscore(data))
##3data = data[(z < 3).all(axis=1)]
###Y2 = data['Y2']
###data = data.drop(['Y2'], axis = 1)
#
#
#
#
#X_train, X_test, y_train, y_test = train_test_split(X_tr, Y2 ,test_size = 0.25, random_state = 0)
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#classifier.fit(X_train, y_train)
#y_pre = classifier.predict(X_test)
#
#cm = confusion_matrix(y_test, y_pre)



