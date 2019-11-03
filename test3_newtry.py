#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:30:19 2019

@author: shijiayang
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# 安装 TensorFlow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#mnist = tf.keras.datasets.mnist

#test data
data1 = np.load('normal_train.npy', allow_pickle=True)
data1_list = data1.tolist()
x_train = data1_list['data_matrix']
# =============================================================================
# len_min = 2000
# for i in range(0,len(x_train)):
#     if len(x_train[i])<len_min:
#         len_min = len(x_train[i])
#     
# =============================================================================
#normaliazation
x_trainpre = np.zeros([len(x_train),900])
for i in range(0,len(x_train)):
    x_trainpre[i] = (x_train[i][:900]-np.mean(x_train[i][:900]))/(np.max(x_train[i][:900])-np.min(x_train[i][:900]))
    
y_train = data1_list['dist_type']
y_trainpre = y_train
y1_trainpre = data1_list['components']

#test data
data2 = np.load('normal_test.npy', allow_pickle=True)
data2_list = data1.tolist()
x_test = data2_list['data_matrix']
# =============================================================================
# len_min = 2000
# for i in range(0,len(x_train)):
#     if len(x_train[i])<len_min:
#         len_min = len(x_train[i])
#     
# =============================================================================
#normaliazation
x_testpre = np.zeros([len(x_test),900])
for i in range(0,len(x_test)):
    x_testpre[i] = (x_test[i][:900]-np.mean(x_test[i][:900]))/(np.max(x_test[i][:900])-np.min(x_test[i][:900]))
    
y_test = data2_list['dist_type']
y_testpre = y_test
y1_testpre = data2_list['components']


#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(900, )),
  tf.keras.layers.Flatten(input_shape=(1800, )),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(21, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("Data Type")
model.fit(x_trainpre, y_trainpre, epochs=50)

model.evaluate(x_testpre,  y_testpre, verbose=2)

print("Number of Components")
model.fit(x_trainpre, y1_trainpre, epochs=50)

model.evaluate(x_testpre,  y1_testpre, verbose=2)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
classifier = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)
print("Data Type")
classifier.fit(x_trainpre, y_trainpre)
y_pre = classifier.predict(x_testpre)
cm1 = confusion_matrix(y_testpre, y_pre)
acc1 = (cm1[1,1]+cm1[2,2]+cm1[0,0])/sum(sum(cm1))*100
print("accuracy:",acc1,"%")

print("Number of Components")
classifier.fit(x_trainpre, y1_trainpre)
y_pre = classifier.predict(x_testpre)
cm2 = confusion_matrix(y1_testpre, y_pre)
acc2 = (cm2[1,1]+cm2[2,2]+cm2[0,0])/sum(sum(cm2))*100
print("accuracy:",acc2,"%")