
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import glob
import csv
import pickle


# In[2]:
y_train_melanoma = []
x_train_melanoma = []
counter = 0
ctr = 1

# Training Dataset Generation
w = 64
with open('Dataset/ISIC-2017_Training_Part3_GroundTruth_melanoma.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    y_train_melanoma = []
    x_train_melanoma = []
    sw = 0
    ctr2 = 0
    for row in readCSV:
        if sw == 1:
            print(ctr2)
            if(ctr2 < 10000):
                ctr2 = ctr2 + 1
                re_label = [float(row[1][0])]
                y_train_melanoma.append(re_label)
                filename = 'Dataset/ISIC-2017_Training_Data/' + row[0] +'.jpg'            
                
                #reading the corresponding image
                img = cv2.imread(filename)
                
                img = cv2.resize(img,(w,w))
                x_train_melanoma.append(img)

                #Use this one if you did not used external training data to balance the dataset

                # if int(row[1][0]) == 1:
                #     s = np.random.randint(2, size=(128, 128))
                #     img2 = img * s
                #     x_train_melanoma.append(img)
                #     y_train_melanoma.append(re_label)

                #     s = np.random.randint(2, size=(128, 128))
                #     img2 = img * s
                #     x_train_melanoma.append(img)
                #     y_train_melanoma.append(re_label)

                #     s = np.random.randint(2, size=(128, 128))
                #     img2 = img * s
                #     x_train_melanoma.append(img)
                #     y_train_melanoma.append(re_label)

                #     s = np.random.randint(2, size=(128, 128))
                #     img2 = img * s
                #     x_train_melanoma.append(img)
            #     y_train_melanoma.append(re_label)
                
        sw = 1

y_train_melanoma = np.reshape(y_train_melanoma, [-1, 1])
x_train_melanoma = np.reshape(x_train_melanoma, [-1, w, w, 3])

train_melanoma = {'features': x_train_melanoma, 'labels': y_train_melanoma}
pickle.dump(train_melanoma, open("ISIC2017_train_melanoma_colored_"+str(w)+".p", "wb"))

print("x_train_melanoma: ",len(x_train_melanoma))
print("num labels: ",len(y_train_melanoma))
print("Done with training set")

#Validation Set
with open('Dataset/ISIC-2017_Validation_Part3_GroundTruth.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    y_val_melanoma = []
    x_val_melanoma = []
    sw = 0
    for row in readCSV:
        if sw == 1:
            re_label = [float(row[1][0])]
            y_val_melanoma.append(re_label)
            filename = 'Dataset/ISIC-2017_Validation_Data/' + row[0] +'.jpg'            
            
            #reading the corresponding image
            img = cv2.imread(filename)
            
            img = cv2.resize(img,(w,w))

            x_val_melanoma.append(img)          
    
        sw = 1

y_val_melanoma = np.reshape(y_val_melanoma, [-1, 1])
x_val_melanoma = np.reshape(x_val_melanoma, [-1, w, w, 3])

val_melanoma = {'features': x_val_melanoma, 'labels': y_val_melanoma}
pickle.dump(val_melanoma, open("ISIC2017_val_melanoma_colored_"+str(w)+".p", "wb"))

print("x_val_melanoma: ",len(x_val_melanoma))
print("num labels: ",len(y_val_melanoma))
print("Done with validation set")

# In[4]:

# Testing Dataset Generation

with open('Dataset/ISIC-2017_Test_v2_Part3_GroundTruth.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    y_test_melanoma = []
    x_test_melanoma = []
    sw = 0
    for row in readCSV:
        if sw == 1:
            re_label = [float(row[1][0])]
            y_test_melanoma.append(re_label)
            filename = 'Dataset/ISIC-2017_Test_v2_Data/' + row[0] +'.jpg'            
            
            #reading the corresponding image
            img = cv2.imread(filename)
            
            img = cv2.resize(img,(w,w))

            x_test_melanoma.append(img)          
    
        sw = 1

y_test_melanoma = np.reshape(y_test_melanoma, [-1, 1])
x_test_melanoma = np.reshape(x_test_melanoma, [-1, w, w, 3])

test_melanoma = {'features': x_test_melanoma, 'labels': y_test_melanoma}
pickle.dump(test_melanoma, open("ISIC2017_test_melanoma_colored_"+str(w)+".p", "wb"))

print("x_test_melanoma: ",len(x_test_melanoma))
print("num labels: ",len(y_test_melanoma))
print("Done with testing set")

# In[4]: