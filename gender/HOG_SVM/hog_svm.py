import os
import cv2
import random
import pathlib
import numpy as np
import multiprocessing
from sklearn import svm


from skimage.morphology import skeletonize
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score
import sys

base_path = "/media/D/lulei/classification"
sys.path.insert(0, base_path)
from tools.utils.utils import *


winSize = (256, 256)
blockSize = (64, 64)
blockStride = (32, 32)
cellSize = (32,32)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 1
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True


cvhog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
                        derivAperture,winSigma,histogramNormType,L2HysThreshold,
                        gammaCorrection,nlevels,signedGradients)

root_dir = "/media/D/lulei/data/gender/origin"
class_dirs = [os.path.join(root_dir, i) for i in os.listdir(root_dir)]

print(class_dirs)

x_train, y_train = [], []
x_test, y_test = [], []

for class_label, class_dir in enumerate(class_dirs):
    train_list, test_list = list_split(get_all_images(class_dir, shuffle = False),
                                       ratio = [0.4, 0.6], shuffle = False)
    #try:
    for image_path in train_list:    
            
        roi = imread(image_path, is_gray = True, width = 256, height = 256)
        feature = cvhog.compute(roi)
        x_train.append(feature)
        y_train.append(class_label)
 
    for image_path in test_list:    
        roi = imread(image_path, is_gray = True, width = 256, height = 256)
        feature = cvhog.compute(roi)
        x_test.append(feature)
        y_test.append(class_label)
    #except:
        #print(image_path)


f_dim = x_test[0].shape[0]
print(f"feature dim : {f_dim}")

x_train = np.asarray(x_train).reshape(-1, f_dim)
x_test = np.asarray(x_test).reshape(-1, f_dim)

# Grid Search
# Parameter Grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}

# Make grid search classifier
clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1, cv = 3, n_jobs = -1) #, pre_dispatch = cpu_count//2

# Train the classifier
clf_grid.fit(x_train, y_train)

# clf = grid.best_estimator_()
print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)

# Make predictions on unseen test data
y_pred = clf_grid.predict(x_test)
# print("Accuracy: {}%".format(clf_grid.score(x_test, y_test) * 100 ))

overall_acc = overall_accuracy(y_test, y_pred)
m_prec = mean_prec(y_test, y_pred)
m_recall = mean_recall(y_test, y_pred)

print(f"Over all acc : {overall_acc*100:.2f}%\nmean prec : {m_prec*100:.2f}%\nmean recall : {m_recall*100:.2f}%")
