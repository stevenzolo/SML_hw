"""handwriting discrimination batch processing with KNN:
   read all data files in a directory, process them and get their average rate
   Created on Sep 23th

   @author: Yan
"""

import os
import operator
import numpy as np

def classify0(inX, dataSet, labels):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(3):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet-np.tile(minVals, (m,1))
    normDataSet = np.array(normDataSet/np.tile(ranges, (m,1)))   #element wise divide
    return normDataSet, ranges, minVals

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def getLabels(folder_name):
    labels = []
    upper_path = "D:\python_spyder\code\statisitc_machine_learning_hw\chapter2_dateweb_knn\digits" # target folder directory
    path = upper_path + '/' + folder_name
    files= os.listdir(path) # get all file names in the directory
    for file in files: # traversing folder
         if not os.path.isdir(file): # if not a folder, open it
             labels.append(int(list(file)[0]))
    return labels

def getDataSet():
    dataSet = []
    path = "D:\python_spyder\code\statisitc_machine_learning_hw\chapter2_dateweb_knn\digits\\trainingDigits" # target folder directory
    files= os.listdir(path) # get all file names in the directory
    for file in files: # traversing folder
         if not os.path.isdir(file): # if not a folder, open it
              file_name = path+"/"+file
              dataSet.append(img2vector(file_name))
    dataSet = np.array(dataSet)
    return dataSet

def batch_handwriting():
    labels_predic = []; true_rate = 0
    normDataSet, ranges, minVals = autoNorm(getDataSet())
    train_labels = getLabels('trainingDigits')
    path = "D:\python_spyder\code\statisitc_machine_learning_hw\chapter2_dateweb_knn\digits\\testDigits" # target folder directory
    files= os.listdir(path) # get all file names in the directory
    for file in files: # traversing folder
         if not os.path.isdir(file): # if not a folder, open it
              file_name = path+"/"+file
              inSet = img2vector(file_name)
              label_predic = classify0((inSet-minVals)/ranges, normDataSet, train_labels)
              labels_predic.append(label_predic)
    labels_fact = getLabels('testDigits')
    for i in range(len(labels_fact)):
        if labels_fact[i] == labels_predic[i]:
            true_rate += 1
    mean_rate = true_rate / len(labels_fact)
    return mean_rate
