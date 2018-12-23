"""ESL-python ex7.9
a program to practice cross validation with prostate data.
Created on Nov 22th.
@ author Yan
"""

import numpy as np
import random

def readData(file_name, foldtime):
    # read data from file and sparse into train and test
    f = open(file_name)
    all_lines = f.read().splitlines()[1:]
    f.close()
    random.shuffle(all_lines)  # randomize lines
    label_list = []; temp_list = []; data_list = [];
    for line in all_lines:
        line = line.strip().split()[1:-1]
        data = line[:-1]
        label = line[-1]
        label_list.append(float(label))
        for datum in data:
            temp_list.append(float(datum))
        data_list.append(temp_list)
        temp_list = []
    break_location = int(len(all_lines)*(int(foldtime-1))/int(foldtime))
    data_train = data_list[:break_location]
    data_test = data_list[break_location:]
    label_train = label_list[:break_location]
    label_test = label_list[break_location:]
    return data_train, data_test, label_train, label_test

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))     # sigmoid function

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)             #convert to NumPy matrix
    labelMat = np.mat(classLabels).transpose() #convert to NumPy matrix
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

def FoldValid(file_name, foldtime):
    predict_list = []; error_list = []
    for i in range(int(foldtime)):
        data_train, data_test, label_train, label_test = readData(file_name, foldtime)
        weights_train = gradAscent(data_train, label_train)
        predict_value = np.dot(np.mat(data_test), weights_train)
        predict_value = [value/100000 for value in predict_value] # fit data
        predict_error = []
        for i in range(len(label_test)):
            temp_error = ((predict_value[i]-label_test[i])**2) / len(label_test)
            predict_error.append(temp_error)  # calculate error
        predict_list.append(predict_value)
        error_list.append(predict_error)
        predict_error = []
    min_index = error_list.index(min(error_list))
    return predict_list[min_index], error_list[min_index]

if __name__ == '__main__':
    fivefold_predict, fifth_error = FoldValid('prostate.txt', 5)
    tenfold_predict, tenth_error = FoldValid('prostate.txt', 10)
