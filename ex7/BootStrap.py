"""ESL-python ex7.9
a program to practice bootstrap with prostate data.
Created on Nov 22th.
@ author Yan
"""

import numpy as np
import random
from crossValidation import *

def selectTrainSample(file_name):
    # select m sample lines, including repeated line
    f = open(file_name)
    all_lines = f.read().splitlines()[1:]
    f.close()
    copy_line = []
    for i in range(len(all_lines)):
        random.shuffle(all_lines)
        sample_line = all_lines[0]
        copy_line.append(sample_line)
    train_sample = removeDupli(copy_line)
    test_sample = []
    for elem in all_lines:
        if elem not in train_sample:
            test_sample.append(elem)
    return train_sample, test_sample

def removeDupli(dulpicate_list):
    # remove duplicated elements in a list
    new_list = []
    for elem in dulpicate_list:
        if elem not in new_list:
            new_list.append(elem)
    return new_list

def sparseData(dataSample):
    # sparse sample into data and label
    label_list = []; temp_list = []; data_list = [];
    for line in dataSample:
        line = line.strip().split()[1:-1]
        data = line[:-1]
        label = line[-1]
        label_list.append(float(label))
        for datum in data:
            temp_list.append(float(datum))
        data_list.append(temp_list)
        temp_list = []
    return data_list, label_list

def calcError(file_name):
    train_sample, test_sample = selectTrainSample(file_name)
    data_train, label_train = sparseData(train_sample)
    data_test, label_test = sparseData(test_sample)
    weights_train = gradAscent(data_train, label_train)
    predict_value = np.dot(np.mat(data_test), weights_train)
    predict_value = [value/100000 for value in predict_value] # fit data
    predict_error = []
    for i in range(len(label_test)):
        temp_error = ((predict_value[i]-label_test[i])**2) / len(label_test)
        predict_error.append(temp_error)  # calculate error
    return predict_value, predict_error

if __name__ == '__main__':
    predict_value, predict_error = calcError('prostate.txt')
