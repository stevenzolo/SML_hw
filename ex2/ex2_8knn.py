"""homework of statisitic machine learning
   do Ex2_8 with KNN
   some functions are excerpted from book <Machine Learning In Action>

   @author Yan
"""


import numpy as np
import operator


# define a classification function of KNN, cited from book ML in action
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# read train and test file
train_file = open('zip.train','r')
train_lines = train_file.read().splitlines()[:]
train_file.close()
test_file = open('zip.test','r')
test_lines = test_file.read().splitlines()[:]
test_file.close()

train_data = []
train_label = []
for line in train_lines:
    train_label.append(line.split()[0])
    train_data.append(line.split()[1:])

test_data = []
test_label = []
for line in test_lines:
    test_label.append(line.split()[0])
    test_data.append(line.split()[1:])


# transfer string data to int or float
row_temp = []
train_data_float = []
for row in train_data:
    for item in row:
        row_temp.append(float(item))
    train_data_float.append(row_temp)
    row_temp = []

train_label_int = []
for item in train_label:
    train_label_int.append(float(item))

test_data_float = []
for row in test_data:
    for item in row:
        row_temp.append(float(item))
    test_data_float.append(row_temp)
    row_temp = []

test_label_int = []
for item in test_label:
    test_label_int.append(float(item))

# for every dataset in test, use classify0() to classify
trainSet = np.asarray(train_data_float)
labelSet = np.asarray(train_label_int)
k = 5
label_predict = []
for row in range(len(test_data_float)):
    test_vector = np.asarray(test_data_float[row])
    predict_value = classify0(test_vector, trainSet, labelSet, k)
    label_predict.append(predict_value)

# get the true rate
true_predict = 0
for item in range(len(label_predict)):
    if label_predict[item] == test_label_int[item]:
        true_predict += 1
true_rate = true_predict/len(label_predict)
