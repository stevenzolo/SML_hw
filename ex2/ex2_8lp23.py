"""homework of statisitic machine learning
   do Ex2_8 with Linear Regress
   some functions are excerpted from book <Machine Learning In Action>

   @author Yan
"""


import numpy as np

def selectLines(old_file, new_file):
    # select 2's and 3's data, write them into a new file
    fr = open(old_file,'r')
    gr = open(new_file,'w')
    for line in fr.readlines():
        lineArr = line.strip().split()
        if float(lineArr[0]) == 2 or float(lineArr[0]) == 3:
            gr.write(line)
    fr.close()
    gr.close()

def loadDataSet(file_name):
    # from list_string to list_float
    dataMat = []; labelMat = []
    fr = open(file_name)
    data_row = []
    for line in fr.readlines():
        lineArr = line.strip().split()
        for item in lineArr:
            data_row.append(float(item))
        dataMat.append([1.0] + data_row[1:])
        data_row = []
        labelMat.append(float(lineArr[0]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX)) + 2   # project real number in range2-3

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

hlist = []
def predict_label(test_vector):
    h = sigmoid(float(np.dot(np.mat(test_vector), np.mat(train_weights))))
    hlist.append(h)
    if abs(h-2) <= abs(h-3):
        predict_value = 2
    else:
        predict_value = 3
    return predict_value

# train the data with function above, get train weights
selectLines('zip.train','train23.txt')
train_data, train_label = loadDataSet('train23.txt')
train_weights = gradAscent(train_data, train_label)

# test data, get predict_array
selectLines('zip.test','test23.txt')
test_data, test_label = loadDataSet('test23.txt')
predict_array = []
for line in test_data:
    predict_value = predict_label(line)
    predict_array.append(predict_value)

true_predict = 0
for i in range(len(test_label)):
    if test_label[i] == predict_array[i]:
        true_predict += 1
true_rate = true_predict/len(test_label)
