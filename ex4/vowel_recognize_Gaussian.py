"""ESL-python ex4.9 
A program to perform a quadratic discriminant analysis by fitting
a separate Gaussian model per class on the vowel data, and compute
the misclassification error for the test data.

Created on Oct 20th. 2018
@ author Yan
"""

import seaborn as sns; sns.set()
from sklearn.naive_bayes import GaussianNB

def readData(file_name):
    # read data from file and save them in float list
    f = open(file_name)
    all_lines = f.read().splitlines()[1:]
    x_input = []
    y_output = []
    temp_list = []
    for line in all_lines:
        line = line.strip().split(',')
        y_output.append(float(line[1]))
        for number in line[2:]:
            temp_list.append(float(number))
        x_input.append(temp_list)
        temp_list = []
    return x_input, y_output

# train and predict
train_x, train_y = readData('vowel.train')
test_x, test_y = readData('vowel.test')
model = GaussianNB()
model.fit(train_x, train_y)
predict_y = list(model.predict(test_x))

# compare the results of predictions and the actual
true_num = 0
for i in range(len(test_x)):
    if test_y[i] == predict_y[i]:
        true_num += 1
true_rate = true_num/(len(test_x))
