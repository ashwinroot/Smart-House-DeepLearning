#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:33:53 2017

@author: ashwinsankar
"""

from math import exp
from math import log
import numpy as np
import matplotlib.pyplot as plt
from random import random
import csv


def baseline():
    sumerror = 0
    correctval = -1
    unique,count = np.unique(occupancy,return_counts=True)
    if(count[0]>count[1]):
        correctval = 0
    else:
        correctval = 1
    for i in occupancy:
        sumerror += error_function(i,1)
    return sumerror / (count[0]+count[1])


def min_max_norm(P):
    P = (P - np.amin(P)) / (np.amax(P) - np.amin(P)) * 2 + -1
    return P

def time_normalisaiton(time):
    timex = []
    for i in range(0,len(time)):
        timex.append(int(time[i].split(' ')[1].split(':')[0] + time[i].split(' ')[1].split(':')[1]))
    return min_max_norm(np.array(timex))
dataset = []
iter_grp = []
error_grp = []
accuracy = []
train_accuracy_grp = []
baseline_grp = []
lrn_rate_grp = []
with open('/Users/ashwinsankar/Desktop/DL/HW2/Code and Dataset/train_data.csv', 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    for row in data:
        dataset.append(row)
dataset.pop(0)
# np.random.shuffle(dataset)  ##uncomment this to shuffle
# assigning from data to specific varaibels
order = np.array([float(row[0]) for row in dataset])
time = np.array([row[1] for row in dataset])
time = time_normalisaiton(time)
temperature = min_max_norm(np.array([float(row[2]) for row in dataset]))
humidity = min_max_norm(np.array([float(row[3]) for row in dataset]))
light = min_max_norm(np.array([float(row[4]) for row in dataset]))
co2 = min_max_norm(np.array([float(row[5]) for row in dataset]))
humidityr = min_max_norm(np.array([float(row[6]) for row in dataset]))
bias = np.array(np.ones(time.shape))
occupancy = np.array([int(row[7]) for row in dataset])
X = np.row_stack((temperature, humidity, light, co2, humidityr,time, bias))
X = np.transpose(X)

##initialising test data
test = []
with open('/Users/ashwinsankar/Desktop/DL/HW2/Code and Dataset/test_data.csv', 'r') as t_csvfile:
    t_data = csv.reader(t_csvfile, delimiter=',')
    for row in t_data:
        test.append(row)
test.pop(0)
t_order = min_max_norm( np.array([float(row[0]) for row in test]))
t_time = np.array([row[1] for row in test])
t_time = time_normalisaiton(t_time)
t_temperature = min_max_norm(np.array([float(row[2]) for row in test]))
t_humidity = min_max_norm(np.array([float(row[3]) for row in test]))
t_light = min_max_norm(np.array([float(row[4]) for row in test]))
t_co2 = min_max_norm(np.array([float(row[5]) for row in test]))
t_humidityr = min_max_norm(np.array([float(row[6]) for row in test]))
t_bias = np.array(np.ones(t_temperature.shape))
t_occupancy = np.array([int(row[7]) for row in test])
data_test = np.row_stack((t_temperature, t_humidity, t_light, t_co2, t_humidityr,t_time, t_bias))

def init_network(n_inputs=2, n_hidden=2, n_outputs=1):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def transfer(activation , type='sigmoid'):
    if(type=='sigmoid'):
        return (1.0 / (1.0 + exp(-activation)))
    elif(type=='tanh'):
        return np.tanh(activation)


def transfer_derivative(output, type='sigmoid'):
    if(type=='sigmoid'):
        return output * (1.0 - output)
    elif(type=='tanh'):
        return 1.0 - x ** 2


def activate(weights, inputs, bias):
    activation = 0.0
    for i in range(len(weights) - 1):
        activation += (weights[i] * inputs[i])
    return transfer(activation)


def feed_forward(network, row, bias):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs, bias)
            neuron['output'] = activation
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs[0]


def back_propagate(network, expected , error_from_expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(error_from_expected)
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def learning_rate_scheduler(epoch):
    return 0.1/(1+0.5*epoch)
'''    return 0.1/pow(0.5,floor((1+epoch)/10.0)) step decay = started to increase the lrnrate
      return 0.1/exp(-0.1*epoch)      exponential decay = very erratic '''


def weight_update(network, train, l_rate, cur_pos, batch_size,epoch):
    batch = train[cur_pos - batch_size: cur_pos + 1]
    lrn_rate = learning_rate_scheduler(epoch)
    lrn_rate_grp.append(lrn_rate)
    for row in batch:
        for i in range(len(network)):
            inputs = row
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += lrn_rate * neuron['delta']  * inputs[j]


def error_function(expected, predicted):  # cross entropy
    return (expected -predicted)**2


def Trainer(network, train, expected, bias, lrn_rate, n_epochs, batch_size):
    for epoch in range(n_epochs):
        sum_error = 0
        total_error = 0
        train_accuracy = 0
        i = 0
        for row in train:
            output = feed_forward(network, row, bias)
            if(occupancy[i]==round(output)):
                train_accuracy+=1
            sum_error += (occupancy[i]-output)
            total_error += error_function(occupancy[i], output)
            if (i % batch_size == 0 and i != 0):
                back_propagate(network, expected[i] , sum_error/batch_size)
                weight_update(network, train, lrn_rate, i, batch_size , epoch)
                sum_error = 0
            i += 1
        print('>epoch=%d, lrate=%.3f, error=%.7f' % (epoch, lrn_rate, total_error / X.shape[0]))
        acc = Tester(network)
        accuracy.append(acc)
        train_accuracy_grp.append(train_accuracy/X.shape[0])
        error_grp.append(sum_error / X.shape[0] )
        iter_grp.append(epoch)


def Tester(network, bias=1):
    i = 0
    testpr = 0
    datatest = np.transpose(data_test)
    for a in datatest:
        t_output = feed_forward(network, a, bias)
        # print(t_output,' ', t_occupancy[i])
        t_output = round(t_output , 0)
        # t_output = 1.0 if t_output > 1.0 else t_output
        # print(t_output,' ', t_occupancy[i])
        if t_output == t_occupancy[i]:
            testpr += 1
        i += 1
    # print("Classification accuracy in test set", testpr / datatest.shape[0])
    return testpr / datatest.shape[0]



baseline = baseline()
print(baseline)
n_input = len(X[0]) - 1
network = init_network(n_input, 5, 1)
Trainer(network, X, occupancy, 1, 0.01,25,64)
for x in range(0,25):
    baseline_grp.append(1-baseline)
print("\n Train Accuracy over epoches : ", train_accuracy_grp, '\n Train Average Accuracy : ', np.mean(train_accuracy_grp))
print('\n Test Accuracy over epoches: ',accuracy, '\n Test Average accuracy : ' ,np.mean(accuracy))


fig = plt.figure("Backprop: min-batch:100; time based decay; tanh")
plt.plot(iter_grp, error_grp,'r',label='Model')
plt.plot(iter_grp,baseline_grp,'b',label='Baseline')
plt.ylabel('Error')
plt.xlabel('Epochs')
plt.title('Error convergence graph')
plt.legend()
plt.show()

fig = plt.figure("Backprop: min-batch:100; time based decay; tanh")
plt.plot(iter_grp, error_grp,'r',label='Model')
plt.ylabel('Error')
plt.xlabel('Epochs')
plt.title('Error convergence graph')
plt.legend()
plt.show()

fig = plt.figure("Backprop: min-batch:100; time based decay; tanh")
plt.plot(iter_grp,train_accuracy_grp,'r',label='train')
plt.plot(iter_grp,accuracy,'b',label='test')
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title('Accuracy test over epochs')
plt.show()




# print(network)
