"""
LightAD (ICSE'24) 中的经典分类器封装：KNN、决策树、单层隐层 MLP（原仓库称 SLFN）。
来源：../LightAD/models/classifiers.py，仅做路径与格式整理，算法不变。
"""
from __future__ import annotations

import time

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def decision_tree(train_data, testing_data, train_labels, **params_dict):
    start_time = time.time()
    clf = tree.DecisionTreeClassifier(**params_dict)
    clf = clf.fit(train_data, train_labels)
    end_time = time.time()
    train_time = end_time - start_time

    start_time = time.time()
    prediction = list(clf.predict(testing_data))
    end_time = time.time()
    infer_time = end_time - start_time
    return prediction, train_time, infer_time


def KNN(train_data, testing_data, train_labels, **params_dict):
    def drop_duplicate(data, labels):
        data_eli = []
        label_eli = []
        for idx, x in enumerate(data):
            if x not in data_eli:
                data_eli.append(x)
                label_eli.append(labels[idx])
        return data_eli, label_eli

    start_time = time.time()
    train_data, train_labels_new = drop_duplicate(train_data, train_labels)
    clf = KNeighborsClassifier(**params_dict)
    clf.fit(train_data, train_labels_new)
    end_time = time.time()
    train_time = end_time - start_time

    start_time = time.time()
    prediction = []
    pre_dict = {}
    for x in testing_data:
        xs = str(x)
        if xs not in pre_dict:
            temp_prediction = list(clf.predict([x]))[0]
            pre_dict[xs] = temp_prediction
            prediction.append(temp_prediction)
        else:
            prediction.append(pre_dict[xs])
    end_time = time.time()
    infer_time = end_time - start_time
    return prediction, train_time, infer_time


def MLP(train_data, testing_data, train_labels, **params_dict):
    start_time = time.time()
    clf = MLPClassifier(**params_dict)
    clf.fit(train_data, train_labels)
    end_time = time.time()
    train_time = end_time - start_time

    start_time = time.time()
    prediction = clf.predict(testing_data)
    end_time = time.time()
    infer_time = end_time - start_time
    return prediction, train_time, infer_time
