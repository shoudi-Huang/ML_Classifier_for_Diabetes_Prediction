import pandas as pd
import numpy as np
from collections import Counter
import re
from statistics import mean

def classify_nn(traning_dataset, testing_dataset, k):
    num_columns = len(traning_dataset.columns)
    class_columns = []
    for i in range(0, num_columns):
        if traning_dataset[i].dtypes == object:
            class_columns.append(i)
    
    X_train = traning_dataset
    for i in class_columns:
        X_train = X_train.drop(i, axis=1)
    X_train = X_train.values
    
    y_train = [traning_dataset[i] for i in class_columns]
    y_train = np.array(y_train)[0]
    
    X_test = testing_dataset
    X_test = X_test.values
    
    predictions = []
    for data_point in X_test:
        distances = np.linalg.norm(X_train - data_point, axis=1)
        nearest_neighbor_ids = distances.argsort()[:k]
        count = Counter(y_train[nearest_neighbor_ids])
        predict = count.most_common(1)[0][0]
        predictions.append(predict)
    
    return predictions

def knn_cv(data_filename, k):
    data_file = open(data_filename, "r")
    last_line = data_file.readlines()[-1]
    data_file.close()

    data_file = open(data_filename, "r")
    dataset_10folds = []
    p = re.compile('fold*')
    dataset_fold = np.array([])
    arr = ""
    
    for line in data_file:
        if p.match(line) is not None:
            continue
        elif line == "\n":
            dataset_10folds.append(dataset_fold)
            dataset_fold = np.array([])
            continue
        
        arr += line.replace("\n", "") ## Replace each newline character in each line so that arr is just one long continuous string
        arr = arr.split(",")
        arr = np.array( arr )
        if dataset_fold.size == 0:
            dataset_fold = arr
        else:
            dataset_fold = np.vstack((dataset_fold, arr))
        arr = ""

        if line == last_line:
            dataset_10folds.append(dataset_fold)
    
    data_file.close()
    
    cv_accuracy = []
    total_TP_num = 0
    total_TN_num = 0
    total_P_num = 0
    total_N_num = 0
    total_predict_P_num = 0
    total_predict_N_num = 0
    for i in range(0, len(dataset_10folds)):
        test_set = dataset_10folds[i]
        test_set_df = pd.DataFrame(test_set)
        for x in range(0, len(test_set_df.columns)-1):
            test_set_df[x] = test_set_df[x].astype(float)
        
        y_test = test_set_df[5].values
        test_set_df = test_set_df.drop(5, axis=1)

        traning_set_ls = [x for index,x in enumerate(dataset_10folds) if index!=i]
        traning_set = np.array([])
        for fold in traning_set_ls:
            if traning_set.size == 0:
                traning_set = fold
            else:
                traning_set = np.vstack((traning_set, fold))
        traning_set_df = pd.DataFrame(traning_set)

        for x in range(0, len(traning_set_df.columns)-1):
            traning_set_df[x] = traning_set_df[x].astype(float)
        
        prediction = np.array(classify_nn(traning_set_df, test_set_df, k))
        positive_index = np.where(y_test=="yes")[0]
        positive_index = positive_index.tolist()
        negative_index = np.where(y_test=="no")[0]
        negative_index = negative_index.tolist()
        
        positive_index_in_prediction = np.where(prediction=="yes")[0].tolist()
        negative_index_in_prediction = np.where(prediction=="no")[0].tolist()
        
        TP_num = Counter(y_test[positive_index]==prediction[positive_index])[True]
        TN_num = Counter(y_test[negative_index]==prediction[negative_index])[True]
        
        total_TP_num += TP_num
        total_P_num += len(positive_index)
        total_TN_num += TN_num
        total_N_num += len(negative_index)
        total_predict_P_num += len(positive_index_in_prediction)
        total_predict_N_num += len(negative_index_in_prediction)
        cv_accuracy.append(Counter(y_test==prediction)[True]/len(y_test))
    
    knn_accuracy = mean(cv_accuracy)
    TPR = total_TP_num/total_P_num
    TNR = total_TN_num/total_N_num
    yes_precision = total_TP_num/total_predict_P_num
    no_precision = total_TN_num/total_predict_N_num
    return {"knn_accuracy":knn_accuracy, "TPR":TPR, "TNR":TNR, "yes_precision":yes_precision, "no_precision":no_precision}
            

oneNN_performance = knn_cv("./../../../../pima-CFS-folds.csv", 1)
fiveNN_performance = knn_cv("./../../../../pima-CFS-folds.csv", 5)
print("10-fold cross validation accuracy of 1NN with CFS: ", oneNN_performance["knn_accuracy"]*100, "%")
print("10-fold cross validation TPR of 1NN with CFS: ", oneNN_performance["TPR"]*100, "%")
print("10-fold cross validation TNR of 1NN with CFS: ", oneNN_performance["TNR"]*100, "%")
print("1NN Yes class Precision: ", oneNN_performance["yes_precision"]*100, "%")
print("1NN No class Precision: ", oneNN_performance["no_precision"]*100, "%")
print()
print("10-fold cross validation accuracy of 5NN with CFS: ", fiveNN_performance["knn_accuracy"]*100, "%")
print("10-fold cross validation TPR of 5NN with CFS: ", fiveNN_performance["TPR"]*100, "%")
print("10-fold cross validation TNR of 5NN with CFS: ", fiveNN_performance["TNR"]*100, "%")
print("5NN Yes class Precision: ", fiveNN_performance["yes_precision"]*100, "%")
print("5NN No class Precision: ", fiveNN_performance["no_precision"]*100, "%")