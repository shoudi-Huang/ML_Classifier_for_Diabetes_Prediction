import pandas as pd
import numpy as np
from collections import Counter

def classify_nn(training_filename, testing_filename, k):
    traning_dataset = pd.read_csv(training_filename, header = None)
    testing_dataset = pd.read_csv(testing_filename, header = None)
    
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
        