import pandas as pd
import numpy as np
import math
import os

def classify_nb(train_set, test_set):
    test_result = []

    # split two yes df and no df
    yes_df, no_df = split_data(train_set)

    # calculate the number of sample for both classes and the p of each class
    num_yes = len(yes_df)
    num_no = len(no_df)

    p_yes = num_yes / (num_yes + num_no)
    p_no = num_no / (num_yes + num_no)

    # calculate mean and std
    yes_mean = cal_mean(yes_df)
    yes_std = cal_std(yes_df)
    no_mean = cal_mean(no_df)
    no_std = cal_std(no_df)

    # loop through each row of test set
    for index, row in test_set.iterrows():
        prob_yes = p_yes
        prob_no = p_no
        # calculate probability of yes and no
        yes_prob = pdf(row.tolist()[:-1], yes_mean, yes_std)
        no_prob = pdf(row.tolist()[:-1], no_mean, no_std)

        for prob in yes_prob:
            if prob != 0:
                prob_yes *= prob

        for prob in no_prob:
            if prob != 0:
                prob_no *= prob

        # compare probability of yes and no
        if prob_yes >= prob_no:
            test_result.append('yes')
        else:
            test_result.append('no')

    return test_result


def read_csv(filename):
    data = pd.read_csv(filename, header=None)
    return data


def split_data(df):
    """
    split pandas dataframe into two parts:
    1. data with class yes
    2. data with class no
    """
    yes_df = df[df.iloc[:, -1] == 'yes']
    no_df = df[df.iloc[:, -1] == 'no']
    return yes_df, no_df


def cal_mean(df):
    """
    calculate mean of each column
    """
    mean_ls = []
    for col in df.iloc[:, :-1]:
        mean_ls.append(df[col].mean())

    return mean_ls


def cal_std(df):
    """
    calculate std of each column
    """
    std_ls = []
    for col in df.iloc[:, :-1]:
        std_ls.append(df[col].std())

    return std_ls


def pdf(x, mean, std):
    """
    calculate probability density function
    """
    res_ls = []
    for x, mean, std in zip(x, mean, std):
        res_ls.append(pdf_func(x, mean, std))
    return res_ls


def pdf_func(x, mean, std):
    """
    calculate probability density function
    """

    return (1 / (std * np.sqrt(2 * np.pi))) * math.e**(-(x - mean) ** 2 / (2 * std ** 2))

### below is the 10 folds cross validation code
def read_fold_csv(folds_path):
    data_folds = []

    for fold in os.listdir(folds_path):
        data = pd.read_csv(os.path.join(folds_path, fold), header=None)
        data_folds.append(data)

    return data_folds

def cv_nb(folds_path):
    """
    k-fold cross validation, k based on the num of file in the folds_path, here is 10
    """
    acc_ls = []
    t_prec_ls = []
    t_recall_ls = []
    t_f1_ls = []
    t_TPR_ls = []
    t_TNR_ls = []

    f_prec_ls = []
    f_recall_ls = []
    f_f1_ls = []
    f_TPR_ls = []
    f_TNR_ls = []

    folds = read_fold_csv(folds_path)
    for i, test_fold in enumerate(folds):
        train_folds = folds[:]
        train_folds.pop(i)
        train_folds = pd.concat(train_folds)
        test_res = classify_nb(train_folds, test_fold)
        accuracy, t_precision, t_recall, t_f1, t_TPR, t_TNR = true_metrics(test_res, test_fold)
        _, f_precision, f_recall, f_f1, f_TPR, f_TNR = false_metrics(test_res, test_fold)
        # print(f"Fold {i+1}:", f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        acc_ls.append(accuracy)
        t_prec_ls.append(t_precision)
        t_recall_ls.append(t_recall)
        t_f1_ls.append(t_f1)
        t_TPR_ls.append(t_TPR)
        t_TNR_ls.append(t_TNR)
        f_prec_ls.append(f_precision)
        f_recall_ls.append(f_recall)
        f_f1_ls.append(f_f1)
        f_TPR_ls.append(f_TPR)
        f_TNR_ls.append(f_TNR)

    print(f"Average:", f"Accuracy: {np.mean(acc_ls)}, ")
    print(f"True Precision: {np.mean(t_prec_ls)}, True Recall: {np.mean(t_recall_ls)}, True F1: {np.mean(t_f1_ls)}, True TPR: {np.mean(t_TPR_ls)}, True TNR: {np.mean(t_TNR_ls)}")
    print(f"False Precision: {np.mean(f_prec_ls)}, False Recall: {np.mean(f_recall_ls)}, False F1: {np.mean(f_f1_ls)}, False TPR: {np.mean(f_TPR_ls)}, False TNR: {np.mean(f_TNR_ls)}")
    print(f"Average Precision: {np.mean(t_prec_ls + f_prec_ls)}, Average Recall: {np.mean(t_recall_ls + f_recall_ls)}, Average F1: {np.mean(t_f1_ls + f_f1_ls)}") # , Average TPR: {np.mean(t_TPR_ls + f_TPR_ls)}, Average TNR: {np.mean(t_TNR_ls + f_TNR_ls)}")
    # print(f"Micro Average Precision: {np.mean(t_prec_ls + f_prec_ls)}, Micro Average Recall: {np.mean(t_recall_ls + f_recall_ls)}, Micro Average F1: {np.mean(t_f1_ls + f_f1_ls)}") # , Average TPR: {np.mean(t_TPR_ls + f_TPR_ls)}, Average TNR: {np.mean(t_TNR_ls + f_TNR_ls)}")
    # print(f"Macro Average Precision: {(np.mean(t_prec_ls) + np.mean(f_prec_ls))/2}, Macro Average Recall: {(np.mean(t_recall_ls) + np.mean(f_recall_ls))/2}, Macro Average F1: {(np.mean(t_f1_ls) + np.mean(f_f1_ls))/2}")  #, Macro Average TPR: {(np.mean(t_TPR_ls) + np.mean(f_TPR_ls))/2}, Macro Average TNR: {(np.mean(t_TNR_ls) + np.mean(f_TNR_ls))/2}")

def true_metrics(test_res, test_fold):
    """
    calculate metrics
    include accuracy, precision, recall, f1
    """
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    total_pos = 0
    total_neg = 0
    for i, row in test_fold.iterrows():
        if row.iloc[-1] == 'yes':
            total_pos += 1
        elif row.iloc[-1] == 'no':
            total_neg += 1
        if row.iloc[-1] == 'yes' and test_res[i] == 'yes':
            true_pos += 1
        elif row.iloc[-1] == 'no' and test_res[i] == 'no':
            true_neg += 1
        elif row.iloc[-1] == 'yes' and test_res[i] == 'no':
            false_neg += 1
        elif row.iloc[-1] == 'no' and test_res[i] == 'yes':
            false_pos += 1
    accuracy = (true_pos + true_neg) / len(test_fold)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * precision * recall / (precision + recall)

    TPR = true_pos / total_pos
    TNR = true_neg / total_neg

    return accuracy, precision, recall, f1, TPR, TNR

def false_metrics(test_res, test_fold):
    """
    calculate precisions for the false class
    """
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    total_pos = 0
    total_neg = 0
    for i, row in test_fold.iterrows():
        if row.iloc[-1] == 'no':
            total_pos += 1
        elif row.iloc[-1] == 'yes':
            total_neg += 1
        if row.iloc[-1] == 'no' and test_res[i] == 'no':
            true_pos += 1
        elif row.iloc[-1] == 'yes' and test_res[i] == 'yes':
            true_neg += 1
        elif row.iloc[-1] == 'no' and test_res[i] == 'yes':
            false_neg += 1
        elif row.iloc[-1] == 'yes' and test_res[i] == 'no':
            false_pos += 1
    accuracy = (true_pos + true_neg) / len(test_fold)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * precision * recall / (precision + recall)

    TPR = true_pos / total_pos
    TNR = true_neg / total_neg

    return accuracy, precision, recall, f1, TPR, TNR


if __name__ == '__main__':
    folds_path = 'folds'
    res = cv_nb(folds_path)
    # print(res)

