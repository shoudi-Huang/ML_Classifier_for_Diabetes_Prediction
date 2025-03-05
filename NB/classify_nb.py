import pandas as pd
import numpy as np
import math

def classify_nb(training_filename, testing_filename):
    test_result = []

    # read train and test csv files
    train_set = read_csv(training_filename)
    test_set = read_csv(testing_filename)

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
        yes_prob = pdf(row.tolist(), yes_mean, yes_std)
        no_prob = pdf(row.tolist(), no_mean, no_std)

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


if __name__ == '__main__':
    train_filename = 'pima.csv'
    test_filename = 'pima.csv'
    res = classify_nb(train_filename, test_filename)
    print(res)

