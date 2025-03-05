import pandas as pd

def divide_10_folds(input_data_filename):
    input_data = pd.read_csv(input_data_filename, header=None)

    class_yes_data = input_data.loc[input_data[8] == "yes"]
    class_no_data = input_data.loc[input_data[8] == "no"]
    
    print(len(class_yes_data))
    print(len(class_no_data))

    class_yes_data.to_csv("yes.csv", sep=',', index=False, header = None)
    class_no_data.to_csv("no.csv", sep=',', index=False, header = None)

divide_10_folds("pima.csv")

    

    