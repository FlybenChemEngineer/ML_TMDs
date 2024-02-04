import pandas as pd


def features_targets_read(df):

 
    features = df.iloc[:, 1:-2]  
    target1 = df.iloc[:, -2]  
    target2 = df.iloc[:, -1]  


    feature_labels = df.columns[1:-2]

    return features, target1, target2, feature_labels


