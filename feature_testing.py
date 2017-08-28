import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split


# Initial classification scores for each type of feature

data_dir = "D:\\Files\\VC\\LSDSS\\2017_Program\\Project\\Data\\"

feat_filename = os.path.join(data_dir, "Features_many_sorted.csv")

feat_df = pd.read_csv(feat_filename, index_col=0, header=0)
feat_names = list(feat_df.columns.values)
feat_df = feat_df[[feat for feat in feat_names if ("SlottedA" not in feat) and ("StetsonK_AC" not in feat)]]
feat_df = feat_df[~feat_df.isin([np.inf, -np.inf]).any(1)]
feat_df = feat_df[~feat_df.isnull().any(1)]
feat_names = list(feat_df.columns.values)

feat_df_sub = feat_df[feat_df["Var_Type"].isin([1, 5])]
feat_df_sub_notype = feat_df_sub[list(feat_df_sub.columns.values)[:-1]]
feat_names_notype = list(feat_df_sub_notype.columns.values)
feat_df_sub_onlytype = feat_df_sub[["Var_Type"]]

score_array = np.zeros([len(feat_names)], dtype="float")

X_train, X_test, Y_train, Y_test = train_test_split(feat_df_sub_notype, feat_df_sub_onlytype)

for i, feat in enumerate(feat_names_notype):
    svm_model = svm.SVC(class_weight='balanced')
    svm_model.fit(X_train[[feat]].as_matrix(), Y_train[["Var_Type"]].as_matrix().ravel())
    score_array[i] = svm_model.score(X_test[[feat]].as_matrix(), Y_test[["Var_Type"]].as_matrix().ravel())
    print(feat + " score: " + str(score_array[i]))
