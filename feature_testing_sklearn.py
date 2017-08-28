import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Initial classification scores for each type of feature

data_dir = "D:\\Files\\VC\\LSDSS\\2017_Program\\Project\\Data\\"

feat_filename = os.path.join(data_dir, "Features_many_sorted.csv")

feat_df = pd.read_csv(feat_filename, index_col=0, header=0)
feat_names = list(feat_df.columns.values)
feat_df = feat_df[[feat for feat in feat_names if ("SlottedA" not in feat) and ("StetsonK_AC" not in feat)]]
feat_df = feat_df[~feat_df.isin([np.inf, -np.inf]).any(1)]
feat_df = feat_df[~feat_df.isnull().any(1)]
feat_names = list(feat_df.columns.values)

feat_df_notype = feat_df[list(feat_df.columns.values)[:-1]]
feat_names_notype = list(feat_df_notype.columns.values)
feat_df_onlytype = feat_df[["Var_Type"]]

score_array = np.zeros([len(feat_names_notype)], dtype="float")

# X_train, X_test, Y_train, Y_test = train_test_split(feat_df_notype, feat_df_onlytype)

X = feat_df_notype.as_matrix()
y = feat_df_onlytype.as_matrix().ravel()

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0,
                              class_weight="balanced_subsample")

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
