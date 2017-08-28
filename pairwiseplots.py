# coding: utf-8
# For use in ipython
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

feat_df = pd.read_csv("Features_many_sorted.csv", index_col=0, header=0)
feat_names = list(feat_df.columns.values)
feat_df = feat_df[[feat for feat in feat_names if ("SlottedA" not in feat) and ("StetsonK" not in feat)]]
feat_df = feat_df[~feat_df.isin([np.nan, np.inf, -np.inf]).any(1)]
feat_names = list(feat_df.columns.values)
print(feat_names)
sns.pairplot(feat_df, hue="Var_Type", vars=feat_names[5:8], dropna=True)
plt.show()
