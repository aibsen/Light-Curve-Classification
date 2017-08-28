# coding: utf-8
# For use in ipython
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

feat_df = pd.read_csv("Features_many_sorted.csv", index_col=0, header=0)
feat_names = list(feat_df.columns.values)
feat_df = feat_df[[feat for feat in feat_names if ("SlottedA" not in feat) and ("StetsonK_AC" not in feat)]]
feat_df = feat_df[~feat_df.isin([np.nan, np.inf, -np.inf]).any(1)]
feat_names = list(feat_df.columns.values)
print(feat_names)
fn_cur = []
fn_cur.extend(feat_names[:3])
fn_cur.extend(feat_names[8:10])
sns.pairplot(feat_df, hue="Var_Type", vars=fn_cur, dropna=True)
plt.show()

# Subset classes 1 and 5
feat_df_sub = feat_df[feat_df["Var_Type"].isin([1, 5])]
fn_cur = []
fn_cur.extend(feat_names[:3])
fn_cur.extend(feat_names[8:10])
sns.pairplot(feat_df_sub, hue="Var_Type", vars=fn_cur, dropna=True)
plt.show()
