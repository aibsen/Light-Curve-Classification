# For use in ipython

import os
import numpy as np
import pandas as pd
import FATS as ft
import warnings


warnings.filterwarnings("ignore")  # temporarily used to ignore the runtime warnings

# data_dir = "D:\\Files\\VC\\LSDSS\\2017_Program\\Project\\Data\\"

# Initialize light curve data
lightcurve_classes_filename = os.path.join(data_dir, "classes_trimmed.csv")
lightcurve_timeseries_filename = os.path.join(data_dir, "AllVar_cleaned_trimmed.csv")
lc_classes_trimmed = pd.read_csv(lightcurve_classes_filename, header=0, delimiter=" ")
lc_timeseries_trimmed = pd.read_csv(lightcurve_timeseries_filename, header=0, delimiter=",")

lctest = lc_timeseries_trimmed[lc_timeseries_trimmed["Numerical_ID"] == lc_timeseries_trimmed["Numerical_ID"][0]]

data_list = ["magnitude", "time", "error"]

feature_space = ft.FeatureSpace(Data=data_list)
feature_list = list(feature_space.featureList)

for feat in feature_list:
    cfeatspace = ft.FeatureSpace(featureList=[feat])
    print("Feature: %s" % feat)
    %timeit cfeatspace.calculateFeature(lctest[["Magnitude", "Time", "Error"]].as_matrix().transpose()).result("array")

