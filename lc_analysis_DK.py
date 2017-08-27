import os
import numpy as np
import FATS as ft
import pandas as pd
import warnings

from sklearn.svm import SVC
from sklearn import datasets

warnings.filterwarnings("ignore")  # temporarily used to ignore the runtime warnings that pop up from FATS


def preprocess_lc_data_part1():
    data_dir = "./../Data"  # Directory of data for processing
    data_dir = "D:\\Files\\VC\\LSDSS\\2017_Program\\Project\\Data"
    # Initialize light curve data
    lightcurve_classes_filename = os.path.join(data_dir, "classes.csv")
    lightcurve_timeseries_filename = os.path.join(data_dir, "AllVar_cleaned.csv")

    lc_classes = pd.read_csv(lightcurve_classes_filename, header=0, delimiter=" ")
    lc_timeseries = pd.read_csv(lightcurve_timeseries_filename, names=["Numerical_ID", "Time", "Magnitude", "Error"], delimiter=",")
    print(lc_classes)
    print(lc_timeseries)

    # Get unique Numerical_IDs from time series data to trim extra IDs from the classes data
    lc_uids = np.unique(lc_timeseries["Numerical_ID"])
    lc_classes_trimmed = lc_classes[lc_classes["Numerical_ID"].isin(lc_uids)]

    # Keep only classes that have greater than 500 samples
    lc_ct_types_unique, lc_ct_types_counts = np.unique(lc_classes_trimmed["Var_Type"], return_counts=True)
    lc_ct_utypes_drop = lc_ct_types_unique[lc_ct_types_counts < 500]
    lc_classes_trimmed = lc_classes_trimmed[~lc_classes_trimmed["Var_Type"].isin(lc_ct_utypes_drop)]
    print(lc_classes_trimmed)
    # Trim IDs from the time series data using lc_classes_trimmed
    lc_timeseries_trimmed = lc_timeseries[lc_timeseries["Numerical_ID"].isin(lc_classes_trimmed["Numerical_ID"])]

    # Save csv files
    lc_classes_newfn = os.path.join(data_dir, "classes_trimmed.csv")
    lc_timeseries_newfn = os.path.join(data_dir, "AllVar_cleaned_trimmed.csv")
    lc_classes_trimmed.to_csv(lc_classes_newfn, sep=" ", index=False, mode='wb', encoding='utf8')
    lc_timeseries_trimmed.to_csv(lc_timeseries_newfn, index=False, mode='wb', encoding='utf8')

def calculate_feature_data(lc_ids, feature_list):
    # check for when feature_list == "all"
    if feature_list and isinstance(feature_list, list):
        # use supplied list of features
        feature_space = ft.FeatureSpace(featureList=feature_list)
    else:
        # otherwise create features from only magnitude, time, and error
        # Needed to apply fixes to FATS before this starts working
        feature_space = ft.FeatureSpace(Data=["magnitude", "time", "error"])
        feature_list = feature_space.featureList
    
    number_of_features = len(feature_space.featureList)
    number_of_lightcurves = len(lc_ids)
    # print("feature_space.featureList: " + str(feature_space.featureList))
    print("number of features: " + str(number_of_features))
    print("number of light curves: " + str(number_of_lightcurves))
    
    # initialize numpy array for feature data of all given ids
    feature_data_nparray = np.zeros((number_of_lightcurves, number_of_features), dtype="float")
    
    # load light curve data, process with FATS, place into numpy array 
    for i, name in enumerate(lc_ids):
        data_filename = os.path.join(data_dir, name+'.dat')
        data = np.loadtxt(data_filename)
        data = data.transpose()
        preprocessed_data = ft.Preprocess_LC(data[0], data[1], data[2])
        [mag, time, error] = preprocessed_data.Preprocess()
        lc_data = np.array([mag, time, error])
        print("calculating features for ID: " + str(name))
        feature_space = feature_space.calculateFeature(lc_data)
        
        # using the dict option and looping over feature_list since the results are not sorted if using the Data=... option
        fs_result = feature_space.result("array")
        feature_list = feature_space.result("features")
        # for i_feat, feat in enumerate(feature_list):
        #     feature_data_nparray[i,i_feat] = fs_result[feat]
        # feature_data_nparray[i,:] =  np.array(feature_space.result())
    
    return feature_data_nparray, feature_list


def fix_feature_data(feature_data, feature_list):
    """This function removes features (columns) with non numerical values (nan or inf) from the feature data array."""
    feat_nanbool = np.isnan(feature_data).any(axis=0)
    feat_infbool = np.isinf(feature_data).any(axis=0)
    feat_fixedbool = np.logical_and(~feat_nanbool, ~feat_infbool)
    print("feat_fixedbool: " + str(feat_fixedbool))
    feature_data = np.copy(feature_data[:, feat_fixedbool])
    feature_list = [feat for i, feat in enumerate(feature_list) if feat_fixedbool[i]]
    return feature_data, feature_list


if __name__ == "__main__":
    print("Start Light Curve Analysis")
    quit()
    
    data_dir = "./../Data/SSS_Per_Var_Cat"  # Directory of data for processing

    # Initialize light curve metadata
    lightcurve_metadata_filename = os.path.join(data_dir, "SSS_Per_Tab.dat")
    lc_metadata = np.loadtxt(lightcurve_metadata_filename, skiprows=3, usecols=(0,7))
    lc_ids = np.array(lc_metadata[:,0], dtype="int64").astype("str")
    lc_types = np.array(lc_metadata[:,1], dtype="int64")
    
    # feature_list = [
    #     # 'Amplitude', #IndexError
    #     'Autocor_length',
    #     # 'AndersonDarling', #IndexError
    #     'Beyond1Std',
    #     # 'CAR_sigma', #runtime warning
    #     # 'CAR_mean', #depends on car_sigma,
    #     # "CAR_tau", #depends on car_sigma,
    #     'Con',
    #     # 'Color', #depends on mag2
    #     # 'Eta_e', #gives nan :/
    #     # 'FluxPercentileRatioMid20', #IndexError
    #     # 'FluxPercentileRatioMid35', #IndexError
    #     # 'FluxPercentileRatioMid50', #IndexError
    #     # 'FluxPercentileRatioMid65', #IndexError
    #     # 'FluxPercentileRatioMid80', #IndexError,
    #     # 'Freq1_harmonics_amplitude_0', #IndexError
    #     'LinearTrend',
    #     # 'MaxSlope', #also gives nan for some values :/
    #     'Mean',
    #     'Meanvariance',
    #     'MedianAbsDev',
    #     'MedianBRP',
    #     'PairSlopeTrend',
    #     'PercentAmplitude',
    #     # 'PercentDifferenceFluxPercentile', #IndexError
    #     # 'PeriodLS', #TypeError
    #     # 'Period_fit', #depends on PeriodLS
    #     # 'Psi_CS', #depends on PeriodLS
    #     # 'Psi_eta', #depends on PeriodLS,
    #     'Q31',
    #     # 'Q31_color', #needs mag2
    #     "Rcs",
    #     "Skew",
    #     # "SlottedA_length", #ValueError when printing results
    #     "SmallKurtosis",
    #     "Std",
    #     # "StetsonJ", #depends on mag2, error2
    #     'StetsonK'
    #     # 'StetsonJ', #Index Error
    #     # 'StetsonK_AC', #Runtime warning
    #     # 'StetsonL', #depends on mag2, error2
    #     # 'VariabilityIndex' #not found
    # ]
    feature_list = "all"
    training_set_length = 15

    lc_ids = lc_ids[:training_set_length]
    lc_types = lc_types[:training_set_length]
    
    feature_data_nparray, feature_list = calculate_feature_data(lc_ids, feature_list)
    feature_data_nparray, feature_list = fix_feature_data(feature_data_nparray, feature_list)
    
    print("lc_types.shape: " + str(lc_types.shape))
    print("feature_data_nparray:")
    print(feature_data_nparray)
    print("feature_list:")
    print(feature_list)
    model = SVC(verbose=True)
    model.fit(feature_data_nparray, lc_types)
    
    print("End")
