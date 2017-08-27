import os
import numpy as np
import FATS as ft
import pandas as pd
import warnings

from sklearn.svm import SVC
from sklearn import datasets

warnings.filterwarnings("ignore")  # temporariy used to ignore the runtime warnings that pop up from FATS

def calculate_feature_data(lc_ids, feature_list):
    # check for when feature_list == "all"
    if feature_list and isinstance(feature_list, list):
        # use supplied list of features
        feature_space = ft.FeatureSpace(featureList=feature_list)
    else:
        # otherwise create features from only magnitude, time, and error
        # needed to fix indexing errors in FeatureFunctionLib.py and lomb.py before this started working
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
