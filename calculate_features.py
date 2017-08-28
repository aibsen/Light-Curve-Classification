import numpy as np
import pandas as pd
import FATS as ft
import csv

def get_features(trainning_set_length):
    # getting features (X)
    featureL=[
        # 'Amplitude',
        'Autocor_length',
        'AndersonDarling', #IndexError
        # 'Beyond1Std',
        # 'CAR_sigma', #runtime warning
        # 'CAR_mean', #depends on car_sigma,
        # "CAR_tau", #depends on car_sigma,
        'Con',
        # 'Eta_e', #gives nan :/
        'FluxPercentileRatioMid20', #IndexError
        # 'FluxPercentileRatioMid35', #IndexError
        # 'FluxPercentileRatioMid50', #IndexError
        # 'FluxPercentileRatioMid65', #IndexError
        # 'FluxPercentileRatioMid80', #IndexError,
        # 'Freq1_harmonics_amplitude_0', #IndexError
        'LinearTrend',
        # 'MaxSlope', #also gives nan for some values :/
        'Mean',
        'Meanvariance',
        'MedianAbsDev',
        'MedianBRP',
        # 'PairSlopeTrend',
        'PercentAmplitude',
        'PercentDifferenceFluxPercentile', #IndexError
        # 'PeriodLS', #TypeError
        # 'Period_fit', #depends on PeriodLS
        # 'Psi_CS', #depends on PeriodLS
        # 'Psi_eta', #depends on PeriodLS,
        # 'Q31',
        # "Rcs",
        # "Skew",
        # "SlottedA_length", #ValueError when printing results
        "SmallKurtosis",
        "Std",
        'StetsonK',
        # 'StetsonJ' #Index Error
        ]
    # featureList = "all"
    fs = ft.FeatureSpace(featureList=featureL)

    number_of_features = len(featureL)
    X = [["Numerical_ID"]]
    for f in featureL:
        X[0].append(f)
    X[0].append("Var_Type")
    print X
    filenameLC = "data/AllVar_cleaned_trimmed.csv"
    filenameY = "data/classes_trimmed.csv"  # Directory of data for processing
    data = pd.read_csv(filenameLC, sep=",", names=["id", "time", "mag", "error"], skiprows=1)
    classes = pd.read_csv(filenameY, sep=" ", names=["id", "tag"])
    ids = data.id.unique()
    number_of_objects = len(ids)

    for id in ids:
        print "calculating features for",id
        data_i = data[data.id==id]
        preprocessed_data = ft.Preprocess_LC(data_i['mag'].tolist(), data_i['time'].tolist(), data_i['error'].tolist())
        [mag, time, error] = preprocessed_data.Preprocess()
        lc = np.array([mag, time, error])
        fs = fs.calculateFeature(lc)
        feature = fs.result()
        feature = feature.tolist()
        class_i = classes[classes.id==str(id)]
        feature.insert(0,int(id))
        feature.append(int(class_i.tag))
        X.append(feature)
    return X
    

if __name__ == "__main__":
    #load data
    number_of_features = 17
    trainning_set_length = 15
    X = get_features(trainning_set_length)
    with open("data/Features2.csv",'wb') as resultFile:
        wr = csv.writer(resultFile,delimiter=",", dialect='excel')
        wr.writerows(X)