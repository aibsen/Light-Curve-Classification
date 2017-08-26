import numpy as np
import pandas as pd
import FATS as ft
from sklearn.svm import SVC
from sklearn import datasets

def get_features(trainning_set_length):
    # getting features (X)
    featureL=[
        'Amplitude',
        'Autocor_length',
        # 'AndersonDarling', #IndexError
        'Beyond1Std',
        # 'CAR_sigma', #runtime warning
        # 'CAR_mean', #depends on car_sigma,
        # "CAR_tau", #depends on car_sigma,
        # 'Con',
        # 'Color', #depends on mag2
        # 'Eta_e', #gives nan :/
        # 'FluxPercentileRatioMid20', #IndexError
        # 'FluxPercentileRatioMid35', #IndexError
        # 'FluxPercentileRatioMid50', #IndexError
        # 'FluxPercentileRatioMid65', #IndexError
        # 'FluxPercentileRatioMid80', #IndexError,
        # 'Freq1_harmonics_amplitude_0', #IndexError
        # 'LinearTrend',
        'MaxSlope', #also gives nan for some values :/
        'Mean',
        # 'Meanvariance',
        # 'MedianAbsDev',
        # 'MedianBRP',
        # 'PairSlopeTrend',
        # 'PercentAmplitude',
        # 'PercentDifferenceFluxPercentile', #IndexError
        # 'PeriodLS', #TypeError
        # 'Period_fit', #depends on PeriodLS
        # 'Psi_CS', #depends on PeriodLS
        # 'Psi_eta', #depends on PeriodLS,
        # 'Q31',
        # 'Q31_color', #needs mag2
        # "Rcs",
        # "Skew",
        # "SlottedA_length", #ValueError when printing results
        # "SmallKurtosis",
        "Std"
        # "StetsonJ", #depends on mag2, error2
        # 'StetsonK',
        # 'StetsonJ', #Index Error
        # 'StetsonK_AC', #Runtime warning
        # 'StetsonL', #depends on mag2, error2
        # 'VariabilityIndex' #not found
        ]
    # featureList = "all"
    fs = ft.FeatureSpace(featureList=featureL)
    # fs = ft.FeatureSpace(Data=["magnitude", "time", "error"])

    number_of_features = len(featureL)
    X = np.empty((0,number_of_features), float)
    Y = np.empty((0,0), float)

    filenameLC = "data/AllVar_cleaned.csv"
    filenameY = "data/classes.csv"  # Directory of data for processing
    data = pd.read_csv(filenameLC, sep=",", names=["id", "time", "mag", "error"])
    classes = pd.read_csv(filenameY, sep=" ", names=["id", "tag"])
    ids = data.id.unique()
    number_of_objects = len(ids)

    for id in ids:#[:trainning_set_length]: #I'm calculating only the first 15 xi, since it's just a test and I don't want to kill my pc
        print "calculating features for",id
        data_i = data[data.id==id]
        preprocessed_data = ft.Preprocess_LC(data_i['mag'].tolist(), data_i['time'].tolist(), data_i['error'].tolist())
        [mag, time, error] = preprocessed_data.Preprocess()
        lc = np.array([mag, time, error])
        fs = fs.calculateFeature(lc)
        feature = fs.result()
        X = np.append(X,np.array([feature]),axis=0)

        class_i = classes[classes.id==str(id)]
        Y = np.append(Y,class_i.tag)
    return [X,Y] 
    

def train_SVM(X, Y, trainning_set_length):
    model = SVC(verbose=True)
    model.fit(X,Y[:trainning_set_length])
    return model

if __name__ == "__main__":
    #load data
    number_of_features = 17
    trainning_set_length = 15
    [X,Y] = get_features(trainning_set_length)
    np.save("features", X)
    np.save("classes",Y)
    # print X.shape
    # print Y.shape
    # print Y
    
    # model = train_SVM(X,Y,trainning_set_length)
    # print Y[:trainning_set_length].shape
    # print X.shape
    # print X