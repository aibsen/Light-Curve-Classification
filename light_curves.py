import numpy as np
import FATS as ft

#get Y
names_tags = np.loadtxt('SSS_Per_Var_Cat/SSS_Per_Tab.dat', skiprows=3, usecols= (0,7))
names = np.array([str(int(name)) for name in names_tags[:,0]])
Y = np.array([int(yi) for yi in names_tags[:,1]])

# getting features (X)

data = np.loadtxt('SSS_Per_Var_Cat/'+names[0]+'.dat')
data = data.transpose()
preprocessed_data = ft.Preprocess_LC(data[0], data[1], data[2])
[mag, time, error] = preprocessed_data.Preprocess()
lc = np.array([mag, time, error])

featureL=[
    # 'Amplitude', #IndexError
    'Autocor_length',
    # 'AndersonDarling', #IndexError
    'Beyond1Std',
    # 'CAR_sigma', #runtime warning
    # 'CAR_mean', #depends on car_sigma,
    # "CAR_tau", #depends on car_sigma,
    'Con',
    # 'Color', #depends on mag2
    'Eta_e',
    # 'FluxPercentileRatioMid20', #IndexError
    # 'FluxPercentileRatioMid35', #IndexError
    # 'FluxPercentileRatioMid50', #IndexError
    # 'FluxPercentileRatioMid65', #IndexError
    # 'FluxPercentileRatioMid80', #IndexError,
    # 'Freq1_harmonics_amplitude_0', #IndexError
    'LinearTrend',
    'MaxSlope',
    'Mean',
    'Meanvariance',
    'MedianAbsDev',
    'MedianBRP',
    'PairSlopeTrend',
    'PercentAmplitude',
    # 'PercentDifferenceFluxPercentile', #IndexError
    # 'PeriodLS', #TypeError
    # 'Period_fit', #depends on PeriodLS
    # 'Psi_CS', #depends on PeriodLS
    # 'Psi_eta', #depends on PeriodLS,
    'Q31',
    # 'Q31_color', #needs mag2
    "Rcs",
    "Skew",
    # "SlottedA_length", #ValueError when printing results
    "SmallKurtosis",
    "Std",
    # "StetsonJ", #depends on mag2, error2
    'StetsonK'
    # 'StetsonJ', #Index Error
    # 'StetsonK_AC', #Runtime warning
    # 'StetsonL', #depends on mag2, error2
    # 'VariabilityIndex' #not found
    ]
a = ft.FeatureSpace(featureList=featureL)
a = a.calculateFeature(lc)
feature = a.result()
print feature
print feature.shape