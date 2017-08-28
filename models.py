import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib


def load_data(filename, feature_list):
    #reading data from csv
    data = pd.read_csv(filename, sep=",")
    feature_list.append("Var_Type")
    data = data[feature_list]
    #taking out class 13 and class 6, because objects are too few
    data = data[data.Var_Type != 13]
    data = data[data.Var_Type != 6]
    fllen=len(feature_list)-1
    feature_list = feature_list[0:fllen]
    X = data[feature_list]
    Y = data["Var_Type"] 
    return X,Y,data
    

def load_data_two_classes(filename, c1, c2, feature_list):
    data = pd.read_csv(filename, sep=",")
    feature_list.append("Var_Type")
    data = data[feature_list]
    data_1 = data[data.Var_Type == c1]
    data_2 = data[data.Var_Type == c2]
    l1 = len(data_1.index)
    l2 = len(data_2.index)
    if(l1 > l2):
        data_1 = data_1.head(l2)
    elif(l1 < l2):
        data_2 = data_2.head(l1)
    sliced_data = data_1.append(data_2)
    # print sliced_data
    fllen=len(feature_list)-1
    feature_list = feature_list[0:fllen]
    X = sliced_data[feature_list]
    Y = sliced_data["Var_Type"]
    return X,Y,sliced_data

def split_datasets(X,Y):
    #converting data to numpy arrays, so we can split dataset
    Xnp = X.as_matrix()
    #normalize X
    # Xnp = preprocessing.normalize(Xnp,norm="l2")
    Ynp = Y.as_matrix()
    # splitting dataset into trainning and test data
    print"splitting data into trainning and test data sets"
    X_train, X_test, y_train, y_test = train_test_split(Xnp, Ynp, test_size=0.3)
    size_of_trainning = X_train.shape
    size_of_test = X_test.shape
    print "size of trainning data set:",size_of_trainning[0]
    print "size of test data set:",size_of_test[0]
    return X_train, X_test, y_train, y_test

def check_balance(y_train, y_test):
    #checking that sets are balanced
    Ytrain_pd = pd.DataFrame({"id":y_train})
    Ytest_pd = pd.DataFrame({"id":y_test})
    for tag in class_list:
        print "class",tag,"accounts for:"
        part_of_trainning = float(len(Ytrain_pd[Ytrain_pd.id==tag].index))/float(len(Ytrain_pd.index))
        part_of_test = float(len(Ytest_pd[Ytest_pd.id==tag].index))/float(len(Ytest_pd.index))
        print("%.3f of trainning data"% part_of_trainning)
        print("%.3f of test data"% part_of_test)


def train_SVM(Xtrain, Ytrain, Xtest, Ytest, filename):
    model = SVC()
    print("training SVM")
    model.fit(Xtrain, Ytrain)
    print("testing SVM")
    predicted = model.predict(Xtest)
    filepkl = filename+".pkl"
    filetxt = filename+".txt"
    joblib.dump(model,filepkl)
    #check how prediction went
    # print(metrics.confusion_matrix(expected, predicted))
    get_model_score(model, Xtrain, Ytrain, Xtest, Ytest, predicted, filetxt)

def train_DecisionTree(Xtrain, Ytrain, Xtest, Ytest, filename):
    model = DecisionTreeClassifier()
    print("training Desicion Tree Classifier")
    model.fit(Xtrain, Ytrain)
    print("testing Desicion Tree Classifier")
    predicted = model.predict(Xtest)
    #check how prediction went
    # print(metrics.classification_report(Ytest, predicted))
    # print(metrics.confusion_matrix(expected, predicted))
    # get_model_score(model, Xtrain, Ytrain, Xtest, Ytest)
    filepkl = filename+".pkl"
    filetxt = filename+".txt"
    joblib.dump(model,filepkl)
    get_model_score(model, Xtrain, Ytrain, Xtest, Ytest, predicted, filetxt)


def train_RandomForest(Xtrain, Ytrain, Xtest, Ytest, filename):
    model = RandomForestClassifier()
    print("training Random Forest Classifier")
    model.fit(Xtrain, Ytrain)
    print("testing Random Forest Classifier")
    predicted = model.predict(Xtest)
    #check how prediction went
    # print(metrics.confusion_matrix(expected, predicted))
    filepkl = filename+".pkl"
    filetxt = filename+".txt"
    joblib.dump(model,filepkl)
    get_model_score(model, Xtrain, Ytrain, Xtest, Ytest, predicted, filetxt)

def get_model_score(model, Xtrain, Ytrain, Xtest, Ytest, predicted, filename):
    print"cross validating"
    cv_scores = cross_val_score(model, Xtrain, Ytrain, cv=5)
    with open(filename,"w") as file:
        report = metrics.classification_report(Ytest, predicted)
        mean_score = np.mean(cv_scores)
        std = np.std(cv_scores)
        score = model.score(Xtest, Ytest)
        print "score",model.score(Xtest, Ytest)
        file.write("report: ")
        file.write(report)
        file.write("mean_score: ")
        file.write(str(mean_score)+"\n")
        file.write("std: ")
        file.write(str(std))

if __name__ == "__main__":
    print("LOADING DATA")
    filename = "data/Features_many_sorted.csv" 
    feature_list =["Autocor_length", "AndersonDarling"]
    # X, Y, data = load_data_two_classes(filename, 1, 2, feature_list)
    X, Y, data = load_data(filename, feature_list)
    Xtrain, Xtest, Ytrain, Ytest = split_datasets(X,Y)
    # print("SVM CLASSIFIER")
    results_filename = "results/DT_1_2_5_4_8" 
    # train_SVM(Xtrain, Ytrain, Xtest, Ytest, results_filename)
    # print("DECISION TREE CLASSIFIER")
    train_DecisionTree(Xtrain, Ytrain, Xtest, Ytest, results_filename)
    # print("RANDOM FOREST CLASSIFIER")
    # train_RandomForest(Xtrain, Ytrain, Xtest, Ytest, results_filename)
    print("done")
