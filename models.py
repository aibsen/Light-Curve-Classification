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


def load_data(filename):
    #reading data from csv
    data = pd.read_csv(filename, sep=",", names=["id", "amplitude", "beyond1std", "pairslopetrend","y"], skiprows=1)
    #taking out class 13 and class 6, because objects are too few
    data = data[data.y != 13]
    data = data[data.y != 6]
    X = data[["id","amplitude", "beyond1std", "pairslopetrend"]]
    Y = data["y"] 
    #how many classes?
    class_list = Y.unique()
    number_of_classes = len(class_list)
    print "there are", number_of_classes,"classes :",class_list
    return X,Y,data
    

def load_data_two_classes(filename):
    data = pd.read_csv(filename, sep=",", names=["id", "amplitude", "beyond1std", "pairslopetrend","y"], skiprows=1)
    # which classes have the most objects?
    # class_list = data["y"].unique()
    # class_len = []
    # for c in class_list:
    #     sample = data[data.y == c]
    #     class_len.append(len(sample.index))
    # print class_list
    # print class_len
    data_5 = data[data.y == 5]
    l = len(data_5.index)
    data_1 = data[data.y == 1]
    data_1 = data_1.head(l)
    dataframes= [data_1,data_5]
    sliced_data = data_1.append(data_5)
    # print sliced_data
    X = sliced_data[["id","amplitude", "beyond1std", "pairslopetrend"]]
    Y = sliced_data["y"]
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


def train_SVM(Xtrain, Ytrain, Xtest, Ytest):
    model = SVC()
    print("training SVM")
    model.fit(Xtrain, Ytrain)
    print("testing SVM")
    predicted = model.predict(Xtest)
    #check how prediction went
    # print(metrics.confusion_matrix(expected, predicted))
    # get_model_score(model, Xtrain, Ytrain, Xtest, Ytest)

def train_DecisionTree(Xtrain, Ytrain, Xtest, Ytest):
    model = DecisionTreeClassifier()
    print("training Desicion Tree Classifier")
    model.fit(Xtrain, Ytrain)
    print("testing Desicion Tree Classifier")
    predicted = model.predict(Xtest)
    #check how prediction went
    # print(metrics.classification_report(Ytest, predicted))
    # print(metrics.confusion_matrix(expected, predicted))
    # get_model_score(model, Xtrain, Ytrain, Xtest, Ytest)


def train_RandomForest(Xtrain, Ytrain, Xtest, Ytest):
    model = RandomForestClassifier()
    print("training Random Forest Classifier")
    model.fit(Xtrain, Ytrain)
    print("testing Random Forest Classifier")
    predicted = model.predict(Xtest)
    #check how prediction went
    # print(metrics.confusion_matrix(expected, predicted))
    joblib.dump(model,"results/RF_1_5.pkl")
    get_model_score(model, Xtrain, Ytrain, Xtest, Ytest, predicted)

def get_model_score(model, Xtrain, Ytrain, Xtest, Ytest, predicted):
    print"cross validating"
    cv_scores = cross_val_score(model, Xtrain, Ytrain, cv=5)
    with open("data/results_RF_1_5.txt","w") as file:
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
    filename = "data/Features.csv" 
    X, Y, data = load_data_two_classes(filename)
    # print data.head()
    print len(data[data.y==1].index)
    print len(data[data.y==5].index)
    # X, Y, data = load_data(filename)
    Xtrain, Xtest, Ytrain, Ytest = split_datasets(X,Y)
    # print("SVM CLASSIFIER")
    # train_SVM(Xtrain, Ytrain, Xtest, Ytest)
    print("DECISION TREE CLASSIFIER")
    train_DecisionTree(Xtrain, Ytrain, Xtest, Ytest)
    # print("RANDOM FOREST CLASSIFIER")
    # train_RandomForest(Xtrain, Ytrain, Xtest, Ytest)
    print("done")
