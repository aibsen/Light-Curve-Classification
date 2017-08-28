import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import preprocessing

def load_data(filename):
    #reading data from csv
    data = pd.read_csv(filename, sep=",", names=["id", "amplitude", "beyond1std", "pairslopetrend","y"], skiprows=1)
    X = data[["id","amplitude", "beyond1std", "pairslopetrend"]]
    Y = data["y"] 
    #how many classes?
    class_list = Y.unique()
    number_of_classes = len(class_list)
    print "there are", number_of_classes,"classes :",class_list
    return X,Y,data

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
    print(metrics.classification_report(Ytest, predicted))
    # print(metrics.confusion_matrix(expected, predicted))

def train_DecisionTree(Xtrain, Ytrain, Xtest, Ytest):
    model = DecisionTreeClassifier()
    print("training Desicion Tree Classifier")
    model.fit(Xtrain, Ytrain)
    print("testing Desicion Tree Classifier")
    predicted = model.predict(Xtest)
    #check how prediction went
    print(metrics.classification_report(Ytest, predicted))
    # print(metrics.confusion_matrix(expected, predicted))

if __name__ == "__main__":
    print("LOADING DATA")
    filename = "data/Features.csv" 
    X, Y, data = load_data(filename)
    Xtrain, Xtest, Ytrain, Ytest = split_datasets(X,Y)
    # print("SVM CLASSIFIER")
    # train_SVM(Xtrain, Ytrain, Xtest, Ytest)
    print("DECISION TREE CLASSIFIER")
    train_DecisionTree(Xtrain, Ytrain, Xtest, Ytest)
    print("done")
