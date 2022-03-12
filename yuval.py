import csv
import random

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def buildTree(k):
    seoalBikeData = pd.read_csv('SeoulBikeData.csv', encoding_errors="ignore")
    seoalBikeData = convertTocategories(seoalBikeData)
    seoalBikeData = seoalBikeData.drop(['Date'], axis=1)
    seoalBikeData = shuffle(seoalBikeData) # to choose data arbitrarily
    treeBase = divideData(k,seoalBikeData, False)
    Y_train=treeBase.iloc[:,:1].values
    X_train = treeBase.iloc[:,1:].values
    classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
    classifier.fit(X_train,Y_train)
    classifier.print_tree()
    validationSet = divideData(k, seoalBikeData, True)
    X_test = validationSet.iloc[:,1:].values
    Y_test = validationSet.iloc[:,:1].values
    Y_pred = classifier.predict(X_test)
    print(Y_pred)
    print(Y_test)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(Y_test, Y_pred))
    classifier.calcError(Y_test,Y_pred)


def divideData(k,seoalBikeData, validationSet):# divide the data to tree building and validation
    treeDataSize = round(len(seoalBikeData)*k)
    if validationSet == False:
        return seoalBikeData.iloc[0:treeDataSize] # for tree training
    else:
        return seoalBikeData.iloc[treeDataSize:len(seoalBikeData)] # for validation

def convertTocategories(df):
    df['Month-Date']= pd.to_datetime(df['Date']).dt.strftime('%m') # create new col who represent the month
    df['Day-Date'] = pd.to_datetime(df['Date']).dt.strftime('%d') # create new col who represent the days
    df.loc[df['Rented Bike Count'] < 650, 'Rented Bike Count'] = 0 # not a busy day
    df.loc[df['Rented Bike Count'] >= 650, 'Rented Bike Count'] = 1 # a busy day
    df['Hour'] = pd.qcut(df['Hour'], q=8, labels=('0-2', '3-5', '6-8','9-11','12-14','15-17','18-20','21-23')) # convert to 3 hours bins
    df['Temperature(C)'] = pd.qcut(df['Temperature(C)'], q=3, labels=('0', '1', '2'))
    df['Humidity(%)'] = pd.qcut(df['Humidity(%)'], q=3, labels=('0', '1', '2'))
    df['Wind speed (m/s)'] = pd.qcut(df['Wind speed (m/s)'], q=3, labels=('0', '1', '2'))
    df['Visibility (10m)'] = pd.qcut(df['Visibility (10m)'], q=3, labels=('0', '1', '2'))
    df['Dew point temperature(C)'] = pd.qcut(df['Dew point temperature(C)'], q=3, labels=('0', '1', '2'))
    df['Solar Radiation (MJ/m2)'] = pd.cut(df['Solar Radiation (MJ/m2)'], 3, labels=('0', '1', '2'))
    df['Rainfall(mm)'] = pd.cut(df['Rainfall(mm)'], 3, labels=('0', '1', '2'))
    df['Snowfall (cm)'] = pd.cut(df['Snowfall (cm)'], 3, labels=('0', '1', '2'))

    return df

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor '''

        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.value = value


class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''

        # initialize the root of the tree
        self.root = None

        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:, :-1], dataset[:, -1]

        num_samples, num_features = np.shape(X)

        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"] > 0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])

        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")

        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # return best split
        return best_split

    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''

        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child):
        ''' function to compute information gain '''

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    def entropy(self, y):
        ''' function to compute entropy '''

        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''

        Y = list(Y)
        return max(Y, key=Y.count)

    def print_tree(self, tree=None, indent = " "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)


    def fit(self, X, Y):
        ''' function to train the tree '''
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        ''' function to predict new dataset '''

        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''

        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def calcError(self,Y_test,Y_pred):  # returns the ratio of errors from group of examples
        numOfErrors = 0
        Y_test = np.array(Y_test)
        Y_pred = np.array(Y_pred)
        for i in Y_test:
            if Y_test[i]!=Y_pred[i]:
                numOfErrors = numOfErrors + 1
        print(numOfErrors)
        print(1-(numOfErrors / len(Y_test)))
        return numOfErrors / len(Y_test)



def main():
    buildTree(0.6)

if __name__ == '__main__':
    main()
