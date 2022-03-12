import numpy as np
import pandas as pd
import pprint as pprint
#################################################PACKAGES###################################################################
###############################################DATA BUILDER#################################################################
def build_data(remove):
    input_table = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape') # reading the csv file
    if remove == True:
        input_table[['day', 'month', 'year']] = input_table['Date'].str.split('/', expand=True)  #split date to day,month and year
        input_table = input_table.drop(columns='Date') #remove uncessery data
        input_table = input_table.drop(columns='year') #remove uncessery data
    copy_column = input_table[["Rented Bike Count"]]
    input = input_table.drop(columns='Rented Bike Count')
    output = copy_column.copy()
    dataframe_actual = pd.concat([input, output], axis=1)
    if remove == True:
        features = input.columns[:-1]  # check the features option for this data
    elif remove == False:
        features = input.columns[:]
    output_feature_name = 'Rented Bike Count'
    dataframe = dataframe_actual
    return  dataframe,features,output_feature_name

def getCategorize(tableChange): #to give categorize for features
    normalize_features = tableChange.describe().columns[:-1]
    for feature in normalize_features:
        if (feature != 'Hour' and feature != 'day' and feature != 'month' and feature != 'year' and feature != 'Seasons' and feature != 'Holiday' and feature != 'Functioning Day'):
            if feature != 'Solar Radiation (MJ/m2)' and feature != 'Rainfall(mm)' and feature != 'Snowfall (cm)':
                tableChange[feature] = pd.qcut(tableChange[feature], q=3, labels=('0', '1', '2'))
            else:
                tableChange[feature] = pd.cut(tableChange[feature], 3, labels=('0', '1', '2'))
    tableChange.loc[(tableChange['Rented Bike Count'] <= 650), 'Rented Bike Count'] = 0
    tableChange.loc[(tableChange['Rented Bike Count'] > 650), 'Rented Bike Count'] = 1
    return tableChange
###############################################DATA BUILDER#################################################################
def getAccuracyAndError(return_answer): #calculate the accuracy and error for answer
    accuracy = (return_answer[0][0] + return_answer[1][1]) / (
return_answer[0][0] + return_answer[0][1] + return_answer[1][0] + return_answer[1][1])
    error = (return_answer[0][1] + return_answer[1][0]) / (
                return_answer[0][0] + return_answer[0][1] + return_answer[1][0] + return_answer[1][1])
    return accuracy,error

def calc_entropy(sub_data): # this function calculates the entropy
    values = np.unique(sub_data)
    countsByvalues = []
    for i in range(len(values)):
        countsByvalues.append(0)
    for i in range (len(sub_data)):
        for j in range(len(values)):
            if sub_data.iat[i] == values[j]:
                countsByvalues[j] = countsByvalues[j] + 1
    entropy = 0
    for i in range(len(values)):
        entropy = entropy + np.sum([(-countsByvalues[i] / np.sum(countsByvalues)) * np.log2(countsByvalues[i] / np.sum(countsByvalues))])# calculating entropy using formula
    return entropy

def IG(data, feature, output_name): #  this function will calculate the information gain
    vals = np.unique(data[feature])
    counts = []
    for i in range(len(vals)):
        counts.append(0)
    for i in range (len(data[feature])):
        for j in range(len(vals)):
            if data[feature].iat[i] == vals[j]:
                counts[j] = counts[j] + 1
    entropy_weight = 0
    for i in range(len(vals)):
        weight = entropy_weight + np.sum([(counts[i] / np.sum(counts)) * calc_entropy(data.where(data[feature] == vals[i]).dropna()[output_name])])
    sum_entropy = calc_entropy(data[output_name])
    information_gain = sum_entropy - weight
    return information_gain


def createDecisionTree(data, originaldata, features, output_name,iteration, parents): #create the tree for answe
    if len(data) == 0:
        return print("Cant solve this")
    elif len(features) == 0 or iteration>4:
        return parents
    elif len(np.unique(data[output_name])) <= 1:  # if there only one answer for the data
        return np.unique(data[output_name])[0]
    else:
        parents = np.unique(data[output_name])[np.argmax(np.unique(data[output_name], return_counts=True)[1])]
        item_values = []
        for feature in features:
            item_values.append(IG(data, feature, output_name))# calculate the information gain
        best_feature_index = np.argmax(item_values)  # # find best feature with high IG
        best_feature = features[best_feature_index]  # find best feature
        entropy_best = item_values[best_feature_index]
        tree = {best_feature: {} ,"entropy calculate":entropy_best} # create empty feature with best feature as root node
        next_feature = []
        index_num_feature = 0
        for index_feature in features:
            if index_feature != best_feature:
                next_feature.append(index_feature)
            index_num_feature = index_num_feature + 1
        features = next_feature
        for value in np.unique(data[best_feature]):  #creating next tree for each feature
            index = value
            next_data = data.where(data[best_feature] == value).dropna()
            subtree = createDecisionTree(next_data, originaldata, features, output_name,iteration+1,parents)
            tree[best_feature][index] = subtree  # put the tree under tree
        return (tree)

def checkFuture(question, tree):
    zero = 0
    listOfKeys = list(question.keys())
    listOftree = list(tree.keys())
    for word in listOfKeys:
        if word in listOftree:
            try:
                answer = tree[word][question[word]]
            except:
                return zero
            checkIfTree =  isinstance(answer, dict)
            if checkIfTree == False:
                return answer
            else:
                return checkFuture(question, answer)

def testDataFrame(testing_data, tree):
    predicted = pd.DataFrame(columns=["predict"])  # creating the data frame for result
    outputs = testing_data.iloc[:, :-1].to_dict(orient="records")  # converting data to dictionary
    for i in range(len(testing_data)):
        predicted.loc[i, "predict"] = checkFuture(outputs[i], tree)  # predicting the output
    return predicted

def getTableconclusion(predicted_result, data_testing):
    df_confusion = pd.crosstab(predicted_result.to_numpy().T[0], data_testing['Rented Bike Count'].to_numpy(),rownames=['Actual'], colnames=['Predicted'], margins=True)
    return df_confusion

#############################################build_tree function###############################################################
def build_tree(ratio):
    to_remove = True
    dataframe, features , output_feature_name  = build_data(to_remove) #create the data
    dataframe = getCategorize(dataframe) #to give category for the features needed
    dataframe = dataframe.sample(frac=1).reset_index(drop=True) #to make shuffle from the data
    length = len(dataframe)
    learnData = dataframe.iloc[0:int(ratio * length), :].reset_index(drop=True) #take 60% to learn
    test_data = dataframe.iloc[int(ratio * length):, :].reset_index(drop=True) #take 40% to test
    tree = createDecisionTree(learnData, learnData, features, output_feature_name,0,None) # create tree
    pprint.pprint(tree)
    predicted_result = testDataFrame(test_data, tree)
    end_print = getTableconclusion(predicted_result, test_data)
    print('------------------------------------')
    print('Conclusion for the tree data:')
    print(end_print)
    accuracyCalculate , errorCalaculate = getAccuracyAndError(end_print)
    print('Accuracy of the tree: ', accuracyCalculate)
    print('Error Rate of the tree: ', errorCalaculate)

#############################################build_tree function################################################################
#############################################is_busy function################################################################
def is_busy(row_input):
    to_remove = False
    temp_row = row_input
    temp_row = np.array([temp_row])
    dataframebusy, features , output_Busy_feature_name  = build_data(to_remove) #create the data
    temp_row = pd.DataFrame(temp_row, columns=features)  # convert to dataframe
    dataframebusy[['day', 'month', 'year']] = dataframebusy['Date'].str.split('/', expand=True)  # split date to day,month
    dataframebusy = dataframebusy.drop(columns='Date')
    dataframebusy = dataframebusy.drop(columns='year')
    temp_row[['day', 'month', 'year']] = temp_row['Date'].str.split('/', expand=True)
    temp_row = temp_row.drop('Date', axis=1)
    temp_row = temp_row.drop('year', axis=1)
    dataframebusy = dataframebusy.append(temp_row).reset_index(drop=True)
    dataframebusy.at[8760,'Rented Bike Count'] = 0
    dataframebusy = getCategorize(dataframebusy) #to give category for the features needed
    for i in range (8760):
        dataframebusy = dataframebusy.drop(i)
    to_remove = True
    dataframe, features, output_feature_name = build_data(to_remove)  # create the data
    dataframe = getCategorize(dataframe)  # to give category for the features needed
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    length = len(dataframe)
    learnData = dataframe.iloc[0:int(0.6 * length), :].reset_index(drop=True)
    tree_busy = createDecisionTree(learnData, learnData, features, output_feature_name, 0, None)  # create tree
    dataframebusy = dataframebusy.drop(columns='Rented Bike Count')
    predicted_result = testDataFrame(dataframebusy, tree_busy)
    find_busy = int(predicted_result['predict'][0])
    if find_busy == 1:
        print("------------------------------------")
        print("With this row input:")
        print("it will be 1 at that day")
        print("------------------------------------")
    else:
        print("------------------------------------")
        print("With this row input:")
        print("it will be 0 at that day")
        print("------------------------------------")
        print()
#############################################is_busy function################################################################
#############################################tree_error function################################################################
def tree_error(k):
    print("------------------------------------")
    print("----------------K Fold--------------")
    dataKerror, features , output_feature_name  = build_data(True) #create the data
    dataKerror = getCategorize(dataKerror)
    dataKerror = dataKerror.sample(frac=1).reset_index(drop=True)
    lengthdataKerror = len(dataKerror)
    fold_length = lengthdataKerror / k
    framesKfold = []
    size = 0
    for i in range(k):
        framesKfold.append(dataKerror.iloc[int(size):int(size + fold_length),:])
        size = size + fold_length
    errorsList = []
    accuracyList = []
    for i in range(k):
        print("-----------------Iteration", i+1,"-------------------")
        test_data = 0
        train_data = None
        for j in range(k):
            if i == j:
                test_data = framesKfold[i]
            else:
                if train_data is None:
                    train_data = framesKfold[j]
                else:
                    train_data = pd.concat([train_data, framesKfold[j]], axis=0)
        tree = createDecisionTree(train_data, train_data, features,output_feature_name,0,None)
        predicted_result = testDataFrame(test_data, tree)
        end_print = getTableconclusion(predicted_result, test_data)
        accuracy , error = getAccuracyAndError(end_print)
        errorsList.append(error)
        accuracyList.append(accuracy)
        print(end_print)
        print("Error:", error, "--Accuracy:", accuracy)

#############################################tree_error function################################################################
################################################main function###################################################################
if __name__ == "__main__":
    build_tree(0.6)
    row_input = ["25/02/2018", 16, -2.95, 55.12, 0.7, 1741, 4.1, 0, 0, 0, 'Autumn', 'No Holiday', 'Yes']
    is_busy(row_input)
    tree_error(3)
################################################main function###################################################################