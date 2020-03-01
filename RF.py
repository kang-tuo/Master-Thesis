
from matplotlib import style
style.use('fivethirtyeight')
import scipy.stats as sps
import numpy as np
from ToDataframe import *



data_path = r"C://Users//admin//Desktop//Output2.json"
TDF = ConvertJsonToDataframe()
data_df = TDF.main(data_path)
dataset = data_df.sample(frac=1)

dataset.columns = ['winnerSide', 'secondAfterBomb', 'aliveTNum', 'aliveCTNum', 'TP1dist', 'TP2dist', 'TP3dist',
                    'TP4dist', 'TP5dist', 'CTP1dist', 'CTP2dist', 'CTP3dist', 'CTP4dist', 'CTP5dist','TP1weapon',
                    'TP2weapon', 'TP3weapon', 'TP4weapon', 'TP5weapon', 'CTP1weapon', 'CTP2weapon',
                    'CTP3weapon', 'CTP4weapon', 'CTP5weapon']


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy

def InfoGain(data, split_attribute_name, target_name="winnerSide"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])

    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

def ID3(data, originaldata, features, target_attribute_name="winnerSide", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[
            np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

    elif len(features) == 0:
        return parent_node_class

    else:
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        features = np.random.choice(features, size=np.int(np.sqrt(len(features))), replace=False)
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in
                       features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}

        features = [i for i in features if i != best_feature]


        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data, dataset, features, target_attribute_name, parent_node_class)

            tree[best_feature][value] = subtree

        return (tree)


def predict(query, tree, default='p'):

    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result, dict):
                return predict(query, result)

            else:
                return result


def train_test_split(dataset):
    training_data = dataset.iloc[:round(0.75 * len(dataset))].reset_index(
        drop=True)  # We drop the index respectively relabel the index
    # starting form 0, because we do not want to run into errors regarding the row labels / indexes
    testing_data = dataset.iloc[round(0.75 * len(dataset)):].reset_index(drop=True)
    return training_data, testing_data

training_data = train_test_split(dataset)[0]
testing_data = train_test_split(dataset)[1]

def RandomForest_Train(dataset, number_of_Trees):
    # Create a list in which the single forests are stored
    random_forest_sub_tree = []

    # Create a number of n models
    for i in range(number_of_Trees):
        # Create a number of bootstrap sampled datasets from the original dataset
        bootstrap_sample = dataset.sample(frac=1, replace=True)

        # Create a training and a testing datset by calling the train_test_split function
        bootstrap_training_data = train_test_split(bootstrap_sample)[0]
        bootstrap_testing_data = train_test_split(bootstrap_sample)[1]

        # Grow a tree model for each of the training data
        # We implement the subspace sampling in the ID3 algorithm itself. Hence take a look at the ID3 algorithm above!
        random_forest_sub_tree.append(ID3(bootstrap_training_data, bootstrap_training_data,
                                          bootstrap_training_data.drop(labels=['winnerSide'], axis=1).columns))

    return random_forest_sub_tree

random_forest = RandomForest_Train(dataset, 50)

def RandomForest_Predict(query, random_forest, default='p'):
    predictions = []
    for tree in random_forest:
        predictions.append(predict(query, tree, default))
    return sps.mode(predictions)[0][0]

query = testing_data.iloc[0, :].drop('winnerSide').to_dict()
query_target = testing_data.iloc[0, 0]
print('target: ', query_target)
prediction = RandomForest_Predict(query, random_forest)

print('prediction: ', prediction)

#######Test the model on the testing data and return the accuracy###########
def RandomForest_Test(data, random_forest):
    data['predictions'] = None
    for i in range(len(data)):
        query = data.iloc[i, :].drop('winnerSide').to_dict()
        data.loc[i, 'predictions'] = RandomForest_Predict(query, random_forest, default='p')
    accuracy = sum(data['predictions'] == data['winnerSide']) / len(data) * 100
    print('The prediction accuracy is: ',sum(data['predictions'] == data['winnerSide'])/len(data)*100,'%')
    return accuracy

RandomForest_Test(testing_data, random_forest)

