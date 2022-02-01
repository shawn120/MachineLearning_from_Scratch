import pandas as pd
import matplotlib.pyplot as plt
import popanda as ppd

class Node():
	def __init__(self, prediction, feature, split, left_tree, right_tree):
		self.prediction = prediction
		self.feature = feature
		self.split = split
		self.left_tree = left_tree
		self.right_tree = right_tree


def DecisionTree(data, features, depth, maxdepth, target_attribute_name="class"):
    """
    data = the data for which the decision tree building algorithm should be run --> In the first run this equals the total dataset

    features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
    we have to remove features from our dataset once we have splitted on a feature
    
    target_attribute_name = the name of the target attribute
    
    depth = the current depth of the node in the tree --- this is needed to remember where you are in the overall tree
    maxdepth =  the stopping condition for growing the tree
    """

    node = Node(None, None, None, None, None)
    #  prediction, feature, split, left_tree, right_tree

    # compute prediction
    pre_0 = data[data[target_attribute_name]==0]
    pre_1 = data[data[target_attribute_name]==1]

    # predict the mojority
    if pre_0.shape[0] > pre_1.shape[0]:
        node.prediction = 0
    else:
        node.prediction = 1
    
    # reach the maxdepth, return
    if depth == maxdepth:
        return node 
    depth += 1

    # if it is pure, return immeidately
    if pre_0.shape[0] == 0 or pre_1.shape[0] == 0:
        node.feature = -1
        return node
    
    # compute the best feature
    gain = -1000
    for feature in features:
        if feature != None:
            gain_temp = ppd.InfoGain(data, feature)
            if gain_temp > gain:
                gain = gain_temp
                feature_used = feature
                index_pick = features.index(feature_used)
    gain_list.append(gain)
    node.feature = index_pick

    # use the feature_used to train
    # mark the used feature as used === delete it
    features[index_pick] = None 
    # compute the split value
    left_0 = data[data[feature_used]==0]
    right_1 = data[data[feature_used]==1]

    if left_0.shape[0] > right_1.shape[0]:
        node.split = 0
    else:
        node.split = 1

    features_left = features.copy()
    features_right = features.copy()

    # compute left tree and right tree
    group = data.groupby(feature_used)
    if len(list(group)) == 2:
        data_left = list(group)[0][1].reset_index(drop=True)
        data_right = list(group)[1][1].reset_index(drop=True)
        node.left_tree = DecisionTree(data_left, features_left, depth, maxdepth)
        node.right_tree = DecisionTree(data_right, features_right, depth, maxdepth)
    else:
        if list(group)[0][0] == 0:
            data_left = list(group)[0][1].reset_index(drop=True)
            node.left_tree = DecisionTree(data_left, features_left, depth, maxdepth)
        else:
            data_right = list(group)[0][1].reset_index(drop=True)
            node.right_tree = DecisionTree(data_right, features_right, depth, maxdepth)

    return node

def predict(example,tree,default = 1):

    feature_index = tree.feature
    if tree.left_tree == None and tree.right_tree == None:
        return tree.prediction
    else:
        if example[feature_index] == 0:
            if tree.left_tree != None:
                return predict(example, tree.left_tree)
            else:
                return tree.prediction
        else:
            if tree.right_tree != None:
                return predict(example, tree.right_tree)
            else:
                return tree.prediction

gain_list = []
# main part start
# predict is a list, real is a dataframe
def accuracy(predict, real):
    if len(predict) == real.shape[0]:
        N = len(predict)
        j = 0
        for i in range(N):
            if predict[i] == real.iloc[i]:
                j+=1
        
        accuracy = j/N

        return accuracy
    else:
        return "check if use the same data set"


data = pd.read_csv("mushroom_train.csv")
vald = pd.read_csv("mushroom_val.csv")

maxdepth = 5

features_list = list(data)
features_list.remove("class")

tree = DecisionTree(data, features_list, 0, maxdepth)

pre = []
for i in range(data.shape[0]):
    example = data.iloc[i]
    pre.append(predict(example, tree))

real = data.loc[:,"class"]
acc = accuracy(pre, real)

# <-----------------------------  output
print("-----result-----")
print("(a)")
features_original = list(data)
print("the first 3 splits are:")
print(tree.split, tree.left_tree.split, tree.right_tree.split)
print("their information gains are:")
print(gain_list[0], gain_list[1], gain_list[2])

maxdepth_list = [2, 5,6,7,8, 10, 15, 20, 25, 30, 100]
tree_dict = {}

features_list = list(data)
features_list.remove("class")
for maxdepth in maxdepth_list:
    tree_dict[maxdepth] = DecisionTree(data, features_list, 0, maxdepth)

accuracy_dict = {}
acc_vald_dict = {}

for maxdepth in maxdepth_list:
    pre = []
    pre_vald = []
    for j in range(data.shape[0]):
        example = data.iloc[j]
        pre.append(predict(example, tree_dict[maxdepth]))
        real = data.loc[:,"class"]
        accuracy_dict[maxdepth] = accuracy(pre, real)
        
    for j in range(vald.shape[0]):
        example_vald = vald.iloc[j]
        pre_vald.append(predict(example_vald, tree_dict[maxdepth]))
        real_vald = vald.loc[:, "class"]
        acc_vald_dict[maxdepth] = accuracy(pre_vald, real_vald)


print("(b)")
print("the accuracy of training set of different dmax are:")
print(accuracy_dict)
print("the accuracy of validation set of different dmax are:")
print(acc_vald_dict)

plt.figure(1)
plt.plot(list(accuracy_dict.keys()), list(accuracy_dict.values()))
plt.title("Accuracy of Training Set VS Different Dmax")
plt.xlabel("Dmax")
plt.ylabel("Accuracy")
plt.savefig("1-b-1-AY_train")
plt.figure(2)
plt.plot(list(acc_vald_dict.keys()), list(acc_vald_dict.values()))
plt.title("Accuracy of Validation Set VS Different Dmax")
plt.xlabel("Dmax")
plt.ylabel("Accuracy")
plt.savefig("1-b-2-AY_train")