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

# This function is a slightly diferent with that in tree.
def DecisionTree(data, features, features_full, depth, maxdepth, target_attribute_name="class"):
    """
    data = the data for which the decision tree building algorithm should be run --> In the first run this equals the total dataset

    features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
    we have to remove features from our dataset once we have splitted on a feature
    
    features_full = the original, full list of features. Because the "features" will be motified in every recursion.
    
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
                index_in_full = features_full.index(feature_used)
                index = features.index(feature_used)
                
    gain_list.append(gain)
    node.feature = index_in_full

    # use the feature_used to train
    # mark the used feature as used === delete it
    features[index] = None 
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
        node.left_tree = DecisionTree(data_left, features_left, features_full, depth, maxdepth)
        node.right_tree = DecisionTree(data_right, features_right, features_full, depth, maxdepth)
    else:
        if list(group)[0][0] == 0:
            data_left = list(group)[0][1].reset_index(drop=True)
            node.left_tree = DecisionTree(data_left, features_left, features_full, depth, maxdepth)
        else:
            data_right = list(group)[0][1].reset_index(drop=True)
            node.right_tree = DecisionTree(data_right, features_right, features_full, depth, maxdepth)

    return node

def predict(example,tree, default = 1):

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


def RandomForest_set(data, m, dmax, size):
    # T: the number of trees
    # m: the number of features to sub-sample in each test selection step.
    # dmax: the maximum depth of the trees in your random forest.
    forest_set = []
    for i in range(size):
        N = data.shape[0]
        pick = data.sample(frac = 2/3)
        add = pick.sample(frac = 0.5)
        new_data = pd.concat([pick, add])
        
        temp = new_data.drop(['class'], axis=1)
        temp = temp.sample(m, axis=1)
        features_list = list(temp)
        forest_set.append(DecisionTree(new_data, features_list,features_full, 0, dmax))
    
    return forest_set

def RandomForest_pick(T, forest_set):
    forest = []
    for i in range(T):
        forest.append(forest_set[i])
    return forest

def forest_prediction(example, forest):
    predictions = []
    n = len(forest)
    for i in range(n):
        predictions.append(predict(example, forest[i], i))
    # print(predictions)
    vote = max(predictions, key = predictions.count)
    # print(vote)
    return vote


data = pd.read_csv("mushroom_train.csv")
vald = pd.read_csv("mushroom_val.csv")
features_full = list(data)
features_full.remove("class")
dmax_list = [1, 2, 5]
m_list = [5, 10, 25, 50]
T_list = [10, 20, 30, 40, 50]
maxsize = 50

print("-----result-----")
fndex = 0
for dmax in dmax_list:
    fndex+=1
    acc_dic = {}
    acc_v_dic = {}
    for m in m_list:
        forest_set = RandomForest_set(data, m, dmax, maxsize)
        acc_dic[m] = []
        acc_v_dic[m] = []
        for T in T_list:
            forest = RandomForest_pick(T, forest_set)
            pre_ft = []
            for i in range(data.shape[0]):
                example = data.iloc[i]
                pre_ft.append(forest_prediction(example, forest))

            real = data.loc[:,"class"]
            acc = accuracy(pre_ft,real)
            # print("Train: dmax:%d, m:%d, T:%d" %(dmax,m,T))
            # print(acc)
            acc_dic[m].append(acc)
            
            pre_fv = []
            for i in range(vald.shape[0]):
                example_v = vald.iloc[i]
                pre_fv.append(forest_prediction(example_v, forest))
            
            real_v = vald.loc[:,"class"]
            acc_v = accuracy(pre_fv,real_v)
            print("Validation: dmax:%d, m:%d, T:%d" %(dmax,m,T))
            print(acc_v)
            acc_v_dic[m].append(acc_v)
    plt.figure(fndex)
    for m in m_list:
        plt.plot(T_list, acc_dic[m], label = "m=%d" %m)
        plt.legend()
    plt.title("train accuracy vs T when dmax=%d" %dmax)
    plt.xlabel("T")
    plt.ylabel("train accuracy")
    plt.savefig("train_dmax=%d" %dmax)
    fndex+=1
    plt.figure(fndex)
    for m in m_list:
        plt.plot(T_list, acc_v_dic[m], label = "m=%d" %m)
        plt.legend()
    plt.title("validation accuracy vs T when dmax=%d" %dmax)
    plt.xlabel("T")
    plt.ylabel("validation accuracy")
    plt.savefig("vali_dmax=%d" %dmax)