'''
Written by Shengxuan Wang at OSU.
Used for data processing and helping to construct model
The name is from "Po" in the movie Kung Fu Panda
can use "import popanda as ppd" to use it

update on 12/23/2021: add the functions about dicision tree
'''
import pandas as pd
import math

# Name: z_core
# Fuction: z-core normalization equation, compute the z value
# Input: data x, mean and standard deviation of the data
# Output: z value, after normalize the data
def z_core(x, mean, std):
    if std == 0:
        z = 0
    else:
        z = (x-mean)/std
    return z

# Name: add_dummy
# Function: add fummy feature in the first position of the data set
# Input: the dataset
# Output: No
def add_dummy(df):
    title=df.columns.tolist()
    title.insert(0, 'dummy')
    df=df.reindex(columns=title)
    df['dummy']=[1]*df.shape[0]

# Name: split_xy
# Function: split x (features) and y (prediction)
# Input: data, the name of prediction class
# Output: two dataframes, x and y
def split_xy(df, NameOfY):
    real_y = df[NameOfY]
    x = df.drop([NameOfY], axis=1)
    return (x, real_y)

# Name: resetYvalue
# Function: reset the y value for some reasons, eg: reset 0 into -1, then m = 0, n = -1
# Input: y data, the original value, target value
# Output: No
def resetYvalue(y, m, n):
    for i in range(y.shape[0]):
        if y.iloc[i] == m:
            y.iloc[i] = n

# Name: squeeze
# Function: "squeeze" a feature to the end
# Input: target df and the feature you want to move
# Output: the result df
def squeeze(df, target_name):
    target = df[target_name]
    df = df.drop([target_name], axis=1)
    df = pd.concat([df, target], axis=1)
    return df

# Name: mergeAndresize_df
# Function: merge two dataframe, and resize it by oder, reset the index.
# Input: two dataframe, wanted size
# Output: the result dataframe
def mergeAndresize_df(df1,df2,size):
    if size < df1.shape[0] + df2.shape[0]:
        output = pd.concat([df1,resize_df(df2, size - df1.shape[0])], axis=0).reset_index(drop=True)
        return output
    else:
        raise Exception("The new size is bigger than original!")

# Name: resize_df
# Function: resize a dataframe
# Input: dataframe, wanted size
# Output: the result dataframe
def resize_df(df, size):
    if size < df.shape[0]:
        output = df.iloc[0:size]
        return output
    else:
        raise Exception("The new size is bigger than original!")

# Name: entropy
# Function: compute the entropy
# Input: "target_col" (should be a Series, pick by "iloc")
# Output: the entropy
def entropy(target_col):

    N = target_col.shape[0]
    distribution = target_col.value_counts()
    if distribution.shape[0] == 2:
        # have both 0 and 1
        num_0 = distribution.loc[0]
        num_1 = distribution.loc[1]
        P_0 = num_0/N
        P_1 = num_1/N
    
        entropy = -P_0*math.log2(P_0)-P_1*math.log2(P_1)
    else:
        entropy = 0
    
    return entropy

# Name: InfoGain
# Function: compute the information gain
# Input: data, the name of the feature for which the information gain should be calculated, the name of the target feature
# Output: the Information Gain
def InfoGain(data,split_attribute_name,target_name="class"):

    Hs = entropy(data.loc[:, target_name])

    left = data[data[split_attribute_name]==1]
    right = data[data[split_attribute_name]==0]

    if left.shape[0] != 0 and right.shape[0] != 0:
        Hs_1 = entropy(left.loc[:, target_name])
        Hs_2 = entropy(right.loc[:, target_name])
    
    else:
        if left.shape[0] == 0:
            Hs_1 = 0
        else:
            Hs_1 = entropy(left.loc[:, target_name])
        if right.shape[0] == 0:
            Hs_2 = 0
        else:
            Hs_2 = Hs_2 = entropy(right.loc[:, target_name])

    
    Information_Gain = Hs - (left.shape[0]/data.shape[0])*Hs_1 - (right.shape[0]/data.shape[0])*Hs_2

    return Information_Gain