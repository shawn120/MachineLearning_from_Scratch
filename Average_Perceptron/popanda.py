'''
Written by Shengxuan Wang at OSU.
Used for processing dataframe, especially for machine learning.
The name is from "Po" in the movie Kung Fu Panda
Use "import popanda as .." to use
'''
import pandas as pd

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