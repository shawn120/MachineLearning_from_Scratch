'''
A Linear_regression Model
Shengxuan Wang at OSU
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import popanda as ppd

# date => y m d as a function, return df
def split_date(df):
    df_2 = df['date'].str.split('/', expand=True)
    df_2.rename(columns={0:'month',1:'day',2:'year'}, inplace=True)
    df = df.drop(['date'], axis=1)
    df = pd.concat([df_2, df], axis=1)
    return df

# replace yr_renovated by age_since_renovated as a function, return df
def generate_age(df):
    title=df.columns.tolist()
    title.insert(16, 'age_since_renovated')
    df=df.reindex(columns=title)
    age_since_renovated_index = list(df.columns).index('age_since_renovated')
    for i in range(df.shape[0]):
        if df['yr_renovated'][i] == 0:
            df.iloc[i, age_since_renovated_index] = df['year'][i]-df['yr_built'][i]
            # need to use "iloc" function, something like df[i][j] can only be read, cannot be written.
        else:
            df.iloc[i, age_since_renovated_index] = df['year'][i]-df['yr_renovated'][i]
    df = df.drop(['yr_renovated'], axis=1)
    return df

# Data preprocessing

# Excluding the ID
df = pd.read_csv("HousePrice_train.csv")
df = df.drop(['id'], axis=1)

df = split_date(df)
ppd.add_dummy(df)

# tranfer data type in to float32
df = df.astype(np.float32)

df = generate_age(df)

# generate new precessed training data set file
df.to_csv("processed_train.csv", index=False, encoding="utf-8")

# z-core Normalize part 1, store mean and std
title_list = []
mean_list = []
std_list = []
normalized_list = ["waterfront", "price", "dummy"]
j = 0
for i in df.columns.tolist():
    if i not in normalized_list:
        title_list.append(i)
        i_index = list(df.columns).index(i)
        mean = np.mean(df[i])
        mean_list.append(mean)
        std = np.std(df[i], ddof=1)
        std_list.append(std)
        j=j+1

dic = {}
for m in range(j):
    dic[title_list[m]] = [mean_list[m],std_list[m]]
# print(dic)
df_z_help = pd.DataFrame(dic)
df_z_help.to_csv("z_help.csv", index=False, encoding="utf-8")
# above is the z_help csv stores //mean// and //std//

# "use the mean and std in z_help to Normalize the input" as a function, return df
def Normalize_with_help(df):
    for i in df.columns.tolist():
        if i not in normalized_list:
            for j in range(df.shape[0]):
                z = ppd.z_core(df[i][j], df_z_help[i][0], df_z_help[i][1])
                i_index = list(df.columns).index(i)
                df.iloc[j, i_index] = z
    return df

# z-core Normalize part 2, compute z-core value, and replace
df = Normalize_with_help(df)

# generate a normalized training data set file
df.to_csv("normalized_train.csv", index=False, encoding="utf-8")

'''
above, finish all the data preprocessing, the outcome include:
a. processed_train.csv: before normalization but complete other processing
b. normalized_train.csv: after normalization
c. z_help.csv: the mean and std value of the data set
Next, start training
'''

# Some help functions for training

# create a function to compute the Los(aka MSE)
# input: w, x, y
# return: Los
def Los_MSE(w, train_x_n, train_real_y):
    # compute predicted y
    pre_y = np.matmul(train_x_n, w)
    # compute MSE
    delt_2 = (pre_y - train_real_y)**2
    Los = delt_2.sum() / train_x_n.shape[0]
    return Los

# create a function to compute the Gradient of Los(MSE)
# input: w, x, y
# output: G_Los
def Gra(w, train_x_n, train_real_y):
    pre_y = np.matmul(train_x_n, w)
    G_Los_pre_sum = train_x_n.mul((pre_y - train_real_y), axis=0)
    G_Los = G_Los_pre_sum.sum()*2/train_x_n.shape[0]
    return G_Los

# Batch Gradient Descent function, update w
# input: learning rate, w, x, y
# output: w (new one)
def Batch_GD(rate, w, train_x_n, train_real_y):
    gra = Gra(w, train_x_n, train_real_y)
    w = w - (rate * gra)
    return w

# Training

'''
in order to short the running time, re-read and rename the csv,
so that, if you have already runned the codes above (processing part),
you can comment all of them (except help function definition),
just use the generated file to run the code below,
that will save a lot of time
'''
df_n = pd.read_csv("normalized_train.csv")

# split x and y
train_x_n, train_real_y = ppd.split_xy(df_n, 'price')

# create default weight array
w=np.ones(train_x_n.shape[1])

# start iteration

# choose learning rate: 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5 to see what will happen
LR_list = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

# store loss
los_dict = {}
for j in LR_list:
    los_dict[str(j)]=[]

# creat a dictionary to store result w
w_dict = {}

# in different learning rate (j = learning rate)
for j in LR_list:
    i=0
    while i < 5000:
        L_0 = Los_MSE(w, train_x_n, train_real_y)
        los_dict[str(j)].append(L_0)
        w = Batch_GD(j, w, train_x_n, train_real_y)
        L_1 = Los_MSE(w, train_x_n, train_real_y)
        if i > 5:
            if L_1 > L_0:
                break
            if abs(L_0-L_1)<0.1:
                break
        i+=1
    # print(j)
    # print(w)
    w_dict[str(j)]=w
    w=np.ones(train_x_n.shape[1])

# print (los_dict)

# creat a figure index
j = 1
for i in LR_list:
    plt.figure(j)
    plt.plot(los_dict[str(i)])
    # plt.show()
    plt.title('1-%d-Ned-learningRate-%s' %(j, str(i)))
    plt.savefig('Figures/1-%d-Ned-learningRate-%s.png' %(j, str(i)))
    j = j+1

MSEs_train = {}

for j in LR_list:
    MSE = Los_MSE(w_dict[str(j)], train_x_n, train_real_y)
    MSEs_train[str(j)]=MSE

print("In training set, for different learning rate, the MSEs are")
print(MSEs_train)

# start validation

# processing the testing set
df_dev = pd.read_csv("HousePrice_dev.csv")
df_dev = df_dev.drop(['id'], axis=1)

df_dev = split_date(df_dev)
ppd.add_dummy(df_dev)

# tranfer data type in to float32
df_dev = df_dev.astype(np.float32)

df_dev = generate_age(df_dev)


# normalized
df_dev = Normalize_with_help(df_dev)

test_x_n, test_real_y = ppd.split_xy(df_dev, 'price')

# new a testing MSE dic for all the w
MSEs = {}

for j in LR_list:
    MSE = Los_MSE(w_dict[str(j)], test_x_n, test_real_y)
    MSEs[str(j)]=MSE
print("In validation set, for different learning rate, the MSEs are")
print(MSEs)

# start real prediction
print("After comparing the two results, we will pick the learning rate: 0.1, to predict a competition data set (without correct price value)")

test_df = pd.read_csv("Competition_test.csv")
id = test_df['id']
test_df = test_df.drop(['id'], axis=1)

test_df = split_date(test_df)
ppd.add_dummy(test_df)

# tranfer data type in to float32
test_df = test_df.astype(np.float32)

test_df = generate_age(test_df)

test_df = Normalize_with_help(test_df)

pre_y = np.matmul(test_df, w_dict["0.1"])
print("prediction is generated into a csv file")
pre_y = pd.concat([id, pre_y], axis=1)
pre_y.columns=['ID','price']
pre_y.to_csv("prediction.csv")