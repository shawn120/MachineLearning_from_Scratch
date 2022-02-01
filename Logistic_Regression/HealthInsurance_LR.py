import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import popanda as ppd

# some helping functions

# make log function easy to use
def log(x):
    return np.log(x)

# sigmoid function
def sigmoid(x):
    y = 1 / (1+np.exp(-x))
    return y

# loss function
# rp stand for regulaization parameter
def Loss_LR2(w, train_x, real_y, rp):
    # this is the part of (WT*Xi)
    pre_y = np.matmul(train_x, w)
    delt = -real_y * log(sigmoid(pre_y)) - (1-real_y)*log(1 - sigmoid(pre_y))
    w_0 = w['dummy']
    w = w.drop(['dummy'])
    tail = w**2
    w.loc['dummy'] = w_0
    Los = (delt.sum() + rp * tail.sum()) / train_x.shape[0]
    return Los

def Gra(w, train_x, real_y, rate):
    pre_y = np.matmul(train_x, w)
    delt = train_x.mul((real_y - sigmoid(pre_y)), axis=0)
    sum = delt.sum()
    gra = sum * rate / train_x.shape[0]
    return gra
    # print(gra)

def GD_LR2(w, train_x, real_y, rp, rate):
    gra = Gra(w, train_x, real_y, rate)
    w = w + gra
    # print(w)
    # print(gra)
    # for i in train_x.columns.tolist():
    #     w[i] = w[i] - rate * rp * w[i]
    w_0 = w['dummy']
    w = w.drop(['dummy'])
    w = w - rate * rp * w
    # print(w)
    w.loc['dummy'] = w_0
    # print(w)
    return w

def accy(predict_y, real_y):
    N = len(predict_y)
    i = 0
    for j in range(N):
        if predict_y[j] == real_y[j]:
            i+=1
    return i/N

def into_binary(raw):
    binary = []
    for i in range(raw.shape[0]):
        if raw[i]>0:
            binary.append(1)
        else:
            binary.append(0)
    return binary

# normalize
df = pd.read_csv("HealthInsurance_train.csv")
df = df.astype(np.float32)

need_normalize_list = ['Age', 'Annual_Premium', 'Vintage']
# z-core Normalize part 1, store mean and std
title_list = []
mean_list = []
std_list = []
j = 0
for i in df.columns.tolist():
    # print(i)
    if i in need_normalize_list:
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

df_z_help = pd.DataFrame(dic)
# above is the z_help csv stores //mean// and //std//

# "use the mean and std in z_help to Normalize the input" as a function, return df
def Normalize_with_help(df):
    for i in df.columns.tolist():
        if i in need_normalize_list:
            for j in range(df.shape[0]):
                z = ppd.z_core(df[i][j], df_z_help[i][0], df_z_help[i][1])
                i_index = list(df.columns).index(i)
                df.iloc[j, i_index] = z
    return df

df = Normalize_with_help(df)

# start training
# Part1: L2

# split x and y
train_x, real_y = ppd.split_xy(df, "Response")

train_x = ppd.squeeze(train_x, "dummy")

# create default weight array
w=np.zeros(train_x.shape[1])

# <============================================================================================================ >>>parameter choose for part 1<<< 
# rate = [0.01, 0.1, 1]
rate = [1]
rp = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
# rp = [1e-4, 1e-3, 1e-2]

# store loss
# los_dict = {}
# for i in rate:
#     for j in rp:
#         # los_dict[str(i)+"=rate, rp="+str(j)] = []
#         los_dict[str(j)] = []

# creat a dictionary to store result w
w_dict = {}

# creat a dictionary to store accuracy (AY)
AY_dict = {}

# start iteration
for i in rate:
    for j in rp:
        m = 0
        while m < 10000:
            if m == 0:
                # in first iteration, set a big L_0
                L_0 = math.inf
            # los_dict[str(i)+"=rate, rp="+str(j)].append(L_0)
            # los_dict[str(j)].append(L_0)
            w = GD_LR2(w, train_x, real_y, j, i)
            L_1 = Loss_LR2(w, train_x, real_y, j)
            if m > 5:
                if L_1 > L_0:
                    break
                if abs(L_0-L_1)<0.01:
                    break
            m+=1
            L_0 = Loss_LR2(w, train_x, real_y, j)
        y_pre_raw = np.matmul(train_x, w)
        y_predict = into_binary(y_pre_raw)
        # print(y_predict)
        AY = accy(y_predict, real_y)
        # AY_dict[str(i)+"=rate, rp="+str(j)]=AY
        AY_dict[str(j)]=AY
        w_dict[str(j)]=w
        w=np.zeros(train_x.shape[1])

# store the number of wi=0
w0_num = {}
for j in rp:
    n = 0
    for m in range(w_dict[str(j)].shape[0]):
        if w_dict[str(j)][m] == 0:
            n+=1
    w0_num[str(j)] = n

df_w = pd.DataFrame(w_dict)
df_w.to_csv("df_w.csv", index=False, encoding="utf-8")

# start testing

df_dev = pd.read_csv("HealthInsurance_dev.csv")
df_dev = df_dev.astype(np.float32)
df_dev = Normalize_with_help(df_dev)
dev_x, dev_real_y = ppd.split_xy(df_dev, "Response")
dev_x = ppd.squeeze(dev_x, "dummy")

AY_dev_dict = {}
for j in rp:
    y_dev_pre_raw = np.matmul(dev_x, w_dict[str(j)])
    y_dev_pre = into_binary(y_dev_pre_raw)
    AY = accy(y_dev_pre, dev_real_y)
    AY_dev_dict[str(j)]=AY

# <===================================================================================================================== Part1 result
print("******Part1 -- L2******")
print("the accuracy of training set using L2, by different regularization parameter:")
print(AY_dict)
print("the accuracy of dev set using L2, by different regularization parameter:")
print(AY_dev_dict)
print("the best lambda (regularization parameter) is 0.001.")
# print("the sparisity of w==0:")
# print(w0_num)

# <============================================================================================================ plot the figures for Part1
# x1 = list(AY_dict.keys())
# y1 = list(AY_dict.values())
# x2 = list(AY_dev_dict.keys())
# y2 = list(AY_dev_dict.values())
# plt.figure(1)
# plt.subplot(1,2,1)
# plt.plot(x1, y1)
# plt.title('L2 training accuracy VS λ')
# plt.subplot(1,2,2)
# plt.plot(x2, y2)
# plt.title('L2 validation accuracy VS λ')
# plt.savefig('L2 regularization.png')
# x3 = list(w0_num.keys())
# y3 = list(w0_num.values())
# plt.figure(2)
# plt.plot(x3, y3)
# plt.title('sparsity of the "0 weight" (L2)')
# plt.savefig('L2 sparsity.png')

# <============================================================================================================ >>>parameter choose for part 2<<< 
# rate = [0.01, 0.1, 1]
rate = [1]
rp = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

# Part2: L1
def Loss_LR1(w, train_x, real_y, rp):
    # this is the part of (WT*Xi)
    pre_y = np.matmul(train_x, w)
    delt = -real_y * log(sigmoid(pre_y)) - (1-real_y)*log(1 - sigmoid(pre_y))
    w_0 = w['dummy']
    w = w.drop(['dummy'])
    tail = np.abs(w)
    w.loc['dummy'] = w_0
    Los = (delt.sum() + rp * tail.sum()) / train_x.shape[0]
    return Los

def GD_LR1(w, train_x, real_y, rp, rate):
    gra = Gra(w, train_x, real_y, rate)
    w = w + gra
    w_0 = w['dummy']
    w = w.drop(['dummy'])
    w = np.sign(w)*np.maximum((np.abs(w)-rate*rp), 0)
    
    # for i in range(train_x.shape[1]-1):
    #     if abs(w[i]) < rate*rp:
    #         w[i] = 0
    #     else:
    #         w[i] = np.sign(w[i])*(abs(w[i])-rate*rp)
    
    # print(w)
    w.loc['dummy'] = w_0
    # print(w)
    return w

df_L1 = pd.read_csv("HealthInsurance_train.csv")
df_L1 = df_L1.astype(np.float32)
df_L1 = Normalize_with_help(df_L1)
L1_x, L1_real_y = ppd.split_xy(df_L1, "Response")
L1_x = ppd.squeeze(L1_x, "dummy")

# create default weight array
w=np.ones(L1_x.shape[1])
# creat a dictionary to store result w
w_dict_L1 = {}
# creat a dictionary to store accuracy (AY)
AY_dict_L1 = {}

# start iteration
for i in rate:
    for j in rp:
        m = 0
        while m < 10000:
            if m == 0:
                L_0 = math.inf
            w = GD_LR1(w, L1_x, L1_real_y, j, i)
            L_1 = Loss_LR1(w, L1_x, L1_real_y, j)
            if m > 5:
                if L_1 > L_0:
                    break
                if abs(L_0-L_1)<0.01:
                    break
            m+=1
            L_0 = Loss_LR1(w, L1_x, L1_real_y, j)
        y_pre_raw = np.matmul(L1_x, w)
        y_predict = into_binary(y_pre_raw)
        # print(y_predict)
        AY = accy(y_predict, L1_real_y)
        # AY_dict_L1[str(i)+"=rate, rp="+str(j)]=AY
        AY_dict_L1[str(j)]=AY
        w_dict_L1[str(j)]=w
        w=np.ones(L1_x.shape[1])

# store the number of wi=0
w0_num_L1 = {}
for j in rp:
    n = 0
    for m in range(w_dict_L1[str(j)].shape[0]):
        if w_dict_L1[str(j)][m] == 0:
            n+=1
    w0_num_L1[str(j)] = n

# print(w0_num_L1)

df_w = pd.DataFrame(w_dict_L1)
df_w.to_csv("df_w_L1.csv", index=False, encoding="utf-8")

# start testing

df_dev_L1 = pd.read_csv("HealthInsurance_dev.csv")
df_dev_L1 = df_dev_L1.astype(np.float32)
df_dev_L1 = Normalize_with_help(df_dev_L1)
dev_x_L1, dev_real_y_L1 = ppd.split_xy(df_dev_L1, "Response")
dev_x_L1 = ppd.squeeze(dev_x_L1, "dummy")

AY_dev_dict_L1 = {}
for j in rp:
    y_dev1_pre_raw = np.matmul(dev_x_L1, w_dict_L1[str(j)])
    y_dev1_pre = into_binary(y_dev1_pre_raw)
    AY = accy(y_dev1_pre, dev_real_y)
    AY_dev_dict_L1[str(j)]=AY

# <========================================================================================== part 2 result
print("******Part2 -- L1******")
print("the accuracy of training set using L1, by different regularization parameter:")
print(AY_dict_L1)
print("the accuracy of dev set using L1, by different regularization parameter:")
print(AY_dev_dict_L1)
print("the best lambda(regularization parameter) is 0.1.")
# print("the sparisity of w==0:")
# print(w0_num_L1)

# <=========================================================================plot the figures
# x1 = list(AY_dict_L1.keys())
# y1 = list(AY_dict_L1.values())
# x2 = list(AY_dev_dict_L1.keys())
# y2 = list(AY_dev_dict_L1.values())
# plt.figure(3)
# plt.subplot(1,2,1)
# plt.plot(x1, y1)
# plt.title('L1 training accuracy VS λ')
# plt.subplot(1,2,2)
# plt.plot(x2, y2)
# plt.title('L1 validation accuracy VS λ')
# plt.savefig('L1 regularization.png')
# x3 = list(w0_num_L1.keys())
# y3 = list(w0_num_L1.values())
# plt.figure(4)
# plt.plot(x3, y3)
# plt.title('sparsity of the "0 weight" (L1)')
# plt.savefig('L1 sparsity.png')

print("by comparing the result, use L2 to predict the competition set, and pick the regularization parameter 0.001")
df_pre = pd.read_csv("Competition_test.csv")
df_pre = df_pre.astype(np.float32)
df_pre = Normalize_with_help(df_pre)
df_pre = ppd.squeeze(df_pre, "dummy")


y_pre_pre_raw = np.matmul(df_pre, w_dict["0.001"])
y_pre_pre = into_binary(y_pre_pre_raw)
result = y_pre_pre

df_result = pd.DataFrame(result)
df_result.columns=['Response']
df_result.to_csv("result.csv", index=False, encoding="utf-8")
print("prediction is generated into a csv file")