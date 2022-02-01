import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import popanda as ppd

# Part 0: some functions

def transferTominusone(y):
    for i in range(y.shape[0]):
        if y.iloc[i] == 0:
            y.iloc[i] = -1

def accy(predict_y, real_y):
    N = len(predict_y)
    i = 0
    for j in range(N):
        if predict_y[j] * real_y[j] > 0:
            i+=1
    return i/N

# ------------------------------------- Read data and normalization
df = pd.read_csv("HealthInsurance_train.csv")
df = df.astype(np.float32)

# store mean and std
title_list = []
mean_list = []
std_list = []
j = 0
norm_list = ['Age', 'Annual_Premium', 'Vintage']
for i in df.columns.tolist():
    # print(i)
    if i in norm_list:
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
df_z_help.to_csv("z_help.csv", index=False, encoding="utf-8")
# above is the z_help csv stores //mean// and //std//

# "use the mean and std in z_help to Normalize the input" as a function, return df
def Normalize_with_help(df):
    for i in df.columns.tolist():
        if i in norm_list:
            for j in range(df.shape[0]):
                z = ppd.z_core(df[i][j], df_z_help[i][0], df_z_help[i][1])
                i_index = list(df.columns).index(i)
                df.iloc[j, i_index] = z
    return df

df = Normalize_with_help(df)

train_x, real_y = ppd.split_xy(df, "Response")
transferTominusone(real_y)

# for dev data
df_dev = pd.read_csv("HealthInsurance_dev.csv")
df_dev = df_dev.astype(np.float32)
df_dev = Normalize_with_help(df_dev)
dev_x, dev_y = ppd.split_xy(df_dev, "Response")
transferTominusone(dev_y)

# -------------------------------------------------- Part 1: Average / Online Perceptron
# start training
# maximum of iterations
maxiter = 100

w = np.zeros(train_x.shape[1])
w_ = np.zeros(train_x.shape[1])

AY_train_online = []
AY_train_average = []
AY_dev_online = []
AY_dev_average = []

stop_online = [0,0]
stop_average = [0,0]

s = 1
for i in range(maxiter):
    for j in range(train_x.shape[0]):
        temp = np.dot(train_x.iloc[j:j+1], w) * real_y[j]
        if temp <= 0:
            w = w + real_y[j]*train_x.iloc[j]
        w_ = (s*w_+w) / (s+1)
        s += 1
    pre_train_online = np.matmul(train_x, w)
    pre_train_average = np.matmul(train_x, w_)
    pre_dev_online = np.matmul(dev_x, w)
    pre_dev_average = np.matmul(dev_x, w_)
    
    AY_train_online.append(accy(pre_train_online, real_y))
    AY_train_average.append(accy(pre_train_average, real_y))
    AY_dev_online.append(accy(pre_dev_online, dev_y))
    AY_dev_average.append(accy(pre_dev_average, dev_y))
    
    if stop_online[0] < accy(pre_dev_online, dev_y):
        stop_online = [accy(pre_dev_online, dev_y), i]
    if stop_average[0] < accy(pre_dev_average, dev_y):
        stop_average = [accy(pre_dev_average, dev_y), i]

print("for online perceptron, the best stopping point is %d, the accurancy now is %f" %(stop_online[1], stop_online[0]))
print("for average perceptron, the best stopping point is %d, the accurancy now is %f" %(stop_average[1], stop_average[0]))
# print(AY_train_online)
# print(AY_train_average)
# print(AY_dev_online)
# print(AY_dev_average)
plt.figure(1)
plt.plot(AY_train_online)
plt.title("Accuracy of Online Perceptron in Training Data")
plt.xlabel("Iteration Numbers")
plt.ylabel("Accuracy")
plt.savefig("1-1-AY_train_online")
plt.figure(2)
plt.plot(AY_train_average)
plt.title("Accuracy of Average Perceptron in Training Data")
plt.xlabel("Iteration Numbers")
plt.ylabel("Accuracy")
plt.savefig("1-2-AY_train_average")
plt.figure(3)
plt.plot(AY_dev_online)
plt.title("Accuracy of Online Perceptron in Validation Data")
plt.xlabel("Iteration Numbers")
plt.ylabel("Accuracy")
plt.savefig("1-3-AY_dev_online")
plt.figure(4)
plt.plot(AY_dev_average)
plt.title("Accuracy of Average Perceptron in Validation Data")
plt.xlabel("Iteration Numbers")
plt.ylabel("Accuracy")
plt.savefig("1-4-AY_dev_average")
