import time
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

df_dev = pd.read_csv("HealthInsurance_dev.csv")
df_dev = df_dev.astype(np.float32)
df_dev = Normalize_with_help(df_dev)
dev_x, dev_y = ppd.split_xy(df_dev, "Response")
transferTominusone(dev_y)

def Kk(x1, x2, p):
    temp = np.dot(x1, x2.T)
    op = np.power(temp, p)
    return op

# -------------------------------------------------- Perceptron with polynomial kernel
# start training
maxiter = 100

p_list = [1,2,3,4,5]

bestA_train = {}
bestA_dev = {}

AY_train = {}
AY_dev = {}

K_dict = {}
for p in p_list:
    bestA_train[p] = [-1, 0]
    bestA_dev[p] = [-1, 0]
    AY_train[p] = []
    AY_dev[p] = []

    K_dict[p] = Kk(train_x, train_x, p)
    
    a = np.zeros(train_x.shape[0])
    
    for i in range(maxiter):
        for j in range(train_x.shape[0]):
            u = 0
            temp = np.multiply(K_dict[p][j],a)
            u = (temp * real_y).sum()
            if u*real_y[j] <= 0:
                a[j] = a[j] + 1
        pre_train = np.matmul(K_dict[p],a*real_y)
        AY_train[p].append(accy(pre_train, real_y))
        if accy(pre_train, real_y) > bestA_train[p][1]:
            bestA_train[p][1] = accy(pre_train, real_y)
            bestA_train[p][0] = i

        K_dev = Kk(dev_x, train_x, p)
        pre_dev = np.matmul(K_dev,a*real_y)
        AY_dev[p].append(accy(pre_dev, dev_y))
        if accy(pre_dev, dev_y) > bestA_dev[p][1]:
            bestA_dev[p][1] = accy(pre_dev, dev_y)
            bestA_dev[p][0] = i

print("****** Perceptron with polynomial kernel ******")
for i in p_list:
    print("for p = %d, the best training accurancy is %f (i = %d), the best validation accuracy is %f (i = %d)" %(i, bestA_train[i][1], bestA_train[i][0], bestA_dev[i][1], bestA_dev[i][0]))
    plt.figure(i+4)
    plt.plot(AY_train[i],label="Training Accuracy")
    plt.legend()
    plt.plot(AY_dev[i],label="Validation Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy-p=%d" %(i))
    plt.xlabel("Iteration Numbers")
    plt.ylabel("Accuracy")
    plt.savefig("2a-Training and Validation Accuracy-p=%d" %(i))


# # -------------------------------------------------- different shape

df_10_x = ppd.resize_df(train_x, 10)
df_10_y = ppd.resize_df(real_y, 10)

df_100_x = ppd.resize_df(train_x, 100)
df_100_y = ppd.resize_df(real_y, 100)

df_1000_x = ppd.resize_df(train_x, 1000)
df_1000_y = ppd.resize_df(real_y, 1000)

df_10000_x = ppd.mergeAndresize_df(train_x, dev_x, 10000)
df_10000_y = ppd.mergeAndresize_df(real_y, dev_y, 10000)

# start training
maxiter = 100
def train_shapes(train_x, real_y):
    K = Kk(train_x, train_x, 1)
    a = np.zeros(train_x.shape[0])
    for i in range(maxiter):
        for j in range(train_x.shape[0]):
            u = 0
            temp = np.multiply(K[j],a)
            u = (temp * real_y).sum()
            if u*real_y[j] <= 0:
                a[j] = a[j] + 1

result_time = []

start = time.perf_counter()
train_shapes(df_10_x, df_10_y)
end = time.perf_counter()
result_time.append(end-start)

start = time.perf_counter()
train_shapes(df_100_x, df_100_y)
end = time.perf_counter()
result_time.append(end-start)

start = time.perf_counter()
train_shapes(df_1000_x, df_1000_y)
end = time.perf_counter()
result_time.append(end-start)

start = time.perf_counter()
train_shapes(df_10000_x, df_10000_y)
end = time.perf_counter()
result_time.append(end-start)

print("****** when different shape ******")
print(result_time)
x = [10, 100, 1000, 10000]
plt.figure(10)
plt.plot(x, result_time)
plt.title("Runtime vs size N")
plt.xlabel("The size of the data")
plt.ylabel("Runtime(s)")
plt.savefig("2ac-Runtime vs size N")

# -------------------------------------------------- batch

maxiter = 100

K = Kk(train_x, train_x, 1)
print("****** batch ******")
figure_index = 10
m=0
for rate in [0.01, 0.1, 1, 10]:
    a = np.zeros(train_x.shape[0])
    a_temp = np.zeros(train_x.shape[0])
    AY_train = []
    AY_dev = []
    for i in range(maxiter):
        for j in range(train_x.shape[0]):
            u = 0
            temp = np.multiply(K[j],a)
            u = (temp * real_y).sum()
            if u*real_y[j] <= 0:
                a_temp[j] = a_temp[j] + rate
        # batch
        a = a_temp + np.zeros(train_x.shape[0])
        pre_train = np.matmul(K,a*real_y)
        AY_train.append(accy(pre_train, real_y))
        K_dev = Kk(dev_x, train_x, 1)
        pre_dev = np.matmul(K_dev,a*real_y)
        AY_dev.append(accy(pre_dev, dev_y))

    m = m+1
    plt.figure(figure_index+m)
    plt.plot(AY_train,label="Training Accuracy")
    plt.legend()
    plt.plot(AY_dev,label="Validation Accuracy")
    plt.legend()
    plt.title("Batch Training and Validation Accuracy-p=1-rate=%.2f" %rate)
    plt.xlabel("Iteration Numbers")
    plt.ylabel("Accuracy")
    plt.savefig("2b-Batch Training and Validation Accuracy-p=1-rate=%.2f.png" %rate)

# -------------------------------------------------- batch with different shapes

maxiter = 100
def train_shapes_batch(train_x, real_y):
    K = Kk(train_x, train_x, 1)
    a = np.zeros(train_x.shape[0])
    a_temp = np.zeros(train_x.shape[0])
    for i in range(maxiter):
        for j in range(train_x.shape[0]):
            u = 0
            temp = np.multiply(K[j],a)
            u = (temp * real_y).sum()
            if u*real_y[j] <= 0:
                a_temp[j] = a_temp[j] + 1
        # batch
        a = a_temp + np.zeros(train_x.shape[0])

result_time = []

start = time.perf_counter()
train_shapes_batch(df_10_x, df_10_y)
end = time.perf_counter()
result_time.append(end-start)

start = time.perf_counter()
train_shapes_batch(df_100_x, df_100_y)
end = time.perf_counter()
result_time.append(end-start)

start = time.perf_counter()
train_shapes_batch(df_1000_x, df_1000_y)
end = time.perf_counter()
result_time.append(end-start)

start = time.perf_counter()
train_shapes_batch(df_10000_x, df_10000_y)
end = time.perf_counter()
result_time.append(end-start)

print("****** batch with different shapes ******")
print(result_time)
x = [10, 100, 1000, 10000]
plt.figure(16)
plt.plot(x, result_time)
plt.title("Runtime vs size N (batch)")
plt.xlabel("The size of the data")
plt.ylabel("Runtime(s)")
plt.savefig("2bb-Runtime vs size N (batch)")