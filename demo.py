# In[1]:
from idlelib import history
import keras.backend as K
from keras.layers import Dense, Lambda, Dot, Activation, Concatenate
from keras.layers import Conv1D, MaxPooling1D, merge
import keras
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, Bidirectional, Multiply, LSTM
from keras.layers.core import *
from keras.models import *
from sklearn.metrics import mean_absolute_error
from keras import backend as K
from tensorflow.python.keras.layers import CuDNNLSTM, GlobalAveragePooling1D
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import matplotlib as mpl
import numpy as np
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = 'Times New Roman'
SINGLE_ATTENTION_VECTOR = False
import os

# 使用第一张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data = pd.read_csv('roomnewxia.csv')

data.drop('id', axis=1, inplace=True)
data = data[['ts', 'HUMBtn', 'TEMPBtn', 'TEMP2', 'HUM2', 'CO22', 'HUM3', 'TEMP3']]
# data = data[['ts', 'HUMBtn', 'TEMPBtn',  'TEMP1', 'HUM1', 'CO21', 'TEMP2', 'HUM2', 'CO22', 'HUM3', 'TEMP3']]

data.isnull().sum()
data.dropna(inplace=True)
data.duplicated().sum()


data["ts"] = data["ts"].map(lambda x: x[:-3])
new_data = data.groupby('ts').mean().reset_index()
new_data["ts"] = pd.to_datetime(new_data["ts"])
new_data.sort_values('ts', inplace=True)

flag1 = new_data.pop("HUM2")
flag2 = new_data.pop("CO22")
flag3 = new_data.pop("TEMP2")

new_data.insert(1, "HUM2", flag1)
new_data.insert(2, "CO22", flag2)
new_data.insert(3, "TEMP2", flag3)


plt.style.use('ggplot')

# In[15]:

values = new_data.set_index('ts').values
# specify columns to plot
groups = [0, 1, 2, 3, 4, 5, 6]
i = 1
# plot each column
plt.figure(figsize=(20, 20),facecolor='white')
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(new_data.set_index('ts').columns[group], y=0.8, loc='right')
    i += 1
plt.show()
# import matplotlib.pyplot as plt
#
# plt.style.use('default')
# plt.rcParams['figure.facecolor'] = 'white'
#
# values = new_data.set_index('ts').values
# groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# i = 1
#
# fig, axs = plt.subplots(len(groups), 1, figsize=(20, 20))
# for group in groups:
#     axs[i-1].plot(values[:, group])
#     axs[i-1].set_title(new_data.set_index('ts').columns[group], y=0.8, loc='right')
#     i += 1
#
# plt.show()


kf = KalmanFilter(n_dim_obs=1, n_dim_state=1, initial_state_mean=0, initial_state_covariance=1,
                  transition_matrices=[1], observation_matrices=[1])

# 变量选择
col_names = data.columns[1:]

# In[16]:

from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

# In[17]:

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# In[19]:

reframed

# drop columns we don't want to predict
reframed.drop(reframed.columns[[10, 11, 12, 13]], axis=1, inplace=True)
reframed.head()

# In[21]:
epoch = 20
batch_size = 64
# split into train and test sets
values = reframed.values
n_train_minuute = 40 * 24 * 60
# train = values[:n_train_minuute, :]
# test = values[n_train_minuute:, :]
# # split into input and outputs
# train_X, train_y = train[:, :-3], train[:, -3:]
# test_X, test_y = test[:, :-3], test[:, -3:]
# 计算要留出的测试集大小
test_size = int(len(values) * 0.1)

# 将后面的test_size条记录作为测试集
train, test = values[:-test_size, :], values[-test_size:, :]

# 分割输入和输出
train_X, train_y = train[:, :-3], train[:, -3:]
test_X, test_y = test[:, :-3], test[:, -3:]


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


## LSTM


# # design network
model = keras.models.Sequential()
model.add(keras.layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(keras.layers.Activation('relu'))
# model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(3))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=epoch, batch_size=72, validation_data=(test_X, test_y), verbose=1,
                    shuffle=False)
# 72

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

yhat1 = model.predict(test_X)

test_X1 = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat1 = keras.backend.concatenate((yhat1, test_X1[:, 3:]), axis=1)
inv_yhat1 = scaler.inverse_transform(inv_yhat1)
inv_yhat1 = inv_yhat1[:, :3]
# invert scaling for actual
test_y1 = test_y.reshape((len(test_y), 3))
inv_y1 = keras.backend.concatenate((test_y1, test_X1[:, 3:]), axis=1)
inv_y1 = scaler.inverse_transform(inv_y1)
inv_y1 = inv_y1[:, :3]

# calculate RMSE
print('Test RMSE: %.3f' % np.sqrt(mean_squared_error(inv_y1, inv_yhat1)))
# calculate Acc
print('Test Acc: %.3f' % r2_score(inv_y1, inv_yhat1))
#

# extract CO2, Temperature and Humidity columns from predicted values
inv_yhat1_co2 = inv_yhat1[:, 0]
inv_yhat1_temp = inv_yhat1[:, 1]
inv_yhat1_humid = inv_yhat1[:, 2]

# extract CO2, Temperature and Humidity columns from actual values
inv_y1_co2 = inv_y1[:, 0]
inv_y1_temp = inv_y1[:, 1]
inv_y1_humid = inv_y1[:, 2]

# calculate RMSE for each variable
rmse_co2 = np.sqrt(mean_squared_error(inv_y1_co2, inv_yhat1_co2))
rmse_temp = np.sqrt(mean_squared_error(inv_y1_temp, inv_yhat1_temp))
rmse_humid = np.sqrt(mean_squared_error(inv_y1_humid, inv_yhat1_humid))

# print the RMSE for each variable
print('Test RMSE for CO2: %.3f' % rmse_co2)
print('Test RMSE for Temperature: %.3f' % rmse_temp)
print('Test RMSE for Humidity: %.3f' % rmse_humid)

# calculate the R-squared for each variable
r2_co2 = r2_score(inv_y1_co2, inv_yhat1_co2)
r2_temp = r2_score(inv_y1_temp, inv_yhat1_temp)
r2_humid = r2_score(inv_y1_humid, inv_yhat1_humid)

# print the R-squared for each variable
print('Test R2 score for CO2: %.3f' % r2_co2)
print('Test R2 score for Temperature: %.3f' % r2_temp)
print('Test R2 score for Humidity: %.3f' % r2_humid)

# calculate the MAE for each variable
mae_co2 = mean_absolute_error(inv_y1_co2, inv_yhat1_co2)
mae_temp = mean_absolute_error(inv_y1_temp, inv_yhat1_temp)
mae_humid = mean_absolute_error(inv_y1_humid, inv_yhat1_humid)

# print the MAE for each variable
print('Test MAE for CO2: %.3f' % mae_co2)
print('Test MAE for Temperature: %.3f' % mae_temp)
print('Test MAE for Humidity: %.3f' % mae_humid)

from sklearn.metrics import mean_absolute_error

# calculate MAE
mae = mean_absolute_error(inv_y1, inv_yhat1)
print('MAE: %.3f' % mae)
import matplotlib.pyplot as plt

# calculate error for each variable
error_co2 = inv_y1_co2 - inv_yhat1_co2
error_temp = inv_y1_temp - inv_yhat1_temp
error_humid = inv_y1_humid - inv_yhat1_humid

# plot error points for CO2
plt.scatter(range(len(error_co2)), error_co2, label='CO2 Error')

# plot error points for Temperature
plt.scatter(range(len(error_temp)), error_temp, label='Temperature Error')

# plot error points for Humidity
plt.scatter(range(len(error_humid)), error_humid, label='Humidity Error')

plt.axhline(y=mae, color='r', linestyle='-', label='MAE')
plt.xlabel('Sample')
plt.ylabel('Error')
plt.legend()
plt.show()

# pd.concat([pd.DataFrame(inv_yhat1[:, 0],columns=["predict"]), pd.DataFrame(inv_y1[:, 0],columns=["true"])], axis=1).plot(figsize=(20, 6))
# plt.title("HUM2", fontsize=18)
# plt.show()
#
# pd.concat([pd.DataFrame(inv_yhat1[:, 1],columns=["predict"]), pd.DataFrame(inv_y1[:, 1],columns=["true"])], axis=1).plot(figsize=(20, 6))
# plt.title("CO22", fontsize=18)
# plt.show()
#
# pd.concat([pd.DataFrame(inv_yhat1[:, 2],columns=["predict"]), pd.DataFrame(inv_y1[:, 2],columns=["true"])], axis=1).plot(figsize=(20, 6))
# plt.title("TEMP2", fontsize=18)
# plt.show()

# RNN

# # 创建模型
model = keras.models.Sequential()
# RNN神经网络
# 隐藏层100
model.add(keras.layers.SimpleRNN(units=100, return_sequences=True))
model.add(keras.layers.Activation('relu'))
# Dropout层用于防止过拟合
# model.add(keras.layers.Dropout(0.1))
# 隐藏层100
model.add(keras.layers.SimpleRNN(units=100))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(3))
model.compile(loss='mse', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=epoch, batch_size=batch_size, validation_data=(test_X, test_y), verbose=1)

# In[31]:


# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plt.show()


yhat3 = model.predict(test_X)

# In[33]:

test_X3 = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast

inv_yhat3 = keras.backend.concatenate((yhat3, test_X3[:, 3:]), axis=1)
inv_yhat3 = scaler.inverse_transform(inv_yhat3)
inv_yhat3 = inv_yhat3[:,:3]
# invert scaling for actual
test_y3 = test_y.reshape((len(test_y), 3))
inv_y3 = keras.backend.concatenate((test_y3, test_X3[:, 3:]), axis=1)
inv_y3 = scaler.inverse_transform(inv_y3)
inv_y3 = inv_y3[:,:3]

# In[34]:


# calculate RMSE
print('Test RMSE: %.3f' % np.sqrt(mean_squared_error(inv_y3, inv_yhat3)))
# calculate Acc
print('Test Acc: %.3f' % r2_score(inv_y3, inv_yhat3))
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# calculate RMSE for temperature
rmse_temp = np.sqrt(mean_squared_error(inv_y3[:, 0], inv_yhat3[:, 0]))
print('Temperature RMSE: %.3f' % rmse_temp)

# calculate RMSE for humidity
rmse_humid = np.sqrt(mean_squared_error(inv_y3[:, 1], inv_yhat3[:, 1]))
print('Humidity RMSE: %.3f' % rmse_humid)

# calculate RMSE for CO2
rmse_co2 = np.sqrt(mean_squared_error(inv_y3[:, 2], inv_yhat3[:, 2]))
print('CO2 RMSE: %.3f' % rmse_co2)

# calculate r2 score for temperature
r2_temp = r2_score(inv_y3[:, 0], inv_yhat3[:, 0])
print('Temperature r2: %.3f' % r2_temp)

# calculate r2 score for humidity
r2_humid = r2_score(inv_y3[:, 1], inv_yhat3[:, 1])
print('Humidity r2: %.3f' % r2_humid)

# calculate r2 score for CO2
r2_co2 = r2_score(inv_y3[:, 2], inv_yhat3[:, 2])
print('CO2 r2: %.3f' % r2_co2)

# calculate MAE for temperature
mae_temp = mean_absolute_error(inv_y3[:, 0], inv_yhat3[:, 0])
print('Temperature MAE: %.3f' % mae_temp)

# calculate MAE for humidity
mae_humid = mean_absolute_error(inv_y3[:, 1], inv_yhat3[:, 1])
print('Humidity MAE: %.3f' % mae_humid)

# calculate MAE for CO2
mae_co2 = mean_absolute_error(inv_y3[:, 2], inv_yhat3[:, 2])
print('CO2 MAE: %.3f' % mae_co2)




# pd.concat([pd.DataFrame(inv_yhat3[:, 0],columns=["predict"]), pd.DataFrame(inv_y3[:, 0],columns=["true"])], axis=1).plot(figsize=(20, 6))
# plt.title("HUM2", fontsize=18)
# plt.show()
#
# pd.concat([pd.DataFrame(inv_yhat3[:, 1],columns=["predict"]), pd.DataFrame(inv_y3[:, 1],columns=["true"])], axis=1).plot(figsize=(20, 6))
# plt.title("CO22", fontsize=18)
# plt.show()
#
# pd.concat([pd.DataFrame(inv_yhat3[:, 2],columns=["predict"]), pd.DataFrame(inv_y3[:, 2],columns=["true"])], axis=1).plot(figsize=(20, 6))
# plt.title("TEMP2", fontsize=18)
# plt.show()
#

class Attention(keras.layers.Layer):

    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.w = self.add_weight(name="att_weights", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.w) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a

        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# In[ ]:
filters = 5000
kernel_size = 1
poolkernel_size = 2
Conv1D_strides_size=4
MaxPooling1D_strides_size=1
# ___________________________________________________________________________________
# model = keras.models.Sequential()
#
# model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=4))
# # model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=4))
# # model.add(MaxPooling1D(pool_size=poolkernel_size,padding='same',strides=1))
# model.add(keras.layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
# # model.add(Attention())
# # model.add(keras.layers.GRU(50, input_shape=(train_X.shape[1], train_X.shape[2])))
#
# # 用于将输入层的数据压成一维的数据，一般用再卷积层和全连接层之间
# # model.add(Dropout(0.25))
# model.add(keras.layers.Dense(3))
# model.add(Activation('sigmoid'))
# model.compile(loss='mae', optimizer='adam')
# # fit network
# history = model.fit(train_X, train_y, epochs=epoch, batch_size=batch_size, validation_data=(test_X, test_y), verbose=1, shuffle=False)

# ___________________________________________________________________________________
model = keras.models.Sequential()

model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=Conv1D_strides_size))
model.add(MaxPooling1D(pool_size=poolkernel_size, padding='same', strides=1))
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(Attention())
model.add(Dense(train_y.shape[1], activation='sigmoid'))

model.compile(loss='mae', optimizer='adam')

history = model.fit(train_X, train_y, epochs=epoch, batch_size=batch_size, validation_data=(test_X, test_y), verbose=1,
                    shuffle=False)

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

yhat2 = model.predict(test_X)

test_X2 = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat2 = keras.backend.concatenate((yhat2, test_X2[:, 3:]), axis=1)
inv_yhat2 = scaler.inverse_transform(inv_yhat2)
inv_yhat2 = inv_yhat2[:, :3]
# invert scaling for actual
test_y2 = test_y.reshape((len(test_y), 3))
inv_y2 = keras.backend.concatenate((test_y2, test_X2[:, 3:]), axis=1)
inv_y2 = scaler.inverse_transform(inv_y2)
inv_y2 = inv_y2[:, :3]

# calculate RMSE
print('Test RMSE: %.3f' % np.sqrt(mean_squared_error(inv_y2, inv_yhat2)))
# calculate Acc
print('Test Acc: %.3f' % r2_score(inv_y2, inv_yhat2))

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# calculate RMSE for temperature
rmse_temp = np.sqrt(mean_squared_error(inv_y2[:, 0], inv_yhat2[:, 0]))
print('Temperature RMSE: %.3f' % rmse_temp)

# calculate RMSE for humidity
rmse_humid = np.sqrt(mean_squared_error(inv_y2[:, 1], inv_yhat2[:, 1]))
print('Humidity RMSE: %.3f' % rmse_humid)

# calculate RMSE for CO2
rmse_co2 = np.sqrt(mean_squared_error(inv_y2[:, 2], inv_yhat2[:, 2]))
print('CO2 RMSE: %.3f' % rmse_co2)

# calculate r2 score for temperature
r2_temp = r2_score(inv_y2[:, 0], inv_yhat2[:, 0])
print('Temperature r2: %.3f' % r2_temp)

# calculate r2 score for humidity
r2_humid = r2_score(inv_y2[:, 1], inv_yhat2[:, 1])
print('Humidity r2: %.3f' % r2_humid)

# calculate r2 score for CO2
r2_co2 = r2_score(inv_y2[:, 2], inv_yhat2[:, 2])
print('CO2 r2: %.3f' % r2_co2)

# calculate MAE for temperature
mae_temp = mean_absolute_error(inv_y2[:, 0], inv_yhat2[:, 0])
print('Temperature MAE: %.3f' % mae_temp)

# calculate MAE for humidity
mae_humid = mean_absolute_error(inv_y2[:, 1], inv_yhat2[:, 1])
print('Humidity MAE: %.3f' % mae_humid)

# calculate MAE for CO2
mae_co2 = mean_absolute_error(inv_y2[:, 2], inv_yhat2[:, 2])
print('CO2 MAE: %.3f' % mae_co2)


# pd.concat([pd.DataFrame(inv_yhat2[:, 0],columns=["predict"]), pd.DataFrame(inv_y2[:, 0],columns=["true"])], axis=1).plot(figsize=(20, 6))
# plt.title("HUM2", fontsize=18)
# plt.show()
#
# pd.concat([pd.DataFrame(inv_yhat2[:, 1],columns=["predict"]), pd.DataFrame(inv_y2[:, 1],columns=["true"])], axis=1).plot(figsize=(20, 6))
# plt.title("CO22", fontsize=18)
# plt.show()
#
# pd.concat([pd.DataFrame(inv_yhat2[:, 2],columns=["predict"]), pd.DataFrame(inv_y2[:, 2],columns=["true"])], axis=1).plot(figsize=(20, 6))
# plt.title("TEMP2", fontsize=18)
# plt.show()
# create subplots
import matplotlib.pyplot as plt

# plot temperature predictions from all models
temp_df = pd.concat([pd.DataFrame(inv_yhat1[:, 2],columns=["Model 1"]),
                     pd.DataFrame(inv_yhat3[:, 2],columns=["Model 2"]),
                     pd.DataFrame(inv_yhat2[:, 2],columns=["Model 3"]),
                     pd.DataFrame(inv_y1[:, 2],columns=["True"])], axis=1)
temp_df.plot(figsize=(20, 6))
plt.title("Temperature Predictions", fontsize=18)
plt.show()

# plot humidity predictions from all models
humid_df = pd.concat([pd.DataFrame(inv_yhat1[:, 0],columns=["Model 1"]),
                      pd.DataFrame(inv_yhat3[:, 0],columns=["Model 2"]),
                      pd.DataFrame(inv_yhat2[:, 0],columns=["Model 3"]),
                      pd.DataFrame(inv_y1[:, 0],columns=["True"])], axis=1)
humid_df.plot(figsize=(20, 6))
plt.title("Humidity Predictions", fontsize=18)
plt.show()

# plot CO2 predictions from all models
co2_df = pd.concat([pd.DataFrame(inv_yhat1[:, 1],columns=["Model 1"]),
                    pd.DataFrame(inv_yhat3[:, 1],columns=["Model 2"]),
                    pd.DataFrame(inv_yhat2[:, 1],columns=["Model 3"]),
                    pd.DataFrame(inv_y1[:, 1],columns=["True"])], axis=1)
co2_df.plot(figsize=(20, 6))
plt.title("CO2 Predictions", fontsize=18)
plt.show()
# 误差
# calculate mean absolute errors
mae1 = np.mean(np.abs(inv_y1 - inv_yhat1), axis=0)
mae2 = np.mean(np.abs(inv_y1 - inv_yhat2), axis=0)
mae3 = np.mean(np.abs(inv_y1 - inv_yhat3), axis=0)
# plot mean absolute errors as scatter plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
ax1.scatter(x=co2_df["True"], y=inv_yhat1[:, 1], label="Model 1", s=16)
ax1.scatter(x=co2_df["True"], y=inv_yhat2[:, 1], label="Model 2", s=16)
ax1.scatter(x=co2_df["True"], y=inv_yhat3[:, 1], label="Model 3", s=16)
ax1.plot([min(co2_df["True"]), max(co2_df["True"])], [min(co2_df["True"]), max(co2_df["True"])], "k--", label="Perfect Prediction")
ax1.set_xlabel("True CO2/(PPM)", fontsize=16)
ax1.set_ylabel("Predicted CO2/(PPM)", fontsize=16)
# ax1.set_title(f"CO2 Predictions (MAE: Model 1 = {mae1[1]:.2f}, Model 2 = {mae2[1]:.2f}, Model 3 = {mae3[1]:.2f})", fontsize=16)
ax1.legend(fontsize=16)
ax2.scatter(x=humid_df["True"], y=inv_yhat1[:, 0], label="Model 1", s=16)
ax2.scatter(x=humid_df["True"], y=inv_yhat2[:, 0], label="Model 2", s=16)
ax2.scatter(x=humid_df["True"], y=inv_yhat3[:, 0], label="Model 3", s=16)
ax2.plot([min(humid_df["True"]), max(humid_df["True"])], [min(humid_df["True"]), max(humid_df["True"])], "k--", label="Perfect Prediction")
ax2.set_xlabel("True Humidity/(%)", fontsize=16)
ax2.set_ylabel("Predicted Humidity/(%)", fontsize=16)
# ax2.set_title(f"Humidity Predictions (MAE: Model 1 = {mae1[0]:.2f}, Model 2 = {mae2[0]:.2f}, Model 3 = {mae3[0]:.2f})", fontsize=16)
ax2.legend(fontsize=16)
ax3.scatter(x=temp_df["True"], y=inv_yhat1[:, 2], label="Model 1", s=16)
ax3.scatter(x=temp_df["True"], y=inv_yhat2[:, 2], label="Model 2", s=16)
ax3.scatter(x=temp_df["True"], y=inv_yhat3[:, 2], label="Model 3", s=16)
ax3.plot([min(temp_df["True"]), max(temp_df["True"])], [min(temp_df["True"]), max(temp_df["True"])], "k--", label="Perfect Prediction")
ax3.set_xlabel("True Temperature/(℃)", fontsize=16)
ax3.set_ylabel("Predicted Temperature/(℃)", fontsize=16)
# ax3.set_title(f"Temperature Predictions (MAE: Model 1 = {mae1[2]:.2f}, Model 2 = {mae2[2]:.2f}, Model 3 = {mae3[2]:.2f})", fontsize=16)
ax3.legend(fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax3.tick_params(axis='both', which='major', labelsize=12)
ax1.legend(fontsize=16, labelcolor='black')
ax2.legend(fontsize=16, labelcolor='black')
ax3.legend(fontsize=16, labelcolor='black')

plt.show()


# create first figure
fig1 = plt.figure(figsize=(20, 6))
ax1 = fig1.add_subplot(111)
ax1.scatter(x=co2_df["True"], y=inv_yhat1[:, 1], label="Model 1", s=16)
ax1.scatter(x=co2_df["True"], y=inv_yhat2[:, 1], label="Model 2", s=16)
ax1.scatter(x=co2_df["True"], y=inv_yhat3[:, 1], label="Model 3", s=16)
ax1.plot([min(co2_df["True"]), max(co2_df["True"])], [min(co2_df["True"]), max(co2_df["True"])], "k--", label="Perfect Prediction")
ax1.set_xlabel("True CO2/(ppm)")
ax1.set_ylabel("Predicted CO2/(ppm)")
ax1.set_title(f"CO2 Predictions (MAE: Model 1 = {mae1[1]:.2f}, Model 2 = {mae2[1]:.2f}, Model 3 = {mae3[1]:.2f})")
ax1.legend(fontsize=12)
plt.show()

# create second figure
fig2 = plt.figure(figsize=(20, 6))
ax2 = fig2.add_subplot(111)
ax2.scatter(x=humid_df["True"], y=inv_yhat1[:, 0], label="Model 1", s=16)
ax2.scatter(x=humid_df["True"], y=inv_yhat2[:, 0], label="Model 2", s=16)
ax2.scatter(x=humid_df["True"], y=inv_yhat3[:, 0], label="Model 3", s=16)
ax2.plot([min(humid_df["True"]), max(humid_df["True"])], [min(humid_df["True"]), max(humid_df["True"])], "k--", label="Perfect Prediction")
ax2.set_xlabel("True Humidity/(%)")
ax2.set_ylabel("Predicted Humidity/(%)")
ax2.set_title(f"Humidity Predictions (MAE: Model 1 = {mae1[0]:.2f}, Model 2 = {mae2[0]:.2f}, Model 3 = {mae3[0]:.2f})")
ax2.legend(fontsize=12)
plt.show()

# create third figure
fig3 = plt.figure(figsize=(20, 6))
ax3 = fig3.add_subplot(111)
ax3.scatter(x=temp_df["True"], y=inv_yhat1[:, 2], label="Model 1", s=16)
ax3.scatter(x=temp_df["True"], y=inv_yhat2[:, 2], label="Model 2", s=16)
ax3.scatter(x=temp_df["True"], y=inv_yhat3[:, 2], label="Model 3", s=16)
ax3.plot([min(temp_df["True"]), max(temp_df["True"])], [min(temp_df["True"]), max(temp_df["True"])], "k--", label="Perfect Prediction")
ax3.set_xlabel("True Temperature/(℃)")
ax3.set_ylabel("Predicted Temperature/(℃)")
ax3.set_title(f"Temperature Predictions (MAE: Model 1 = {mae1[2]:.2f}, Model 2 = {mae2[2]:.2f}, Model 3 = {mae3[2]:.2f})")
ax3.legend(fontsize=12)
plt.show()

