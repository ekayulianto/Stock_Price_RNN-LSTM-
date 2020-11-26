# RNN For Stock Price



# P1. Data Processing

## Import Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# mengambil nilai (numpy) pada kolom yang interest (open/kolom ke 1), karena untuk predict harus dalam bentuk numeri(numpy)
training_set = dataset_train.iloc[:, 1:2].values             

print(min(training_set))
print(max(training_set))

## Feature Scaling (Normalization)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

print(training_set_scaled[0:60])
## Creating a data structure with 60 timesteps and 1 output
# Note: RNN akan look up (harga saham) 60 hari sebelumnnya, dan berdasarkan trendnya akan memprediksi saham pada waktu T+1 (next day dari X_train)
# Note: X_train akan berisi 60 hari sebelum harga "sekarang" financial days, dan y_train next day dari X_train (untuk prediksinya (60 hari sebelumnya +1)
X_train = []
y_train = []

for i in range (60,1258):                                   ## start dari 60 pertama
    X_train.append(training_set_scaled[i-60:i, 0])          ## untuk i=60 (maka diambil 0:60) dari indeks 0 sampai 59, pada kolom '0'
    y_train.append(training_set_scaled[i,0])                ## untuk i=60 maka diambil indeks ke 60 pada kolom '0'
X_train, y_train = np.array(X_train), np.array(y_train)     #3 mengubahnya kedalam np array


## Note: pada X_train nya (row ke 0 merupakan data pada hari pd index ke 0-59, dan pada row ke=1 merupakan hari pd indeks ke 1-60, dst)

## Reshaping
### Note: kalo mau menambahkan dimensi pd numpy harus melakukan reshaping
### pada np.reshape('data yg mau direshape', 'Bentukan baru')
### dimana bentukan baru dalam hal ini jadi 3D (batch_size = jumlah row, timestamp=jumlah kolom, 'indicator yang diguanakan')
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# P2. Building RNN Model (LSTM)
## Import Keras Library
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

## Inisialising 
regressor = Seqeuntial()                        ## namanya regressor karena mengguna regression(continue value, bukan klasifikasi)



# Prediction and visualization the reuslts