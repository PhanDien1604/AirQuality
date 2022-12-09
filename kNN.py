import numpy as np
import pandas as pd
import math
import mysql.connector as mysql
from mysql.connector import Error


def connect():
    conn = None
    try:
        conn = mysql.connect(host='localhost',database='air_quality',user='root',password='anhluong123')
        if conn.is_connected:
            return conn
    except Error as e:
        print('Error',e)
    return conn

def train_test_split(features, label_name, test_size, random_state):
    shuffle_feature_df = features.sample(frac = 1,random_state=random_state)
    test_size = int(test_size*len(features))
    X_train = shuffle_feature_df[test_size:]
    X_test = shuffle_feature_df[:test_size]
    y_train = X_train[label_name]
    y_test = X_test[label_name]
    X_train = X_train.drop(columns=[label_name])
    X_test = X_test.drop(columns=[label_name])
    return X_train, X_test, y_train, y_test
class kNN:
    def __init__(self, k):
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        y_predict = []
        for i in range(len(x_test)):
            distance = []
            for j in range(len(x_train)): 
                d = np.sqrt(np.sum((x_test[i,:] - x_train[j,:])**2))
                distance.append((d,y_train[j]))
            distance = sorted(distance)
            neighbour = []
            for i in range(self.k):
                neighbour.append(distance[i][1])
            y_predict.append(np.mean(neighbour))
        return y_predict

conn = connect()
print(conn)
air_quality_data = pd.read_sql_query('SELECT * FROM air_quality.air_quality_raw;',conn)

x = air_quality_data.drop(columns=['Xylene', 'City', 'Date', 'AQI_Bucket']).dropna()
print(x.shape)
# # divide dataset into train (80%) and test (20%) sets
x_train, x_test, y_train, y_test = train_test_split(x, 'AQI',test_size = 0.2,random_state = 11)
print(x_train.shape)
x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values

knn = kNN(k = 7)

# fit the data
knn.fit(x_train, y_train)

# predict x_test
y_pred = knn.predict(x_test)

result = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(result)

# def insert_book(actural, predict):
#     query = "INSERT INTO `air_quality`.`result` (`actural`, `predict`) VALUES (%s,%s);"
#     args = (actural, predict)
 
#     try:
 
#         conn = connect()
 
#         cursor = conn.cursor()
#         cursor.execute(query, args)

#         conn.commit()
#     except Error as error:
#         print(error)
 
#     finally:
#         # Đóng kết nối
#         cursor.close()
#         conn.close()