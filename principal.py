# Fires Forest Prediction - Main Program
# Axel Campero Vega

# Import of libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import statistics
import keras_metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from tensorflow.keras import utils
utils.to_categorical

from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Function that creates the dataset for training and testing
def create_dataset(dataset, look_back):
    dataX, dataY = [], []                                # dataX = Inputs, dataY = Outputs
    for i in range(len(dataset)-look_back):              
        dataX.append(dataset[i:(i+look_back), 0])        # Add to dataX dataset[0:29], then dataset[1:30] ...
        dataY.append(dataset[i + look_back, 0])          # Add to dataY dataset[30], then dataset[31] ...
    return np.array(dataX), np.array(dataY)              # Return to numpy type

# Read previously filtered fire data
fires_records = pd.read_csv('./data/filtrado_2001-2020.csv')
fires_2021 = pd.read_csv('./data/filtrado_2021.csv')

# Change data format of acq_time column
fires_records['acq_date'] = pd.to_datetime(fires_records['acq_date'])
fires_2021['acq_date'] = pd.to_datetime(fires_2021['acq_date'])

# Graph fire coordinates 2001-2020
plt.figure(figsize=(5, 5))
plt.scatter(fires_records['longitude'], fires_records['latitude'], color='red')
plt.title('Wild Forest Fire in Bolivia 2001-2020')
plt.show()

# Count the number of fires that occurred each day from 2001 to 2020
fires_records = fires_records.groupby(['acq_date']).size().reset_index(name='number_of_fires')
fires_records.to_csv('./data/incendios_2001-2020.csv', index=False)

# Contar el Number of Wild Fires que hubo cada dia en el año 2021
fires_2021 = fires_2021.groupby(['acq_date']).size().reset_index(name='number_of_fires')
fires_2021.to_csv('./data/fires_2021.csv', index=False)

# Graph Number of fires by date, from 2001 to 2020
plt.figure(figsize=(10, 5))
plt.plot(fires_records['acq_date'], fires_records['number_of_fires'])
plt.xlabel('Date')
plt.ylabel('Number of Wild Fires')
plt.title('Wild Fires 2001-2020')
plt.show()

# Graph Number of fires by date in 2021
plt.figure(figsize=(10, 5))
plt.plot(fires_2021['acq_date'], fires_2021['number_of_fires'])
plt.xlabel('Date')
plt.ylabel('Number of Wild Fires')
plt.title('Wild Fires 2021')
plt.show()

# Data frame Training
df_training = fires_records                           # Save the data in a new dataframe
df_training.set_index(['acq_date'], inplace=True)     # The dataframe index becomes the acq_date column
idx = pd.date_range('2001-1-1', '2020-12-31')         # Create date range from 2001 to 2020
df_training = df_training.reindex(idx, fill_value=0)  # Add rows with missing dates in the range
dataset_training = df_training.values                 # Get the values and store them as numpy type
dataset_training = dataset_training.astype('float32') # I convert the number of fires to float

# Data frame Test
df_prueba = fires_2021                                      # Save the data in a new dataframe
df_prueba.set_index(['acq_date'], inplace=True)             # The dataframe index becomes the acq_date column
idx = pd.date_range('2021-1-1', '2021-12-31')               # Create date range from 2001 to 2020
df_prueba = df_prueba.reindex(idx, fill_value=0)            # Add rows with missing dates in the range
dataset_prueba = df_prueba.values                           # Get the values and store them as numpy type
dataset_prueba = dataset_prueba.astype('float32')           # Convert the number of fires to float

# Data scaling
dataset = np.concatenate((dataset_training,dataset_prueba))# Concatenate training and test data into a single df
scaler = MinMaxScaler(feature_range=(0, 1))                # Define the new scale that the values will have
dataset = scaler.fit_transform(dataset)                    # Perform scaling in the new range 0 to 1

# Define training and test data
train = dataset[0:len(dataset_training)]
test = dataset[len(dataset_training):len(dataset)]

# Convert data to numpy arrays
look_back = 30
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape the data to prepare it for the neural network
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Random numpy creation
np.random.seed(100)

# Create the LSTM model
model = Sequential()

# 1st layer of 256 entries, layer of 256 neurons
model.add(LSTM(256, input_shape=(1, look_back), return_sequences=True))

# Avoid overfitting
model.add(Dropout(0.9))

# 2nd layer of 256 entries
model.add(LSTM(256, input_shape=(1, int(look_back/3))))

# Dense layer, the layer that receives everything.
model.add(Dense(1))

# Activation function
model.add(Activation('selu'))

# Compile the neural network
model.compile(loss='mean_squared_error',optimizer='rmsprop', metrics=['accuracy', 'mae'])

# Train the neural network with the "epochs" dataset times.
model.fit(trainX, trainY, epochs=15, batch_size=15, verbose=0)

# Result of predictions on training data and test data
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Scale values to initial range
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Get RMSE statistics
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.4f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.4f RMSE' % (testScore))

# Calculate the average and NRMSE (normalized)

trainAvgY = statistics.mean(trainY[0])
print('trainAvg: %.4f TrainAVG' % (trainAvgY))

trainAvgX = statistics.mean(fires_records['number_of_fires'])
print('trainAvgX: %.4f TrainAVGX' % (trainAvgX))

# Data offset for trainpredict graph
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# Data offset for testpredict plot
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict

# Graph data, real training and testing
plt.figure(figsize=(10, 5))
plt.plot(scaler.inverse_transform(dataset), label='Real')
plt.plot(trainPredictPlot, label='Training')
plt.plot(testPredictPlot, color='red', label='Test')
plt.title('Fire Forest 2001-2020')
plt.xlabel('Days')
plt.ylabel('Number of Fires')
plt.legend()
plt.show()

# Graph only from the year 2021
# These graph details interest me in the red ones testPredictPlot
testPredictPlot = np.empty_like(test)
testPredictPlot[:, :] = np.nan
testPredictPlot[look_back:len(testPredict)+look_back, :] = testPredict
plt.figure(figsize=(10, 5))
plt.plot(scaler.inverse_transform(test), label='Real')
plt.plot(testPredictPlot, color='red', label='Test')
plt.title('Fires 2021')
plt.xlabel('Days')
plt.ylabel('Number of Fires')
plt.legend()
plt.show()
# print (testPredictPlot) This prints the output of the last year in values
