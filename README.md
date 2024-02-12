# Wild Forest Fire System Prediction using Neural Networks <br />
Axel Daniela Campero Vega <br />
Department of Computer Science <br />
Golisano College of Computing and Information Sciences <br />
Rochester Institute of Technology <br />
Rochester, NY 14623, USA  <br />
ac6887@g.rit.edu <br />

## Introduction <br />
The development of artificial intelligence and Machine Learning has promoted the studies and data analysis to forecast the occurrence of forest fires. Machine learning algorithms are applied, modified or created to understand the complex interplay of multiple variables associated with wildfires [1]. The most used algorithms are neural networks, gradient boosting, k-nearest neighbors. However, no "best method" has been established for analyzing this problem. The works reviewed adopt specific development models, require high-capacity computing stations, use of specialized software, and extensive user coding experience.

This project develops a forest fire seasonal prediction for Bolivia. Data from more than 20 years of fires provided by NASA's FIRMS-MODIS satellite [2] are used. Analysis of the seasonal behavior of this phenomenon is carried out using Long Short-term Memory (LSTM) [3], which helps us establish a seasonal model. All these factors are taken into account to make a forest fire probability prediction function based on the time of year. This prediction function is evaluated using RMSE by comparing results with the most recent real data. 

### I. DATA ACQUISITION <br />

NASA has implemented a system (FIRMS) [2], capable of monitoring forest fires globally and registering them in a freely accessible database throughout the world, which is an important source of data to study this problem in an attempt to find solutions that manage to mitigate it.
<p align="center">
 <img width="302" alt="image" src="https://github.com/AxelCamperoVega/Forest-Fire-Prediction-System/assets/43591999/8c8f9c76-e63b-4f2e-8025-3f8e85dd905f"> <br />
  <sup><sub> Data Source: Fire Information Resource Management System (FIRMS) – NASA </sub></sup>
</p>

Data acquisition is done by downloading FIRMS system files. The data has the format csv (comma-separated text files), and registers are also include Excel-type files format. FIRMS-Modis has data from 2000 to 2023 and they are arranged by country.
<p align="center">
<img width="452" alt="image" src="https://github.com/AxelCamperoVega/Forest-Fire-Prediction-System/assets/43591999/1415b0a8-f357-49a1-9365-5e8df02b6ffd"> <br />
</p>
 
#### Generation of initial files
In the "modis_bolivia" folder are the files that we will use with the following format: modis_year_bolivia.csv, where "year" varies from 2001 to 2023. The first step is to group the files from the year 2001 to 2020 in a single file which we will call "data_2001-2020.csv" and save it in the data folder. On the other hand, we must copy the “modis_2021-2022_bolivia.csv” file to the data folder with the name “datos_2021.csv”. To achieve that we will execute the program “unir_archivos.py”. Which allows us to automate this process with the help of the pandas python [23] library. To unite files a python program was developed: unir_archivos.py (Generation of initial files). The python program is in annex 3 of this work:
#### Data Filtering 
The following filters will be carried out in the pre-processing of the information:
•	Filtering of the attribute Type = 0 (Presumed vegetation fire)
•	Filtering of the Confidence attribute (0-100%), which gives the probability that it is a forest fire. For security, values greater than or equal to 90% will be filtered, with which we can ensure that we include only the data on forest fires.
•	Geographical area filtering: This parameter does not have a specific indication in the FIRMS records. However, the latitude and longitude coordinates are available, which will be used to perform this filtering (clusterization).
To filter the data a python program was developed [4]. The file filtered and joined from 2001 to 2020, has 228,349 records that we can consider forest fires with 90% certainty between the years indicated. To get an idea of its geographic distribution we plot the data. The data from the latitude and longitude columns of the fire historical data frame were used. Each annual file has on average more than 70,000 records throughout the year, so the total data considered exceeds one and a half million data to be considered and filtered. The files corresponding to each year are stored in a database for data mining and analysis that can be carried out later.

### II. MACHINE LEARNING - LSTM 
As part of the Deep Learning options there is Recurrent Neural Networks (RNN)  [5], but it is a network that presents a "vanishing gradient" problem since in each "back propagation" the gradients decrease and therefore the weights and bias leave to update, that is, the algorithm stops learning. Long Short Term Memory (LSTM) [3] is an artificial neural network, it works in a similar way to RNN without the gradient problem, in addition to having a good degree of learning with long datasets. For its implementation, Python resources and libraries will be used; Pandas, plotly, tensorflow, keras, keras_metrics, matplotlib, scikit-learn. 
 <p align="center">
 <img width="336" alt="image" src="https://github.com/AxelCamperoVega/Forest-Fire-Prediction-System/assets/43591999/0603c6d5-a1ad-4ef5-a4a7-4b887bb487d9"> <br />
  <sup><sub> Architecture</sub></sup>
</p>

##### A.	CODE DEVELOPMENT <br />
For the seasonal prediction model with LSTM, 3 programs were developed. <br />
•	unite_files.py (Generation of initial files)<br />
•	filter_data.py (Data Filtering) <br />
•	principal.py (Main Program) <br />
**unite_files.py -** Generation of initial files: We will use the files downloaded from FIRMS [2], detailed in section data acquisition, that have csv format: modis_year_bolivia.csv, where “year” varies from 2001 to 2022. We need to group the files from year 2001 to 2020 into a single file to use as training data. On the other hand, we will use the file “modis_2022_bolivia.csv” as test data to compare the predictions with the real data. We prefer to use a full year. The file joining program executes these actions and saves the data in folders that we will later use in LSTM.
<br />
**filter_data.py -** Data filtering: Once the two initial files with all the records have been obtained, the filtering will be done as follows: <br />
•	Filter the data for a confidence greater than 90% (Eliminate all data that has a values less than 90). This guarantees that the data to be used is reliable and represents forest fires.
<br />
•	Filter case “0” from the classification that represents the atmospheric phenomenon we want to study (forest fires) and model (eliminate all records that are not identified as case “0”). After running the program, new files will be generated with the data that will be used in LSTM. <br />
**principal.py - Main Program:** The main program executes the following actions:
<br />
•	Import libraries: necessary for the execution of the program.
<br />
•	Reading filtered data: “filtrado2001-2020.csv” and “filtrado2022.csv” and storing them in the fire_records and fires_2022 dataframes.
<br />
•	Format change: from acq_date (date) to date format: We convert the data in the acq_date column to a real date format. This makes it easier to further process the date data.
<br />
•	Initial plot of the data: We create a figure in which the coordinates where fires occurred from 2001 to 2020 are plotted. The data from the latitude and longitude columns of the historical_fires dataframe are used.
<p align="center">
 <img width="227" alt="image" src="https://github.com/AxelCamperoVega/Forest-Fire-Prediction-System/assets/43591999/c0d10123-22ed-40c1-8be0-9fd39f7c65d7"><br />
  <sup><sub> Initial data plotting!</sub></sup>
</p>

•	Count number of fires per day: Based on the data filtered and using the data filtered, the next thing is to count the number of fires per day both from the historical records and from the year 2021 and 2022. To achieve this, the program Groups records by date and counts how many records correspond to each date. Then add the heading number of fires to the column. This processed data is saved in other files.
<br />
•	Filling in dates without fires: From the processed data, it is observed that on several dates of the year, no forest fires occurred, which is why there is a discontinuity in the temporal data. These missing dates will be added and padded with 0 as part of your preparation for LSTM training. We change the index of the data which will become the acq_date column. We use the reindex function to combine our data, with the defined date range, and fill in the missing dates in our data with 0s.
<br />

 
#### Re-indexing data <br />

•	Plot number of fires by date: The program shows a graph of the number of fires by date based on the data obtained in the previous point, both historical data (2001-2020) and the year 2022 as a test. <br />

##### B.	LSTM implementation <br />
To use the LSTM algorithm, we need the data (number of fires per day) to be in the range 0 to 1 so it needs scaling. For the scaling to be the same, we must unite the training and test data into a single set that we will call the data set; we do this by concatenating both sets. Then we define the new scale (0,1) and transform the dataset to the new scale. Once the dataset is scaled, we separate it again to have two different sets: train and test.
<br />
**Converting train and test dataframes to numpy arrays**
Initially we define the historical prediction record parameter to be used by LSTM “look_back”, which will have the value of 30 which corresponds to approximately one month. Then we use the “create_dataset” function which will provide us with two numpy arrays based on the train and test data sets. We also do a reshape (adjust the number of rows and columns) to format the data so we can enter it into the model. trainX and testX will be the inputs of the LSTM network, while trainY and testY will be the outputs that it must learn and then predict. Each row of trainX and testX will be composed of 30 fire logs (30 days). Each row of trainY and testY will be composed of 1 record (1 day). <br />

This will allow the LSTM network to take each of the rows of trainX as input and each of the rows of trainY as output to learn to predict the number of fires for the next day (trainY). Later testX will serve as input data for testing the generated model. <br />
**Creating a numpy random seed to use in the LSTM network:**  <br />
Generates a seed of random numbers for the LSTM network. The topology of the neural network that will be used:
<p align="center">
 <img width="338" alt="image" src="https://github.com/AxelCamperoVega/Forest-Fire-Prediction-System/assets/43591999/e64b4993-a742-4eea-abd3-8f6b444657e7">
<br />
  <sup><sub> LSTM Topology</sub></sup>
</p>
 
We proceed to create an LSTM model, using the “Sequential” function that is part of the Keras (Tensorflow) library, since the data can be viewed as a temporal sequence. There will be 2 layers of 256 neurons. The first LSTM layer will receive a vector of the size of the lookback. The Dropout layer will serve to discard values based on their probability, allowing over fitting to be controlled. The second LSTM layer will receive the values resulting from the first layer. Finally, the results of the second layer pass to the dense layer, which is a single neuron with activation function “SELU”, which will deliver the final result of a single prediction at a time.
<br />

#### Neural network compilation:
Once the parameters of the neural network have been defined, we move on to generating the model, calculating the root mean square error (RMSE). The LSTM network is trained "epochs" times, with the numpy array list trainX (inputs), and the result list TrainY (outputs). Based on the constructed model, prediction was performed from the training and testing data, using trainPredict=model.predict(trainX) and testPredict=model.predict(testX). After the prediction is made, we de normalize the results that are on a scale from 0 to 1, using the “scaler.inverse” function. We then proceed to calculate the Root Median Square Error (RMSE), using the function “math.sqrt(mean_squared_error)”, for the training data and for the test data. The result is printed in the console. As a clarification, the RMSE value may vary slightly due to the random result of the seed.
<br />
Graphs generation; training and testing: For the graphs, we initially shift the results obtained on the time axis and use the python plotting functions.
<br />

•	The model is trained, using 15 "epochs", with the numpy array list trainX (inputs), and the result list TrainY (outputs). 
<br />
•	Time to run: 73.2 second 
<br />
•	Train Score: 65.0303 RMSE
<br />
•	Test Score: 64.0191 RMSE
<br />
•	Range: Maximum = 2413; Minimum = 0  (in the training)
<br />
•	Root Median Square Error (RMSE): 65.03
<br />
•	Normalized Root Median Square Error (NRMSE): RMSE / (Maximum – Minimum)    
<br />
•	NRMSE = 65.03/2413 = 0.027
<br />
•	Accuracy: 1-0.027 = 0.973  97.3%
<br />
For the part of the test, that is, the year 2022, we expand the scale to see only that year and we obtain the following graph:
<p align="center">
 <img width="429" alt="image" src="https://github.com/AxelCamperoVega/Forest-Fire-Prediction-System/assets/43591999/2bfff622-685b-4b85-9ca0-5113560fbcdc">
<br />
  <sup><sub>One year prediction (2022) compared to actual data
 </sub></sup>
</p>

Once the process is finished, what we have is the annual seasonal prediction of forest fires in Bolivia using LSTM with 20 years of training data and the prediction made for 2022 with the purpose of comparing the prediction data with the real ones. It must be taken into account that LSTM cannot make long-term predictions [6] that is, we will not be able to predict a complete year from the data of the current date, due to the mechanics (Long-Short) [3] that it handles. The data used to predict a single day is the previous 30 days, plus 20 years of training. In practice, this does not allow for long-term predictions. However, the prediction is better than with any other Machine Learning mechanism for long temporal sequences. 
 <br />
**References:** <br />
<sup><sub> [1] Predicting Forest Fires in Madagascar. Jessica Edwards, Manana Hakobyan, Alexander Lin and Christopher Golden. School of Engineering and Applied Sciences, Harvard University, Cambridge, MA, USA
https://projects.iq.harvard.edu/files/cs288/files/madagascar_fires.pdf <br />
[2] FIRMS (Fire Information Resource Management System – NASA) https://firms.modaps.eosdis.nasa.gov/country/  <br />
[3] Hochreiter, Sepp & Schmidhuber, Jürgen. (1997). Long Short-term Memory. Neural computation. 9. 1735-80. 10.1162/neco.1997.9.8.1735.  <br />
[4] KMeans Explained, Practical Guide To Data Clustering & How To In Python  https://spotintelligence.com/2023/08/23/kmeans/  <br />
[5] Superdatascience https://www.superdatascience.com/blogs/recurrent-neural-networks-rnn-lstm-practical-intuition  <br />
[6] Benjamin Lindemann et al. Procedia CIRP - Volume 99, 2021. A survey on long short-term memory networks for time series prediction https://www.sciencedirect.com/science/article/pii/S2212827121003796



