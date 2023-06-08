# final-proj-benpiekarz-avonderg-kku2
CSCI1470 (Deep Learning) Final Project

Link to DevPost: https://devpost.com/software/time-series-forecasting-with-rnns?ref_content=user-portfolio&ref_feature=in_progress

<p align="center">
  <img src="https://d112y698adiu2z.cloudfront.net/photos/production/software_photos/002/477/426/datas/original.png" width="700" title="DL Day poster">
</p> 


# Reflection: Time Series Forecasting with LSTM and GRU by Kleo Ku, Alexandra von der Goltz, Ben Piekarz
Overall, we thoroughly enjoyed applying some of our learnings from CSCI1470 to an applicable real-world project in a space that interested us: asset price forecasting.

# Introduction
We implemented a financial forecasting model which trains on data including stock market closing prices of financial institutions, as well as volume of weekly calls. We are utilizing deep learning methods (both LSTMs and GRUs) to tackle time series forecasting. We are implementing this paper by Gissel Velarde, Ph.D. The paper’s objectives are primarily centered around utilizing data patterns in time series data to predict data such as stock market closing prices. The problem we are solving is therefore a regression problem, given that we are predicting a continuous numerical variable. The model we implemented is trained to learn a function that maps input features (such as past stock prices and volume of weekly calls) to a continuous output variable (future stock prices).

# Methodology
Data was gathered from two different datasets. The first is the Activities dataset (activities.csv in our code) which "contains ten synthetic time series with five days of high activity and two days of low activity", resembling the volume of weekly activity in the stock market. The second is the BANKEX dataset of stock market close prices of 10 financial institutions on the Indian stock market. This data was normalized and split into training and testing components. Three different forecasts were compared: a GRU model, an LSTM model and a baseline value (repeats last observed value).

Our model consists of either a LSTM or GRU network containing a layer of 128 units, and a dense layer outputting f steps ahead. We will use an Adam optimizer, as well as Mean Squared Error (MSE) for our loss function and will train the model for 200 epochs. Since we are implementing an existing paper, we believe that the hardest part about the model’s implementation will be evaluating the results and measuring Root Mean Squared Error (RMSE) and Directional Accuracy (DA) between the actual and predicted test set values, as well as normalizing the data to inspect the results.

# Results
The results we achieved were in line with our expectations and similar to the results obtained in the paper. We spent time tuning the hyper parameters, including the learning rate, window sizes, hidden layer sizes, number of epochs, and more in order to maximize the effectiveness of our model. We also experimented with different architectures for the GRU and LSTM models, and discovered that one Dense layer for the LSTM and two Dense layers for the GRU produced the most accurate models.

# Activities Dataset (20 time steps)
## GRU WITH 100 EPOCHS:
Mean squared error on the test set: 0.0490 Root Mean Squared Error: 0.2214 Mean Directional Accuracy: 0.6033

## LSTM WITH 100 EPOCHS:
Mean squared error on the test set: 0.1820 Root Mean Squared Error: 0.4266 Mean Directional Accuracy: 0.5422

# BANKEX Dataset (20 time steps):
## GRU WITH 100 EPOCHS:
Mean squared error on the test set: 0.0009 Root Mean Squared Error: 0.0303 Mean Directional Accuracy: 0.4900

## LSTM WITH 100 EPOCHS:
Mean squared error on the test set: 0.0009 Root Mean Squared Error: 0.0302 Mean Directional Accuracy: 0.4956

** See example visuals on poster, or more detailed visuals and customizable parameters by cloning our GitHub repo and running assignment.py.

# Challenges
We encountered several challenges throughout this assignment:

PyTorch implementation: The paper implemented these models in Tensorflow, so we took on the challenge of implementing them in PyTorch, a framework we were previously unfamiliar with. It took some time to get acquainted with and we dealt with some simple PyTorch-related bugs, but we were able to complete the models with help from the PyTorch documentation and other online tutorials.

Evaluating the Model: We had some difficulty comparing the model to the model implemented in the paper, as some of the testing measures in the paper were not clearly defined.

# Initial Plan
## Introduction
We will be implementing a financial forecasting model which trains on data including stock market closing prices of financial institutions, as well as volume of weekly calls. Therefore, we are utilizing deep learning methods to tackle time series forecasting. We are implementing this paper by Gissel Velarde, Ph.D. The paper’s objectives are primarily centered around utilizing data patterns in time series data to predict data such as stock market closing prices. The problem we are solving is therefore a regression problem, given that we are predicting a continuous numerical variable. Thus, the model we are implementing is trained to learn a function that maps input features (such as past stock prices and volume of weekly calls) to a continuous output variable (future stock prices).

## Related Work
Time series forecasting has various applications, most commonly in financial and healthcare sectors. Within healthcare applications, deep learning methods have shown to be effective for time series prediction tasks, as they are able to learn representation of medical concepts and minimally processed patient and healthcare data. The attached paper on healthcare forecasting found various researchers that have contributed to time series forecasting in various research streams. The attached paper on financial forecasting discusses usage CNNs for stock price change predictions based on a time series plot.

## Data
We are utilizing the BANKEX data referenced in the paper. The BANKEX dataset contains the stock market closing prices of ten different financial institutions, with the closing price in Indian Rupee (INR). There are a total of 3,032 samples, with the data collected between July 12th, 2005 and November 3, 2017. In terms of pre-preprocessing, we will have to normalize our data between 0 and 1, define a train and test set partition, as well as select a specific time series to train the model on (the remaining samples will be used for testing).

## Methodology
Our model consists of either a LSTM or GRU network containing a layer of 128 units, and a dense layer outputting f steps ahead. We will use an Adam optimizer, as well as Mean Squared Error (MSE) for our loss function and will train the model for 200 epochs. Since we are implementing an existing paper, we believe that the hardest part about the model’s implementation will be evaluating the results and measuring Root Mean Squared Error (RMSE) and Directional Accuracy (DA) between the actual and predicted test set values, as well as normalizing the data to inspect the results.

## Metrics
Performance will be measured using the Root Mean Squared Error (RMSE) and Directional Accuracy (DA) as it is in the paper. We will also attempt to visualize the results to have an improved understanding of their comparative performance of the GRU versus LSTM models.

## Ethics
Deep learning is a good approach to this problem because it is not a subject that has high stakes or impacts human lives. Although these models can have applications with more ethical implications, such as in healthcare or finance, in our project and this paper specifically there are negligible ethical consequences.

In application, time series forecasting models can have significant stakeholders. Common applications of time series analytics include finance and business, medicine, astronomy, weather, and business development. The most significant stakeholders in all of these applications are the medical patients of which time series forecasting is used to track the progress of. The consequences of mistakes made by the model may be failure to or incorrect diagnosis or prescription, both of which may be life-threatening.

## Division of Labor
Alex: LSTM model implementation
Kleo: GRU model implementation
Ben: performance evaluation with RSME + DA
