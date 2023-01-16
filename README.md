# Module_14_Challenge

Implementing an algorithmic trading strategy that uses machine learning to automate the trade decisions - Machine Learning Trading Bot

![image](https://user-images.githubusercontent.com/110797348/212583458-1066ef0b-2455-470f-8c35-d9b814c201e0.png)

# Machine Learning Trading Bot

In this Challenge, the role of a financial advisor at one of the top five financial advisory firms in the world is assumes. Here I plan to improve the existing algorithmic trading systems and maintain the firm’s competitive advantage in the market place. To do so, there is the need to enhance the existing trading signals with machine learning algorithms that can adapt to new data.

# Instructions:

## Use the starter code file to complete the steps that the instructions outline. The steps for this Challenge are divided into the following sections:

Establish a Baseline Performance

Tune the Baseline Trading Algorithm

Evaluate a New Machine Learning Classifier

Create an Evaluation Report

# Libraries & Dependencies
The following libraries & dependencies were imported
import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report

# Project findings

### Step 1: Tune the training algorithm by adjusting the size of the training dataset. 

#### To do so, slice your data into different periods. Rerun the notebook with the updated parameters, and record the results in your `README.md` file. 

#### Answer the following question: What impact resulted from increasing or decreasing the training window?

Original Testing Report of the SVM Model 3 Months
              precision    recall  f1-score   support

        -1.0       0.43      0.04      0.07      1804
         1.0       0.56      0.96      0.71      2288

    accuracy                           0.55      4092
   macro avg       0.49      0.50      0.39      4092
weighted avg       0.50      0.55      0.43      4092

accuracy 0.55(55%)
macro avg 0.49 0.50 0.39 4092 
weighted avg 0.50 0.55 0.43 4092

### Answer the following question: What impact resulted from increasing or decreasing the training window? An increase in the training window results in a marginal increase in the accuracy level by 1% i.e from 55% to 56% 

Testing Report of the SVM Model 24 Months

              precision    recall  f1-score   support

    -1.0       0.80      0.00      0.01      1229
     1.0       0.56      1.00      0.72      1565

accuracy                           0.56      2794

macro avg 0.68 0.50 0.36 2794 
weighted avg 0.67 0.56 0.41 2794


### Step 2: Tune the trading algorithm by adjusting the SMA input features. 

Adjust one or both of the windows for the algorithm. Rerun the notebook with the updated parameters, and record the results in your `README.md` file. 

Answer the following question: What impact resulted from increasing or decreasing either or both of the SMA windows?.

Original Report of the SVM Model(Simple Moving Average Crossover strategy)-short_window = 4, long_window = 100 i.e a faster moving average(short-term) of 25days and a slower(long-term) moving average of 200 days

              precision    recall  f1-score   support

        -1.0       0.43      0.04      0.07      1804
         1.0       0.56      0.96      0.71      2288

    accuracy                           0.55      4092
   macro avg       0.49      0.50      0.39      4092
weighted avg       0.50      0.55      0.43      4092

accuracy -0.55(55%)
macro avg 0.49 0.50 0.39 4092 
weighted avg 0.50 0.55 0.43 4092

### Answer the following question:What impact resulted from increasing or decreasing either or both of the SMA windows? As seen in the Machine_learning_tuning of the trading window(see below), the accuracy increase marginally by 1% i.e from 55% to 56% while the Precision also decreased both for the macro avg and the weighted avg while recall stays alomost at the same level.

Testing Report(increasing the training window) of the SVM Model(Simple Moving Average Crossover strategy)-short_window = 25, long_window = 200 i.e a faster moving average(short-term) of 25days and a slower(long-term) moving average of 200 days

                precision    recall  f1-score   support

        -1.0       0.00      0.00      0.00      1694
         1.0       0.56      1.00      0.72      2161

    accuracy                           0.56      3855
   macro avg       0.28      0.50      0.36      3855
weighted avg       0.31      0.56      0.40      3855

accuracy - 0.56(or 56%)
macro avg     0.28 0.50 0.36 3855
weighted avg  0.31 0.56 0.40 3855

### Note further:
A moving average, as a line by itself, is often overlaid in price charts to indicate price trends. A crossover occurs when a faster moving average (i.e. a shorter period moving average) crosses a slower moving average (i.e. a longer period moving average). In stock trading, this meeting point can be used as a potential indicator to buy or sell an asset.

When the short term moving average crosses above the long term moving average, this indicates a buy signal.
Contrary, when the short term moving average crosses below the long term moving average, it may be a good moment to sell.

![image](https://user-images.githubusercontent.com/110797348/212585266-1b78d0bc-b3dc-43c0-9613-b679bbe3afa3.png)

![image](https://user-images.githubusercontent.com/110797348/212585347-c6dcd5d6-c103-41f8-a823-15c7f6f46765.png)

# Step 3: Backtest the new model to evaluate its performance. 

Save a PNG image of the cumulative product of the actual returns vs. the strategy returns for this updated trading algorithm, and write your conclusions in your `README.md` file. 

## Answer the following questions: 
### Did this new model perform better or worse than the provided baseline model? 
### Did this new model perform better or worse than your tuned trading algorithm?

Its always a very tricky to make a decision which strategy to adopt on the basis of which is better or worse. The baseline model seems like predicting an actual trend is better than the second model i.e the tuned trading algorithm in terms of accuracy score. In this section, we see that first model project results closer to the actual ones, but classification report shows that the first one has slightly better quality of prediction(accuracy score at 55% compared with 52%). Therefore, in order make a decision as to which is better or worse, there is the need to consider to run more version of the model to make sure the prediction outcome is correct and can be relied upon.

### Baseline(first) model

              precision    recall  f1-score   support

        -1.0       0.43      0.04      0.07      1804
         1.0       0.56      0.96      0.71      2288

    accuracy                           0.55      4092
   macro avg       0.49      0.50      0.39      4092
weighted avg       0.50      0.55      0.43      4092


### second model

              precision    recall  f1-score   support

        -1.0       0.44      0.33      0.38      1804
         1.0       0.56      0.66      0.61      2288

    accuracy                           0.52      4092
   macro avg       0.50      0.50      0.49      4092
weighted avg       0.51      0.52      0.51      4092




