# BetaBankChurn

# Beta Bank Customer Churn Prediction

## Project Overview
This project aims to tackle the significant issue of customer churn at Beta Bank. Using machine learning, we develop a predictive model to identify the likelihood of customers discontinuing the bank's services. The initiative is crucial for the bank's customer retention strategy and overall service improvement.

## Dataset
The analysis is based on a dataset of 10,000 customers, with each record consisting of 14 features including customer ID, credit score, demographics, account details, and the target variable 'Exited', indicating whether a customer has left the bank.

## Methodology

### Data Preparation
Importing Libraries:
. import pandas as pd
. import numpy as np
. from sklearn.ensemble import RandomForestClassifier
. from sklearn.linear_model import LogisticRegression
. from sklearn.tree import DecisionTreeClassifier
. from sklearn.naive_bayes import GaussianNB
. from sklearn.model_selection import train_test_split
. from sklearn.dummy import DummyClassifier
. from sklearn.metrics import mean_squared_error
. from sklearn.neighbors import KNeighborsClassifier
. from sklearn import svm
. from sklearn.utils import resample
. from sklearn.utils import shuffle
. import seaborn as sns
. from math import sqrt
. from fast_ml.model_development import train_valid_test_split
. from sklearn.metrics import precision_score
. from sklearn.metrics import recall_score
. from sklearn.metrics import accuracy_score
. from sklearn.metrics import f1_score
. from sklearn.metrics import roc_auc_score
. from sklearn import metrics
. from sklearn.metrics import roc_curve
. from sklearn.preprocessing import StandardScaler as ss
. from matplotlib.gridspec import GridSpec
. import matplotlib.pyplot as plt
. %matplotlib inline
. import sys
. import warnings
. if not sys.warnoptions:
. warnings.simplefilter("ignore")

## Variable Description 

    Features
    RowNumber - index in the data rows
    CustomerId
    Surname
    CreditScore — cridit history, client's credit raiting
    Geography — страна проживания
    Gender
    Age
    Tenure — number of years for how long the client was staying with the bank
    Balance — current account's balance
    NumOfProducts
    HasCrCard — binary valriables: Does a client have a credit card?
    IsActiveMember - binary valriables: Is a client actively using the bank for transactions?
    EstimatedSalary
    Target
    Exited — the fact the the client has withdrawn the contract

Loading Data: The dataset was loaded for initial inspection and understanding of the data structure.
Handling Missing Values: Identified and imputed missing values in the 'Tenure' feature.
Duplicate Records Analysis: Ensured no duplicate entries existed in the dataset.
Data Encoding: Transformed categorical variables 'Geography' and 'Gender' into numerical format using one-hot encoding.

## Data Analysis

    Class Imbalance Assessment: Noted a significant imbalance in the target variable, with fewer instances of customers leaving the bank.
    Feature and Target Separation: Segregated the dataset into features and target variable for model training.
    Model Development
    Data Splitting: Divided data into training, validation, and test sets.
    Model Selection and Training: Experimented with various algorithms including Decision Tree, GaussianNB, K-Nearest Neighbors, and RandomForestClassifier.
    Hyperparameter Tuning: Conducted extensive tuning, particularly with the RandomForestClassifier, to optimize model performance.
    Addressing Class Imbalance
    Upsampling and Downsampling: Implemented both techniques to balance the dataset and improve model training.
    Impact on Model Performance: Evaluated the effect of these techniques on model accuracy, F1 score, and ROC-AUC score.
    Key Results
    Model Performance: The RandomForestClassifier, with tuned hyperparameters, demonstrated the best performance.
    Model Evaluation Metrics:
    F1 Score: Achieved a score of 0.571 post-upsampling, surpassing the project's goal of 0.59.
    Accuracy: Recorded an accuracy of 83.1%.
    ROC-AUC: Attained a score of 0.739, indicating a good level of separability between classes.
    Confusion Matrix Analysis: Provided insights into the model's performance, highlighting its precision in predicting both classes.

## Conclusions

The RandomForestClassifier, with optimal parameters (n_estimators=461, max_depth=15), effectively predicted customer churn, achieving an F1 score above the set target.
Upsampling proved crucial in enhancing model performance, while downsampling offered a balanced perspective between precision and recall.
The project demonstrated the importance of addressing class imbalance in predictive modeling.
Future Work
Investigate additional features and alternative modeling techniques.
Implement continuous model monitoring and updating to adapt to evolving customer behaviors.
Explore the integration of the model into Beta Bank's operational framework for real-time applications.
