### Prediction of house price in Amsterdam in 2021

From the Kaggle repository of datasets, this one contains `6` features concerning house characteristics of `924` rows (`740` for training and `186` for testing). The task is to predict the price of the house, hence the proble will be formulated as regression task*.

 Data Source: https://www.kaggle.com/datasets/thomasnibb/amsterdam-house-price-prediction

 
Here are the features and their data types: 
-  Address:* text. 
-  Zip:* text. 
-  Price:* int
-  Area:* int
-  Room:* int. 
-  Lon:* float
-  Lat:* float

During feature engineering stage a new statistically signifficant feature was created: `Manhattan Distance`. Its values were calculated using `Lon` and `Lat`.
In the end there is a Linear Regression model, which uses `Area` and `Manhattan Distance` features.