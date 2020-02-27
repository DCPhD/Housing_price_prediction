import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# change output settings
pd.set_option("display.width", 400)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 200)

# import dataset
df = pd.read_csv('housing.csv')
#print(df.head())
#print(df['ocean_proximity'].value_counts())
#print(df.describe())

# replacing NaN values with median bedroom value
median = df['total_bedrooms'].median()
df['total_bedrooms'].fillna(median, inplace=True)

# transform categorical columnn to integer
le = preprocessing.LabelEncoder()
df['ocean_proximity'] = le.fit_transform(df['ocean_proximity'])

# eliminate target variable (these had lowest correlation with predicting house value)
X = df.drop(columns=['median_house_value', 'total_rooms', 'total_bedrooms', 'households'])

# choose only target variable
y = df['median_house_value'].values

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# apply feature scaling to normalize the range of each feature
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
scaler.transform(X_test)

####################################

# parameter grid search- looks for ideal combinations of hyperparameters to train the model
param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
              {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},]

regressor = RandomForestRegressor()

grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
#{'max_features': 6, 'n_estimators': 30}

# contribution of each predictor to the model
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
#[Longitude: 0.13289917 Latitude: 0.12862393 House Age: 0.05045673 # Rooms: 0.03522438
# bedrooms: 0.03060562 population: 0.04096127 Houses: 0.02769309 House value: 0.44566006 Ocean prox: 0.10787576]

######################################

# Run Random Forest Regressor with 6 best performing features based on Grid Search
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# check accuracy
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Mean Absolute Error: 32046.870898740308
# Mean Squared Error: 2494014756.9515824
# Root Mean Squared Error: 49940.11170343517

#### Run on test set ####

# apply feature scaling to normalize the range of each feature
scaler = StandardScaler()
X_test_scaler = scaler.fit_transform(X_test)
scaler.transform(X_train)

# try Random Forest Regressor
regressor = RandomForestRegressor()
regressor.fit(X_test, y_test)

y_pred = regressor.predict(X_train)

# check accuracy
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred))
print('Root Mean Squared Error :', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))


# Mean Absolute Error: 36852.82017926357
# Mean Squared Error: 3007605401.257404
# Root Mean Squared Error: 54841.63930133201