from utils import db_connect
engine = db_connect()

from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

# load the .env file variables
load_dotenv()


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine

## **US county-level sociodemographic and health resource data (2018-2019).**
### Sociodemographic and health resource data have been collected by county in the United States and we want to find out if there is any relationship between health resources and sociodemographic data.

### To do this, you need to set a target variable (health-related) to conduct the analysis.
# Import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from pickle import dump
### **Step 1:** Loading the dataset
# Load the dataset from this link: https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv
resource_data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv')
resource_data
### **Step 2:** Full EDA
# Data info
resource_data.info()
# Data columns
resource_data.columns
# Duplicate values
resource_data.duplicated().sum()
# Unique values
resource_data.nunique()
# Null values
print(resource_data.isnull().sum())
print('\n\n')
print(resource_data.isna().sum())
# categorical values
resource_data.select_dtypes(include=['object']).columns
# numerical values
resource_data.select_dtypes(include=['int64','float64']).columns

#### **Part 1: Sociodemographic data**
# Select the socialdemographic columns
sociodemographic = resource_data.iloc[:, 0:60]
sociodemographic
sociodemographic.columns
# Drop the columns that are not needed
sociodemographic.drop(
    [
        '0-9','0-9 y/o % of total pop','19-Oct', '10-19 y/o % of total pop', '20-29', '20-29 y/o % of total pop', '30-39', '30-39 y/o % of total pop', '40-49',
        '40-49 y/o % of total pop', '50-59', '50-59 y/o % of total pop', '60-69', '60-69 y/o % of total pop', '70-79', '70-79 y/o % of total pop', '80+', '80+ y/o % of total pop',
        'White-alone pop', '% White-alone', 'Black-alone pop', '% Black-alone', 'Native American/American Indian-alone pop', '% NA/AI-alone', 'Asian-alone pop', '% Asian-alone',
        'Hawaiian/Pacific Islander-alone pop', '% Hawaiian/PI-alone', 'Two or more races pop', '% Two or more races', 'POP_ESTIMATE_2018', 'N_POP_CHG_2018', 'GQ_ESTIMATES_2018',
        'Percent of adults with less than a high school diploma 2014-18', 'Percent of adults with a high school diploma only 2014-18',
        "Percent of adults completing some college or associate's degree 2014-18", "Percent of adults with a bachelor's degree or higher 2014-18", 'PCTPOVALL_2018', 'PCTPOV017_2018',
        'PCTPOV517_2018', 'CI90LBINC_2018', 'CI90UBINC_2018'
    ]
    , axis=1, inplace=True)
sociodemographic
# Add the columns state and county
sociodemographic['state'] = resource_data['STATE_NAME']
sociodemographic['county'] = resource_data['COUNTY_NAME']
# Dataframe info
sociodemographic.info()
# Dataframe null values
sociodemographic.isnull().sum()
# Duplicate values
sociodemographic.duplicated().sum()

# Visualize the histograms of the sociodemographic columns
plt.figure(figsize=(20, 20))

for i, col in enumerate(sociodemographic.columns):
    plt.subplot(5, 4, i+1)
    plt.hist(sociodemographic[col])
    plt.title(col)
    plt.xticks(rotation=90)
    plt.tight_layout()
plt.show()

#### **Part 2: Health resource data**
# Select the health resources data
health_resources = resource_data.iloc[:, 62:75]
health_resources
# Dataframe info
health_resources.info()
# Dataframe null values
health_resources.isnull().sum()
# Duplicate values
health_resources.duplicated().sum()
# Visualize the histograms of the health resources columns
plt.figure(figsize=(20, 20))

for i, col in enumerate(health_resources.columns):
    plt.subplot(3, 5, i+1)
    plt.hist(health_resources[col])
    plt.title(col)
    plt.xticks(rotation=90)
    plt.tight_layout()
#### **Part 3: Target variable**
# Select the target column
target = resource_data[['diabetes_number']]
target
#### **Part 4: Joining all the parts together**
# Join the dataframes
total_data = pd.concat([sociodemographic, health_resources, target], axis=1)
total_data
#### **Part 5: Correlation matrix**
# Convert the categorical columns into numerical columns using factorize
total_data['state'] = pd.factorize(total_data['state'])[0]
total_data['county'] = pd.factorize(total_data['county'])[0]

# Correlation the data with the target column
corr = total_data.corr()['diabetes_number']
corr.sort_values(ascending=False)
# Create a reglot and heatmap to visualize the correlation between variables
fig, ax = plt.subplots(14, 5, figsize=(50, 50))

# Define the colors
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.regplot(ax=ax[0, 0], x = 'fips', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[1, 0], data = total_data[['fips', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[0, 1], x = 'TOT_POP', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[1, 1], data = total_data[['TOT_POP', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[0, 2], x = 'R_birth_2018', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[1, 2], data = total_data[['R_birth_2018', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[0, 3], x = 'R_death_2018', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[1, 3], data = total_data[['R_death_2018', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[0, 4], x = 'R_NATURAL_INC_2018', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[1, 4], data = total_data[['R_NATURAL_INC_2018', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[2, 0], x = 'R_INTERNATIONAL_MIG_2018', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[3, 0], data = total_data[['R_INTERNATIONAL_MIG_2018', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[2, 1], x = 'R_DOMESTIC_MIG_2018', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[3, 1], data = total_data[['R_DOMESTIC_MIG_2018', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[2, 2], x = 'R_NET_MIG_2018', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[3, 2], data = total_data[['R_NET_MIG_2018', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[2, 3], x = 'Less than a high school diploma 2014-18', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[3, 3], data = total_data[['Less than a high school diploma 2014-18', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[2, 4], x = 'High school diploma only 2014-18', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[3, 4], data = total_data[['High school diploma only 2014-18', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[4, 0], x = "Some college or associate's degree 2014-18", y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[5, 0], data = total_data[["Some college or associate's degree 2014-18", 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[4, 1], x = "Bachelor's degree or higher 2014-18", y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[5, 1], data = total_data[["Bachelor's degree or higher 2014-18", 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[4, 2], x = 'POVALL_2018', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[5, 2], data = total_data[['POVALL_2018', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[4, 3], x = 'MEDHHINC_2018', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[5, 3], data = total_data[['MEDHHINC_2018', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[4, 4], x = 'Civilian_labor_force_2018', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[5, 4], data = total_data[['Civilian_labor_force_2018', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[6, 0], x = 'Employed_2018', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[7, 0], data = total_data[['Employed_2018', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[6, 1], x = 'Unemployed_2018', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[7, 1], data = total_data[['Unemployed_2018', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[6, 2], x = 'Unemployment_rate_2018', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[7, 2], data = total_data[['Unemployment_rate_2018', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[6, 3], x = 'state', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[7, 3], data = total_data[['state', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[6, 4], x = 'county', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[7, 4], data = total_data[['county', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[8, 0], x = 'Active Physicians per 100000 Population 2018 (AAMC)', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[9, 0], data = total_data[['Active Physicians per 100000 Population 2018 (AAMC)', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[8, 1], x = 'Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[9, 1], data = total_data[['Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[8, 2], x = 'Active Primary Care Physicians per 100000 Population 2018 (AAMC)', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[9, 2], data = total_data[['Active Primary Care Physicians per 100000 Population 2018 (AAMC)', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[8, 3], x = 'Active Patient Care Primary Care Physicians per 100000 Population 2018 (AAMC)', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[9, 3], data = total_data[['Active Patient Care Primary Care Physicians per 100000 Population 2018 (AAMC)', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[8, 4], x = 'Active General Surgeons per 100000 Population 2018 (AAMC)', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[9, 4], data = total_data[['Active General Surgeons per 100000 Population 2018 (AAMC)', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[10, 0], x = 'Active Patient Care General Surgeons per 100000 Population 2018 (AAMC)', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[11, 0], data = total_data[['Active Patient Care General Surgeons per 100000 Population 2018 (AAMC)', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[10, 1], x = 'Total nurse practitioners (2019)', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[11, 1], data = total_data[['Total nurse practitioners (2019)', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[10, 2], x = 'Total physician assistants (2019)', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[11, 2], data = total_data[['Total physician assistants (2019)', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[10, 3], x = 'Total Hospitals (2019)', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[11, 3], data = total_data[['Total Hospitals (2019)', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[10, 4], x = 'Internal Medicine Primary Care (2019)', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[11, 4], data = total_data[['Internal Medicine Primary Care (2019)', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[12, 0], x = 'Family Medicine/General Practice Primary Care (2019)', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[13, 0], data = total_data[['Family Medicine/General Practice Primary Care (2019)', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[12, 1], x = 'Total Specialist Physicians (2019)', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[13, 1], data = total_data[['Total Specialist Physicians (2019)', 'diabetes_number']].corr().values, annot=True, cmap=cmap)
sns.regplot(ax=ax[12, 2], x = 'ICU Beds_x', y = 'diabetes_number', data = total_data, color='blue')
sns.heatmap(ax=ax[13, 2], data = total_data[['ICU Beds_x', 'diabetes_number']].corr().values, annot=True, cmap=cmap)

# Delete the columns that are not needed
plt.delaxes(ax[12, 3])
plt.delaxes(ax[12, 4])
plt.delaxes(ax[13, 3])
plt.delaxes(ax[13, 4])

plt.show()
# Feature Engineering
# Select the columns that have a correlation greater than 0.5
columns_selected = [
    'TOT_POP', 'High school diploma only 2014-18', 'Unemployed_2018', 'Civilian_labor_force_2018', 'Employed_2018', "Some college or associate's degree 2014-18",
    'POVALL_2018', 'Family Medicine/General Practice Primary Care (2019)', 'Less than a high school diploma 2014-18', 'Total nurse practitioners (2019)', 'Total Specialist Physicians (2019)',
    "Bachelor's degree or higher 2014-18", 'Internal Medicine Primary Care (2019)', 'Total physician assistants (2019)', 'Total Hospitals (2019)', 'ICU Beds_x', 'diabetes_number'
]

# Create the scaler object
scaler = MinMaxScaler()

# Scale the data
scaled_features = scaler.fit_transform(total_data[columns_selected])
diabetes_prediction = pd.DataFrame(scaled_features, columns=columns_selected)
diabetes_prediction
# Feature selection
# Separte the data into features and target
X = diabetes_prediction.drop(['diabetes_number'], axis=1)
y = diabetes_prediction['diabetes_number']

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
train_indices = list(X_train.index)
test_indices = list(X_test.index)

# Select the best features with SelectKBest
best_features = SelectKBest(score_func=f_regression, k=10)
fit = best_features.fit(X_train, y_train)
ix = fit.get_support()

# Create a dataframe with the best features
X_train = pd.DataFrame(best_features.transform(X_train), index = train_indices, columns=X_train.columns.values[ix])
X_test = pd.DataFrame(best_features.transform(X_test), index = test_indices, columns=X_test.columns.values[ix])

# Display the dataframe
X_train
X_test
# Add the target column to the dataframes
X_train['diabetes_number'] = y_train
X_test['diabetes_number'] = y_test

# Save the train and test dataframes
X_train.to_csv('../data/processed/train.csv', index = False)
X_test.to_csv('../data/processed/test.csv', index = False)
### **Step 3:** Build a Regression Model
# Read the data from file
train_df = pd.read_csv('../data/processed/train.csv')
test_df = pd.read_csv('../data/processed/test.csv')

# Show the first rows of the train dataframe
train_df.head()
test_df.head()
# Separate the feature and the target columns
X_train = train_df.drop(columns=['diabetes_number'])
y_train = train_df['diabetes_number']

# Separate the feature and the target columns
X_test = test_df.drop(columns=['diabetes_number'])
y_test = test_df['diabetes_number']
# Create the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
# Know the parameters of the model
print('Intercept:', linear_model.intercept_)
print('Coefficients:', linear_model.coef_)
# Predict the values
y_pred = linear_model.predict(X_test)
y_pred
# Evaluate the lineal regression model
score = linear_model.score(X_test, y_test)
print("Coefficients:", linear_model.coef_)
print("R2 score:", score)
# create a Lasso regressor
lasso_model = Lasso(alpha=1.0, max_iter=300)

# Train the model
lasso_model.fit(X_train, y_train)

# Predict the values
y_pred_2 = lasso_model.predict(X_test)
y_pred_2
# Evaluamos el rendimiento del modelo en los datos de prueba
score = lasso_model.score(X_test, y_test)
print("Coefficients:", lasso_model.coef_)
print("R2 score:", score)
### **Step 4:** Optimize the model
# Define the parameters
hyperparameters = {
    'alpha': [0.1, 0.5, 1.0, 10.0, 100.0],
    'max_iter': [100, 200, 300, 400, 500]
}

# Create the grid search
grid = GridSearchCV(lasso_model, hyperparameters, cv=5)
grid
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

grid.fit(X_train, y_train)

print(f"Best hyperparameters: {grid.best_params_}")
model_grid = lasso_model.set_params(alpha = 0.1, max_iter = 100)
y_pred_2 = model_grid.predict(X_test)

# Evaluamos el rendimiento del modelo en los datos de prueba
score = model_grid.score(X_test, y_test)
print("Coefficients:", model_grid.coef_)
print("R2 score:", score)
# Save the models in a file
dump(linear_model, open('../models/linear_model.pkl', 'wb'))
dump(lasso_model, open('../models/lasso_model.pkl', 'wb'))
dump(model_grid, open('../models/model_grid.pkl', 'wb'))
