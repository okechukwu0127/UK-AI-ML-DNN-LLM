#!/usr/bin/env python
# coding: utf-8

# ### Task 1: Regression – Predicting House Prices
# 
# #### Summary of code 
# - This script demonstrates a real-world linear regression model using actual data from a CSV file named houseprice_data.csv.
# - The objective of this task is to develop a predictive model that estimates house prices in King County, USA, based on various features such as number of bedrooms, bathrooms, living area, and more.

# In[421]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# sklearn package for machine learning in python:
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# In[ ]:





# #### Simple Linear Regression Model
# A simple linear regression model was fit using only **sqft_living** as the predictor variable:

# In[422]:


# Load the dataset
df = pd.read_csv('houseprice_data.csv') # read data (make sure .csv in folder)
#df = pd.read_csv('heart_data.csv') # read data (make sure .csv in folder)

X_single = df[['sqft_living']] # inputs sqft_living
y = df['price'] # outputs price


# In[423]:


#print(df.head(),'\n') # print first 5 rows of data


# In[ ]:





# ### Split the data into training and test sets
#  - Into training (2/3) and testing (1/3) sets using train_test_split.

# In[424]:


# Split data
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_single, y, test_size=0.33, random_state=0)


# In[ ]:





# ### Fit the linear least-squares regression line to the training data:
# #### Single Linear Regression model:
# 
#  - Using the training data (X_train, y_train).
#  - Fits a best-fit line using least squares method.
# 

# In[425]:


# Build and train the model

r_single = LinearRegression()
lr_single.fit(X_train_s, y_train_s)


# In[ ]:





# In[426]:


#Predict single value:
y_pred_s = lr_single.predict(X_test_s)


# In[ ]:





# In[427]:


# The coefficients
print('Coefficients: ', lr_single.coef_)

# The Intercept
print('Intercept: ', lr_single.intercept_)

# The mean squared error
print('Mean squared error: %.2f'
% mean_squared_error(y_test, y_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
% r2_score(y_test, y_pred))

print(f"Mean Absolute Error: ${mean_absolute_error(y_test, y_pred_s):.2f}")


# 
#  - Coefficient (slope) and intercept of the regression line.
#  - Mean Squared Error (MSE) – measures average squared difference between predicted and actual values.
#  - R² (coefficient of determination) – measures how well the model explains the variance in the data (1 is perfect).

# In[ ]:





# #### Visualise training data set results

# In[428]:


# visualise training data set results
fig2, ax2 = plt.subplots()
ax2.scatter(X_train_s, y_train_s, color='blue')
ax2.plot(X_train_s, lr_single.predict(X_train_s), color='red')
ax2.set_xlabel('Sqft_living')
ax2.set_title('Price vs Sqft_living (Training set)')
ax2.set_ylabel('Price')


# In[ ]:





# #### Visualise test data set results 

# In[429]:


# visualise test data set results
fig3, ax3 = plt.subplots()
ax3.scatter(X_test_s, y_test_s, color='red')
ax3.plot(X_test_s, lr_single.predict(X_test_s), color='blue')
ax3.set_xlabel('Sqft_living')
ax3.set_title('Price vs Sqft_living (Test set)')
ax3.set_ylabel('Price')


# In[ ]:





# ### Let us identify the top N features Using Correlation with Price
# This method ranks features by how strongly they correlate with the target price.

# In[430]:


# Compute absolute correlations with 'price'
corr = df.corr(numeric_only=True)['price'].abs().sort_values(ascending=False)

# Exclude 'price' itself and select top N features
top_features = corr.drop('price')#.head(5)
print("\nTop features with most correlated with price:\n", top_features)

#Return only the feature names
best_feature_names = top_features.index.tolist()
print("\nTop 5 feature names most correlated with price: ", best_feature_names[:5])

#best_feature_names[1]


# In[ ]:





# #### Correlation heatmap showing relationships between top features
# Feature importance based on coefficients
# 

# In[447]:


# Additional: Create a 2D correlation heatmap for the top features
plt.figure(figsize=(10,8))
top_features_with_price = best_feature_names[:18] + ['price']  # Top 3 features + price
corr_matrix = df[top_features_with_price].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.1f')
plt.title('Correlation Matrix: Top Features with Price')
plt.tight_layout()
plt.savefig(fname='Correlation_heatmap_relationships.png',bbox_inches='tight')
plt.show()


# In[ ]:





# ### Multiple regression model for the top 2 features:
#  - We are generating data with 2 input features (sqft_living and grade).
#  - Because X has two columns, the model learns two coefficients — one for each feature.
#  - This is Multiple Linear Regression because:
#  - y=b0​+b1​⋅X1​+b2​⋅X2​

# In[432]:


features = [best_feature_names[0], best_feature_names[1]] # input sqft_living and grade
X_two = df[features]
#y = df['price']


# In[ ]:





# In[433]:


# split the data into training and test sets:
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_two, y, test_size = 0.33, random_state = 0)


# In[ ]:





# In[434]:


# fit the linear least-squares regression line to the training data:
lr_two = LinearRegression()
lr_two.fit(X_train_t, y_train_t)


# In[435]:


y_pred_t = lr_two.predict(X_test_t)


# In[436]:


# The coefficients
print('Coefficients: ', lr_two.coef_)

# The coefficients
print('Intercept: ', lr_two.intercept_)

# The mean squared error
print('Mean squared error: %.2f'
% mean_squared_error(y_test_t, y_pred_t))

#r2_score(y_test_f, y_pred_final):.3f
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
% r2_score(y_test_t, y_pred_t))

print(f"Mean Absolute Error: ${mean_absolute_error(y_test_t, y_pred_t):.2f}")


# In[ ]:





# #### Create a 3D Scatter Plot Vsializing Training and Test data
# Let’s visualize the fitted plane on the training and test data

# In[437]:


# Create 3D visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot training data points
ax.scatter(X_train_t[features[0]], X_train_t[features[1]], y_train_t, 
           color='blue', alpha=0.6, label='Training Data', s=10)

# Plot test data points
ax.scatter(X_test_t[features[0]], X_test_t[features[1]], y_test_t, 
           color='red', alpha=0.6, label='Test Data', s=10)

# Create meshgrid for regression plane
x1_range = np.linspace(X_train_t[features[0]].min(), X_train_t[features[0]].max(), 20)
x2_range = np.linspace(X_train_t[features[1]].min(), X_train_t[features[1]].max(), 20)
X1, X2 = np.meshgrid(x1_range, x2_range)

# Calculate Z values for regression plane
Z = lr_two.coef_[0] * X1 + lr_two.coef_[1] * X2 + lr_two.intercept_


# Plot regression surface
ax.plot_surface(
    x_surf, y_surf, z_pred,
    alpha=0.4, cmap='viridis', edgecolor='none', label='Regression Surface'
)

# Set labels and title
ax.set_xlabel(features[0].replace('_', ' ').title())
ax.set_ylabel(features[1].replace('_', ' ').title())
ax.set_zlabel('Price')
ax.set_title(f'Multiple Regression: {features[0]} and {features[1]} vs Price\n'
             f'R² = {r2_score(y_test_t, y_pred_t):.2f}')

# Set viewing angle
ax.view_init(elev=20, azim=-45)

# Add legend
ax.legend()

plt.tight_layout()
plt.show()


# In[ ]:





# #### Multiple Regression Model for the Top 5 features
# The model was then expanded to include several the top 5 features sqft_living, grade, sqft_above, sqft_living15, bathrooms:

# In[448]:


#features_five = ['sqft_living', 'grade', 'bathrooms', 'sqft_above', 'view']
#print(best_feature_names[:5])


features_five = best_feature_names[:5] #['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']
X_five = df[features_five]

y = df['price']

#print(X_multi_)
#print(best_feature_names)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_five, y, test_size=0.33, random_state=42)

lr_five = LinearRegression()
lr_five.fit(X_train_f, y_train_f)

y_pred_f = lr_five.predict(X_test_f)


# The coefficients
print('Coefficients: ', lr_five.coef_)

# The coefficients
print('Intercept: ', lr_five.intercept_)

# The mean squared error
print('Mean squared error: %.2f'
% mean_squared_error(y_test_f, y_pred_f))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
% r2_score(y_test_f, y_pred_f))

print(f"Mean Absolute Error: ${mean_absolute_error(y_test_f, y_pred_f):.2f}")


# In[ ]:





# #### Multiple Regression Model: All  features
# The model was then expanded to include all the features:

# In[439]:


# Select all columns EXCEPT the target 'price' as input features
X_all = df.drop(columns=['price'])

# Select 'price' as the target variable
y_all = df['price']


# In[ ]:





# In[440]:


X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.33, random_state=0)

lr_multi_all = LinearRegression()
lr_multi_all.fit(X_train, y_train)
y_pred_multi_all = lr_multi_all.predict(X_test)

# The coefficients
print('Coefficients: ', lr_multi_all.coef_)

# The coefficients
print('Intercept: ', lr_multi_all.intercept_)

# The mean squared error
print('Mean squared error: %.2f'
% mean_squared_error(y_test, y_pred_multi_all))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
% r2_score(y_test, y_pred_multi_all))


# In[ ]:





# In[ ]:





# In[449]:


# Results compilation
results = [
    ['Single Feature', 0.50, -29484, 72.50],
    ['Two Features', 0.54, -585687, 66.90],
    ['Five Features', 0.538058, -628967, 69.250911],
    ['All Features', 0.68, 183619, 45.60]
]

results_df = pd.DataFrame(results, columns=['Model', 'R-squared', 'Intercept', 'MSE (Billion)'])


# In[ ]:





# In[450]:


#Subplot 2: Model comparison
plt.subplot(1, 2, 2)
models = results_df['Model']
r_squared = results_df['R-squared']
plt.bar(models, r_squared, color=['lightblue', 'lightgreen', 'orange', 'coral'])
plt.xlabel('Model Complexity')
plt.ylabel('R-squared')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
#plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:




