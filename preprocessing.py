# %% [markdown]
# <p style="text-align:center;"> <span style="font-size:36px;"> <b> Understanding the data and performing regressions </b> </span> </p>
# 
# <img src="https://cdn-images-1.medium.com/max/1024/1*hLWUW_4ZPjKyP01Bnlyv4w.png" width="700px">
# 
# ## Acknowledgements:
# 
# For the purpose of training and selecting the model, I took reference from these following notebooks: [
# Getting Started with 30 Days of ML Competition](https://www.kaggle.com/alexisbcook/getting-started-with-30-days-of-ml-competition) and [
# 3rd-place solution: Ensembling GBDTs](https://www.kaggle.com/kntyshd/3rd-place-solution-ensembling-gbdts), and is inspired from the different courses offered by Kaggle. And for performing the EDA on the dataset to understand it better, the following notebook helped me a lot: [TPS Feb 2021 EDA](https://www.kaggle.com/dwin183287/tps-feb-2021-eda/)
# 

# %% [markdown]
# # Step 1: Import helpful libraries

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:45:45.383996Z","iopub.execute_input":"2021-08-27T21:45:45.384743Z","iopub.status.idle":"2021-08-27T21:45:48.354773Z","shell.execute_reply.started":"2021-08-27T21:45:45.384634Z","shell.execute_reply":"2021-08-27T21:45:48.354027Z"}}
import numpy as np
import pandas as pd
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlxtend.preprocessing import minmax_scaling

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# %% [markdown]
# # Step 2: Load the data
# 
# Next, we'll load the training and test data.  
# 
# We set `index_col=0` in the code cell below to use the `id` column to index the DataFrame.  (*If you're not sure how this works, try temporarily removing `index_col=0` and see how it changes the result.*)

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:45:48.355918Z","iopub.execute_input":"2021-08-27T21:45:48.356361Z","iopub.status.idle":"2021-08-27T21:45:52.811534Z","shell.execute_reply.started":"2021-08-27T21:45:48.356332Z","shell.execute_reply":"2021-08-27T21:45:52.810525Z"}}
# Load the training data
train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)

# Preview the data
train.head()


# %% [markdown]
# ### Observations:
# 
# - Train set has 300,000 rows while test set has 200,000 rows.
# - There are 10 categorical features from `cat0` - `cat9` and 14 continuous features from `cont0` - `cont13`.
# - There is no missing values in the train and test dataset but there is no category `G` in `cat6` test dataset.
# - Categorical features ranging from alphabet A - O but it varies from each categorical feature with `cat0`, `cat1`, `cat3`, `cat5` and `cat6` are dominated by one category.
# - Continuous features on train anda test dataset ranging from -0.1 to 1.25 which are a multimodal distribution and they resemble each other.
# - target has a range between 6.8 to 10.5 and has a bimodal distribution.
# 
# 
# Ideas:
# 
# Drop features that are dominated by one category cat0, cat1, cat3, cat5 and cat6 as they don't give variation to the dataset but further analysis still be needed.

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:45:53.253424Z","iopub.execute_input":"2021-08-27T21:45:53.253804Z","iopub.status.idle":"2021-08-27T21:45:53.259418Z","shell.execute_reply.started":"2021-08-27T21:45:53.253772Z","shell.execute_reply":"2021-08-27T21:45:53.257822Z"}}
cat_features = [feature for feature in train.columns if 'cat' in feature]
cont_features = [feature for feature in train.columns if 'cont' in feature]


# %% [markdown]
# # Step 4: Prepare the data
# 
# The next code cell separates the target (which we assign to `y`) from the training features (which we assign to `features`).

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:46:49.493443Z","iopub.execute_input":"2021-08-27T21:46:49.493762Z","iopub.status.idle":"2021-08-27T21:46:49.570788Z","shell.execute_reply.started":"2021-08-27T21:46:49.493732Z","shell.execute_reply":"2021-08-27T21:46:49.570104Z"}}
# Separate target from features
y = train['target']
features = train.drop(['target'], axis=1)

# List of features for later use
feature_list = list(features.columns)

# Preview features
features.head()

# %% [markdown]
# Now, we'll need to handle the categorical columns (`cat0`, `cat1`, ... `cat9`).  
# 
# From the lesson **[Categorical Variables lesson](https://www.kaggle.com/alexisbcook/categorical-variables)**, we'll use ordinal encoding and save our encoded features as new variables `X` and `X_test`.
# 
# #### What is Ordinal Encoder and why we choose it?
# In ordinal encoding, each unique category value is assigned an integer value. The encoding involves mapping each unique label to an integer value.
# For example, `red` is 1, `green` is 2, and `blue` is 3.
# 
# This is called an ordinal encoding or an integer encoding and is easily reversible. Often, integer values starting at zero are used. The integer values have a natural ordered relationship between each other and machine learning algorithms may be able to understand and harness this relationship.
# 
# There are three common approaches for converting categorical variables to numerical values. They are:
# 
# - Ordinal Encoding
# - One-Hot Encoding
# - Dummy Variable Encoding
# 
# But in here, we are using Ordinal because our dataset is naturally in order, alphabetically. Hence, when we encode it, the converted numerics consequently take a natural integral order like in its original form. Take a note of that in the following table. 
# 
# _Eg: 'A' -> 0.0, 'B' -> 1.0, 'C' -> 2.0,, and so on..._
# 
# But for categorical variables, where no such ordinal relationship exist, it may be advisable to use the other two. Nevertheless, for this dataset, ordinal encoding will suffice. But you can definitely use other encoding approaches as well.

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:46:49.571876Z","iopub.execute_input":"2021-08-27T21:46:49.572327Z","iopub.status.idle":"2021-08-27T21:46:53.681341Z","shell.execute_reply.started":"2021-08-27T21:46:49.572286Z","shell.execute_reply":"2021-08-27T21:46:53.680592Z"}}
# List of categorical columns
object_cols = [col for col in features.columns if 'cat' in col]

# ordinal-encode categorical columns
X = features.copy()
X_test = test.copy()

# ordinal encode input variables
ordinal_encoder = OrdinalEncoder()
X[object_cols] = ordinal_encoder.fit_transform(features[object_cols])
X_test[object_cols] = ordinal_encoder.transform(test[object_cols])

# Preview the ordinal-encoded features
X.head()

# %% [markdown]
# We can also prepare the target in the same manner. Ordinal encoding of target variable uses LabelEncoder(). But we do not use it here since it is used to normalize labels, and to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels. Since our target is already numeric, we do not need it here.
# 
# Next, we break off a validation set from the training data.

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:46:53.682443Z","iopub.execute_input":"2021-08-27T21:46:53.682873Z","iopub.status.idle":"2021-08-27T21:46:53.814395Z","shell.execute_reply.started":"2021-08-27T21:46:53.682833Z","shell.execute_reply":"2021-08-27T21:46:53.813661Z"}}
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42, test_size=0.2)

# %% [markdown]
# # Step 5: Train a model
# 
# Now that the data is prepared, the next step is to train a model.  
# 
# ## ✔ 5.1: Random Forest
# 
# From the lesson **[Random Forests](https://www.kaggle.com/dansbecker/random-forests)**, we learnt how to fit a random forest model to the data.
# 
# #### What is Random Forest, and why are we using it here?
# 
# Random forest is a type of supervised learning algorithm that uses ensemble methods (bagging) to solve both regression and classification problems. The algorithm operates by constructing a multitude of decision trees at training time and outputting the mean/mode of prediction of the individual trees.
# 
# - Each tree is created from a different sample of rows and at each node, a different sample of features is selected for splitting. 
# - Each of the trees makes its own individual prediction. 
# - These predictions are then averaged to produce a single result. 
# 
# ![Random Forest](https://upload.wikimedia.org/wikipedia/commons/7/76/Random_forest_diagram_complete.png)
# 
# The averaging makes a Random Forest better than a single Decision Tree hence improves its accuracy and reduces overfitting. 
# 
# A prediction from the Random Forest Regressor is an average of the predictions produced by the trees in the forest. 

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:46:53.815614Z","iopub.execute_input":"2021-08-27T21:46:53.816132Z","iopub.status.idle":"2021-08-27T21:50:59.661828Z","shell.execute_reply.started":"2021-08-27T21:46:53.816080Z","shell.execute_reply":"2021-08-27T21:50:59.660869Z"}}
# Define the model 

print("training RandomForestRegressor model...")
model_rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=30,
           max_features='sqrt', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=2, min_samples_split=5,
           min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=-1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
# Train the model
model_rf.fit(X_train, y_train)
preds_valid = model_rf.predict(X_valid)
print(mean_squared_error(y_valid, preds_valid, squared=False))

# %% [markdown]
# In the code cell above, we set `squared=False` to get the root mean squared error (RMSE) on the validation data.

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:50:59.665720Z","iopub.execute_input":"2021-08-27T21:50:59.666030Z","iopub.status.idle":"2021-08-27T21:51:26.827539Z","shell.execute_reply.started":"2021-08-27T21:50:59.666001Z","shell.execute_reply":"2021-08-27T21:51:26.826513Z"}}
plt.rcParams["axes.labelsize"] = 12
rf_prob_train = model_rf.predict(X_train) - y_train
plt.figure(figsize=(6,6))
sp.stats.probplot(rf_prob_train, plot=plt, fit=True)
plt.title('Train Probability Plot for Random Forest', fontsize=10)
plt.show()

rf_prob_test = model_rf.predict(X_valid) - y_valid
plt.figure(figsize=(6,6))
sp.stats.probplot(rf_prob_test, plot=plt, fit=True)
plt.title('Test Probability Plot for Random Forest', fontsize=10)
plt.show()

# %% [markdown]
# But when the Random Forest Regressor is tasked with the problem of predicting for values not previously seen, it will always predict an average of the values seen previously. Meaning, that the Random Forest Regressor is unable to discover trends that would enable it in extrapolating values that fall outside the training set. 
# 
# The random forest performs implicit feature selection because it splits nodes on the most important variables, but other machine learning models do not. One approach to improve the models is therefore to use the random forest feature importances to reduce the number of variables in the problem. In our case, we will use the feature importances to decrease the number of features for our random forest model, because, in addition to potentially increasing performance, reducing the number of features will shorten the run time of the model. Let's see if it improves the accuracy of our model.

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2021-08-27T21:51:26.829479Z","iopub.execute_input":"2021-08-27T21:51:26.829768Z","iopub.status.idle":"2021-08-27T21:51:27.118580Z","shell.execute_reply.started":"2021-08-27T21:51:26.829739Z","shell.execute_reply":"2021-08-27T21:51:27.117559Z"}}
# Get numerical feature importances

print("getting feature importance for RandomForestRegressor model...")
importances = list(model_rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# %% [markdown]
# These stats definitely prove that some variables are much more important to our problem than others! Given that there are so many variables with zero importance (or near-zero due to rounding). But it seems like we should be able to get rid of some of them without impacting performance.
# 
# The following graph represents the relative differences in feature importances.

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:51:27.119848Z","iopub.execute_input":"2021-08-27T21:51:27.120208Z","iopub.status.idle":"2021-08-27T21:51:27.447519Z","shell.execute_reply.started":"2021-08-27T21:51:27.120169Z","shell.execute_reply":"2021-08-27T21:51:27.446415Z"}}
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

# %% [markdown]
# Consequently, we can also make a cumulative importance graph that shows the contribution to the overall importance of each additional variable. The dashed line is drawn at 95% of total importance accounted for.

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:51:27.448788Z","iopub.execute_input":"2021-08-27T21:51:27.449088Z","iopub.status.idle":"2021-08-27T21:51:27.737848Z","shell.execute_reply.started":"2021-08-27T21:51:27.449062Z","shell.execute_reply":"2021-08-27T21:51:27.736723Z"}}
# List of features sorted from most to least important
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]
# Cumulative importances
cumulative_importances = np.cumsum(sorted_importances)
# Make a line graph
plt.plot(x_values, cumulative_importances, 'g-')
# Draw line at 95% of importance retained
plt.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
# Format x ticks and labels
plt.xticks(x_values, sorted_features, rotation = 'vertical')
# Axis labels and title
plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances');

# %% [markdown]
# We can now use this to remove unimportant features. 95% is an arbitrary threshold, but if it leads to noticeably poor performance we can adjust the value. First, we need to find the exact number of features to exceed 95% importance:

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:51:27.739095Z","iopub.execute_input":"2021-08-27T21:51:27.739397Z","iopub.status.idle":"2021-08-27T21:51:27.746137Z","shell.execute_reply.started":"2021-08-27T21:51:27.739369Z","shell.execute_reply":"2021-08-27T21:51:27.744673Z"}}
# Find number of features for cumulative importance of 95%
# Add 1 because Python is zero-indexed
print('Number of features for 95% importance:', np.where(cumulative_importances > 0.95)[0][0] + 1)

# %% [markdown]
# We see that almost all of our features are important. But we can check that further by creating a new training and testing set retaining only the 18 most important features, and using the model on it.

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:51:27.748075Z","iopub.execute_input":"2021-08-27T21:51:27.748544Z","iopub.status.idle":"2021-08-27T21:51:27.791344Z","shell.execute_reply.started":"2021-08-27T21:51:27.748501Z","shell.execute_reply":"2021-08-27T21:51:27.789851Z"}}
# Extract the names of the most important features
important_feature_names = [feature[0] for feature in feature_importances[0:17]]
# Create training and testing sets with only the important features
important_train_features = X_train[important_feature_names]
important_test_features = X_valid[important_feature_names]
# Sanity check on operations
print('Important train features shape:', important_train_features.shape)
print('Important test features shape:', important_test_features.shape)

# %% [markdown] {"execution":{"iopub.status.busy":"2021-08-17T21:17:19.876066Z","iopub.execute_input":"2021-08-17T21:17:19.876495Z","iopub.status.idle":"2021-08-17T21:17:19.884045Z","shell.execute_reply.started":"2021-08-17T21:17:19.876461Z","shell.execute_reply":"2021-08-17T21:17:19.881952Z"}}
# We first test the accuracy with the selected features.

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:51:27.793183Z","iopub.execute_input":"2021-08-27T21:51:27.793607Z","iopub.status.idle":"2021-08-27T21:56:37.240014Z","shell.execute_reply.started":"2021-08-27T21:51:27.793559Z","shell.execute_reply":"2021-08-27T21:56:37.239067Z"}}
# Train the expanded model on only the important features
model_rf.fit(important_train_features, y_train);
# Make predictions on test data
new_predictions = model_rf.predict(important_test_features)
# Performance metrics
errors = mean_squared_error(y_valid, new_predictions, squared=False)
print('RMSE:', errors)
# Calculate rmse
rmse = np.mean(100 * (errors / y_valid))
# Calculate and display accuracy
accuracy = 100 - rmse
print('Accuracy:', round(accuracy, 2), '%.')

# %% [markdown]
# Now with the original features. _(We are re-training it back to fit the original features)_

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:56:37.241240Z","iopub.execute_input":"2021-08-27T21:56:37.241543Z","iopub.status.idle":"2021-08-27T22:00:37.903808Z","shell.execute_reply.started":"2021-08-27T21:56:37.241511Z","shell.execute_reply":"2021-08-27T22:00:37.902800Z"}}
# Train the model first on the original features
model_rf.fit(X_train, y_train);
# Make predictions on test data
predictions = model_rf.predict(X_valid)
# Performance metrics
errors = mean_squared_error(y_valid, predictions, squared=False)
print('Metrics for Random Forest Trained on original Data')
print('RMSE:', errors)
# Calculate rmse
rmse_new = np.mean(100 * (errors / y_valid))
# Calculate and display accuracy
accuracy = 100 - rmse_new
print('Accuracy:', round(accuracy, 2), '%.')

# %% [markdown]
# This is just an instance of feature selection with one of our models. But since we do not see any improvement, we move forward without the selected features for our dataset.
# 
# Let's check the feature importances for the model. 

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T22:00:37.906887Z","iopub.execute_input":"2021-08-27T22:00:37.907266Z","iopub.status.idle":"2021-08-27T22:00:38.408207Z","shell.execute_reply.started":"2021-08-27T22:00:37.907235Z","shell.execute_reply":"2021-08-27T22:00:38.406893Z"}}
pd.Series(model_rf.feature_importances_, index = X_train.columns).nlargest(10).plot(kind = 'barh',
                                                                               figsize = (6, 6),
                                                                              title = 'Feature importance from Random Forest').invert_yaxis();

# %% [markdown]
# Following the same pattern, we try it with on other models.
# 
# ## ✔ 5.2: XGBoost
# 
# From the lesson **[XGBoost](https://www.kaggle.com/alexisbcook/xgboost)**, we learnt how to fit a XGBoost model to the data.
# 
# #### What is XGBoost?
# XGBoost is termed as Extreme Gradient Boosting Algorithm which is again an ensemble method that works by boosting trees. XGboost makes use of a gradient descent algorithm which is the reason that it is called Gradient Boosting. The whole idea is to correct the previous mistake done by the model, learn from it and its next step improves the performance. The previous results are rectified and performance is enhanced.
# 
# ![XGBoost Regressor](https://miro.medium.com/max/1400/1*FLshv-wVDfu-i54OqvZdHg.png)
# 
# #### Why is it preferred?
# - Speed and perfoermance: Comparatively faster than other ensemble classifiers
# - Core algorithm is parallelizable: Because the core XGBoost algorithm is parallelizable it can harness the power of multi-core computers. It is also parallelizable onto GPU’s and across networks of computers making it feasible to train on very large datasets as well.
# - Consistently outperforms other algorithm methods : It has shown better performance on a variety of machine learning benchmark datasets.
# - Wide variety of tuning parameters : XGBoost internally has parameters for cross-validation, regularization, user-defined objective functions, missing values, tree parameters, scikit-learn compatible API etc.
# 
# 

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T22:00:38.409715Z","iopub.execute_input":"2021-08-27T22:00:38.410035Z"}}


print("training XGBoost model (Initial Settings)...")

xgb_params = {'objective': 'reg:squarederror',
              'n_estimators': 10000,
              'learning_rate': 0.036,
              'subsample': 0.926,
              'colsample_bytree': 0.118,
              'grow_policy':'lossguide',
              'max_depth': 3,
              'booster': 'gbtree', 
              'reg_lambda': 45.1,
              'reg_alpha': 34.9,
              'random_state': 42,
              'reg_lambda': 0.00087,
              'reg_alpha': 23.13181079976304,
              'n_jobs': -1}


model_XGB = XGBRegressor(**xgb_params)
model_XGB.fit(X_train, y_train) 
predictions_XGB = model_XGB.predict(X_valid)
print(mean_squared_error(y_valid, predictions_XGB, squared=False))

# %% [code]
plt.rcParams["axes.labelsize"] = 12
xgb_prob_train = model_XGB.predict(X_train) - y_train
plt.figure(figsize=(6,6))
sp.stats.probplot(xgb_prob_train, plot=plt, fit=True)
plt.title('Train Probability Plot for XGBoost', fontsize=10)
plt.show()

xgb_prob_test = model_XGB.predict(X_valid) - y_valid
plt.figure(figsize=(6,6))
sp.stats.probplot(xgb_prob_test, plot=plt, fit=True)
plt.title('Test Probability Plot for XGBoost', fontsize=10)
plt.show()

# %% [markdown]
# Comparing the feature importances as before, we see for XGBoost, the outcome is slightly different than Random Forest.

# %% [code]
pd.Series(model_XGB.feature_importances_, index = X_train.columns).nlargest(10).plot(kind = 'barh',
                                                                               figsize = (6, 6),
                                                                              title = 'Feature importance from XGBoost').invert_yaxis();


# %% [markdown]
# Now, let's train our dataset in XGBoost with other different parameters.

# %% [code]
xgb_1_params = {
    'n_estimators': 5000,
    'booster': 'gbtree', 
    'random_state':40,
    'learning_rate': 0.07853392035787837,
    'reg_lambda': 1.7549293092194938e-05,
    'reg_alpha': 14.68267919457715, 
    'subsample': 0.8031450486786944, 
    'colsample_bytree': 0.170759104940733, 
    'max_depth': 3
    }

model_XGB_1 = XGBRegressor(**xgb_1_params)
model_XGB_1.fit(X_train, y_train) 
predictions_XGB_1 = model_XGB_1.predict(X_valid)
print(mean_squared_error(y_valid, predictions_XGB_1, squared=False))

# %% [code]
xgb_2_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 5000,
    'learning_rate': 0.12,
    'subsample': 0.96,
    'colsample_bytree': 0.12,
    'max_depth': 2,
    'booster': 'gbtree', 
    'reg_lambda': 65.1,
    'reg_alpha': 15.9,
    'random_state':40
}

model_XGB_2 = XGBRegressor(**xgb_2_params)
model_XGB_2.fit(X_train, y_train) 
predictions_XGB_2 = model_XGB_2.predict(X_valid)
print(mean_squared_error(y_valid, predictions_XGB_2, squared=False))

# %% [code]
xgb_3_params = {
    'objective': 'reg:squarederror',
    "learning_rate" : 0.05,
    'n_estimators': 5000,
    "max_depth":12,
    "min_child_weight" :110,
    "gamma" :0.01,
    'booster': 'gbtree', 
    "subsample" : 0.7,
    "colsample_bytree" : 0.1,
    "reg_lambda" :65,
    "reg_alpha":71,
    "max_delta_step":10}

model_XGB_3 = XGBRegressor(**xgb_3_params)
model_XGB_3.fit(X_train, y_train) 
predictions_XGB_3 = model_XGB_3.predict(X_valid)
print(mean_squared_error(y_valid, predictions_XGB_3, squared=False))

# %% [code]
xgb_4_params = {
    'random_state': 1, 
    'n_jobs': 4,
    'booster': 'gbtree',
    'n_estimators': 10000,
    'learning_rate': 0.03628302216953097,
    'reg_lambda': 0.0008746338866473539,
    'reg_alpha': 23.13181079976304,
    'subsample': 0.7875490025178415,
    'colsample_bytree': 0.11807135201147481,
    'max_depth': 3}

model_XGB_4 = XGBRegressor(**xgb_4_params)
model_XGB_4.fit(X_train, y_train) 
predictions_XGB_4 = model_XGB_4.predict(X_valid)
print(mean_squared_error(y_valid, predictions_XGB_4, squared=False))

# %% [code]
xgb_5_params = {'learning_rate': 0.07853392035787837, 
          'reg_lambda': 1.7549293092194938e-05, 
          'reg_alpha': 14.68267919457715, 
          'subsample': 0.8031450486786944, 
          'colsample_bytree': 0.170759104940733, 
          'max_depth': 3,
          'n_estimators': 5000
         }
model_XGB_5 = XGBRegressor(**xgb_5_params)
model_XGB_5.fit(X_train, y_train) 
predictions_XGB_5 = model_XGB_5.predict(X_valid)
print(mean_squared_error(y_valid, predictions_XGB_5, squared=False))

# %% [markdown]
# ## ✔ 5.3: LightGBM
# 
# Light GBM is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithm, used for ranking, classification and many other machine learning tasks. Since it is based on decision tree algorithms, it splits the tree leaf wise with the best fit whereas other boosting algorithms split the tree depth wise or level wise rather than leaf-wise. So when growing on the same leaf in Light GBM, the leaf-wise algorithm can reduce more loss than the level-wise algorithm and hence results in much better accuracy which can rarely be achieved by any of the existing boosting algorithms. Also, it is surprisingly very fast, hence the word ‘Light’.
# 
# ![LightGBM](https://miro.medium.com/max/1400/1*mKkwlQF25Rq1ilne5UiEXA.png)
# 
# #### Advantages of Light GBM
# - Faster training speed and higher efficiency: Light GBM use histogram based algorithm i.e it buckets continuous feature values into discrete bins which fasten the training procedure.
# - Lower memory usage: Replaces continuous values to discrete bins which result in lower memory usage.
# - Better accuracy than any other boosting algorithm: It produces much more complex trees by following leaf wise split approach rather than a level-wise approach which is the main factor in achieving higher accuracy. However, it can sometimes lead to overfitting which can be avoided by setting the max_depth parameter.
# - Compatibility with Large Datasets: It is capable of performing equally good with large datasets with a significant reduction in training time as compared to XGBOOST.
# - Parallel learning supported.

# %% [code]
lgbm_parameters = {
    'metric': 'RMSE',
    'feature_pre_filter': False,
    'lambda_l1': 0.45,
    'lambda_l2': 4.8,
    'learning_rate': 0.005,
    'num_trees': 80000,
    'num_leaves': 10, 
    'feature_fraction': 0.4, 
    'bagging_fraction': 1.0, 
    'bagging_freq': 0, 
    'min_child_samples': 100,
    'num_threads': 4
}

lgbm_model = LGBMRegressor(**lgbm_parameters)
lgbm_model.fit(X_train, y_train, eval_set = ((X_valid,y_valid)),verbose = -1, early_stopping_rounds = 1000,categorical_feature=object_cols) 
predictions_LGBM = lgbm_model.predict(X_valid)
print(mean_squared_error(y_valid, predictions_LGBM, squared=False))

# %% [markdown]
# As we can see, it gives a better result than XGBoost. LightGBM, apart from being more accurate and time-saving than XGBOOST has been limited in usage due to less documentation available. However, this algorithm has shown far better results and has outperformed existing boosting algorithms.
# 
# ![Comparison](https://image.slidesharecdn.com/xgboostandlightgbm-180201121028/95/xgboost-lightgbm-21-638.jpg?cb=1517487076)
# 
# Following are the probability plot of LGBM and the feature importance for the model.

# %% [code]
plt.rcParams["axes.labelsize"] = 12
lgbm_prob_train = lgbm_model.predict(X_train) - y_train
plt.figure(figsize=(6,6))
sp.stats.probplot(lgbm_prob_train, plot=plt, fit=True)
plt.title('Train Probability Plot for LGBM', fontsize=10)
plt.show()

lgbm_prob_test = lgbm_model.predict(X_valid) - y_valid
plt.figure(figsize=(6,6))
sp.stats.probplot(lgbm_prob_test, plot=plt, fit=True)
plt.title('Test Probability Plot for LGBM', fontsize=10)
plt.show()

# %% [code]
pd.Series(lgbm_model.feature_importances_, index = X_train.columns).nlargest(10).plot(kind = 'barh',
                                                                               figsize = (6, 6),
                                                                              title = 'Feature importance from LGBM').invert_yaxis();


# %% [markdown]
# Training with varied parameters of LGBM model.

# %% [code]
lgbm_parameters_1 = {
    'metric': 'RMSE',
    'feature_pre_filter': False,
    'reg_alpha': 0.4972562469417825, 
    'reg_lambda': 0.3273637203281044, 
    'num_leaves': 50, 
    'learning_rate': 0.032108486615557354,                      
    'max_depth': 40,                     
    'n_estimators': 4060, 
    'min_child_weight': 0.0173353329222102,
    'subsample': 0.9493343850444064, 
    'colsample_bytree': 0.5328221263825876, 
    'min_child_samples': 80
}

lgbm_model_1 = LGBMRegressor(**lgbm_parameters_1)
lgbm_model_1.fit(X_train, y_train, eval_set = ((X_valid,y_valid)),verbose = -1, early_stopping_rounds = 1000,categorical_feature=object_cols) 
predictions_LGBM_1 = lgbm_model_1.predict(X_valid)
print(mean_squared_error(y_valid, predictions_LGBM_1, squared=False))

# %% [code]
lgbm_parameters_2 = {
    'metric': 'RMSE',
    'feature_pre_filter': False,
    'reg_alpha': 0.4994758073847213, 
    'reg_lambda': 0.32496035638807086, 
    'num_leaves': 55, 
    'learning_rate': 0.03292764050310852, 
    'max_depth': 32, 
    'n_estimators': 6059, 
    'min_child_weight': 0.018085927063358823, 
    'subsample': 0.9553223859131216, 
    'colsample_bytree': 0.5253243484788512, 
    'min_child_samples': 77
}

lgbm_model_2 = LGBMRegressor(**lgbm_parameters_2)
lgbm_model_2.fit(X_train, y_train, eval_set = ((X_valid,y_valid)),verbose = -1, early_stopping_rounds = 2000,categorical_feature=object_cols) 
predictions_LGBM_2 = lgbm_model_2.predict(X_valid)
print(mean_squared_error(y_valid, predictions_LGBM_2, squared=False))

# %% [markdown]
# ## ✔ 5.4: CatBoost
# CatBoost is an algorithm for gradient boosting on decision trees. And it is the only boosting algorithm with very less prediction time. Because of its symmetric tree structure. It is comparatively 8x faster than XGBoost while predicting. One main difference between CatBoost and other boosting algorithms is that the CatBoost implements symmetric trees. Though when datasets have many numerical features (like this one), CatBoost takes so much time to train than Light GBM.
# 
# But CatBoost also offers an idiosyncratic way of handling categorical data, requiring a minimum of categorical feature transformation, opposed to the majority of other machine learning algorithms, that cannot handle non-numeric values. From a feature engineering perspective, the transformation from a non-numeric state to numeric values can be a very non-trivial and tedious task, and CatBoost makes this step obsolete. (Though here, we had used feature engineering for conducting training in all the models)
# 
# ![CatBoost](https://www.kdnuggets.com/wp-content/uploads/mwiti-catboost-0.png)

# %% [code]
cat_parameters_1 = {    
    'iterations':1600,
    'learning_rate':0.024,
    'l2_leaf_reg':20,
    'random_strength':1.5,
    'grow_policy':'Depthwise',
    'leaf_estimation_method':'Newton', 
    'bootstrap_type':'Bernoulli',
    'thread_count':4,
    'verbose':False,
    'loss_function':'RMSE',
    'eval_metric':'RMSE',
    'od_type':'Iter'
}

cat_model_1 = CatBoostRegressor(**cat_parameters_1)
cat_model_1.fit(X_train, y_train, verbose =200) 
predictions_cat_1 = cat_model_1.predict(X_valid)
print(mean_squared_error(y_valid, predictions_cat_1, squared=False))

# %% [code]
pd.Series(cat_model_1.feature_importances_, index = X_train.columns).nlargest(10).plot(kind = 'barh',
                                                                               figsize = (6, 6),
                                                                              title = 'Feature importance from LGBM').invert_yaxis();


# %% [markdown]
# # Step 6: Feature importances
# Now, let's compare all of the models together along with their feature importances in the following graph.

# %% [code]
a1 = model_rf.feature_importances_
a2 = model_XGB.feature_importances_
a3 = lgbm_model.feature_importances_
a4 = cat_model_1.feature_importances_

axis_x  = X.columns.values
axis_y1 = minmax_scaling(a1, columns=[0])
axis_y2 = minmax_scaling(a2, columns=[0])
axis_y3 = minmax_scaling(a3, columns=[0])
axis_y4 = minmax_scaling(a4, columns=[0])

plt.style.use('seaborn-whitegrid') 
plt.figure(figsize=(16, 6), facecolor='lightgray')
plt.title(f'\nF e a t u r e   I m p o r t a n c e s\n', fontsize=14)  

plt.scatter(axis_x, axis_y1, s=20, label='Random Forest') 
plt.scatter(axis_x, axis_y2, s=20, label='XGBoost')
plt.scatter(axis_x, axis_y3, s=20, label='Light GBM') 
plt.scatter(axis_x, axis_y4, s=20, label='CatBoost')

plt.legend(fontsize=12, loc=2)
plt.show()

# %% [markdown]
# #### Why are we using only one of the models of each regressor to check the most important features for those models?
# That's because the feature importances depend on the models, not on the parameters. We can check with any variations of the parameters for each of the regressors.
# 
# # Step 7: Submit to the competition
# 
# We'll begin by using the trained model to generate predictions, which we'll save to a CSV file.
# 
# ### But which model to use for prediction?
# 
# Maybe we should try to use all of the models we trained! Here is a fun way to put weightage to the predictions of all the models to get the final outcome. 
# 
# *NOTE:* Remember to put more weightage on the models with lesser RMSE value to get the best result. 
# 
# ### Ensemble
# Try ensembling. Using model blending weights optimisation technique similar to the one used in [this](https://www.kaggle.com/gogo827jz/optimise-blending-weights-with-bonus-0) notebook, we have tried it on our models. Ensembling different models can be necessary to improve scores.

# %% [code]
# Use the models to generate predictions
pred_1 = model_rf.predict(X_test)
pred_2 = model_XGB.predict(X_test)
pred_3 = model_XGB_1.predict(X_test)
pred_4 = model_XGB_2.predict(X_test)
pred_5 = model_XGB_3.predict(X_test)
pred_6 = model_XGB_4.predict(X_test)
pred_7 = model_XGB_5.predict(X_test)
pred_8 = lgbm_model.predict(X_test)
pred_9 = lgbm_model_1.predict(X_test)
pred_10= lgbm_model_2.predict(X_test)
pred_11 = cat_model_1.predict(X_test)

# Make sure to check that the weights sum up to ~1 by averaging the weights later 
preds = [pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9, pred_10, pred_11]
weights = [0., 10000., 100., 50., 20000., 100., 10., 9., 5., 1., 1.] 

sample_submission = pd.read_csv("../input/30-days-of-ml/sample_submission.csv")
sample_submission.target = 0.0

for pred, weight in zip(preds, weights):
    sample_submission.target += weight * pred / sum(weights)

sample_submission.to_csv('submission.csv', index=False)

# %% [code]
sample_submission.head()

# %% [markdown]
# # Step 8: Extras
# 
# A few of the linear regression models were further used for experimentation:
# 
# - **Linear Regression** : Linear regression is a linear model, e.g. a model that assumes a linear relationship between the input variables (x) and the single output variable (y). More specifically, that y can be calculated from a linear combination of the input variables (x).
# - **Ridge Regression** : Ridge regression is a model tuning method that is used to analyse any data that suffers from multicollinearity. This method performs L2 regularization. 
# - **LASSO Regression** : Lasso regression is a regularization technique. It is used over regression methods for a more accurate prediction. This model uses shrinkage. Shrinkage is where data values are shrunk towards a central point as the mean. The lasso procedure encourages simple, sparse models (i.e. models with fewer parameters).
# - **Elastic Net** : It is a linear regression model trained with L1 and L2 prior as regularizer. This combination allows for learning a sparse model where few of the weights are non-zero like Lasso, while still maintaining the regularization properties of Ridge.
# - **Stochastic Gradient Descent** : Stochastic gradient descent is an optimization algorithm often used in machine learning applications to find the model parameters that correspond to the best fit between predicted and actual outputs.
# 
# But the RMSE scores from these were much higher compared to the previous models used in this notebook, so it was not used for the final submission. 

# %% [code]
from sklearn.linear_model import LinearRegression

model = LinearRegression(normalize=True)
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
print(mean_squared_error(y_valid, predictions, squared=False))

# %% [code]
from sklearn.linear_model import Ridge

model = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
print(mean_squared_error(y_valid, predictions, squared=False))

# %% [code]
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1, 
              precompute=True, 
#               warm_start=True, 
              positive=True, 
              selection='random',
              random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
print(mean_squared_error(y_valid, predictions, squared=False))

# %% [code]
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
print(mean_squared_error(y_valid, predictions, squared=False))

# %% [code]
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(n_iter_no_change=250, penalty=None, eta0=0.0001, max_iter=100000)
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
print(mean_squared_error(y_valid, predictions, squared=False))

# %% [markdown]
# ## Thank you so much for reading! Please do upvote if you liked it. :)