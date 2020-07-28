#!/usr/bin/env python
# coding: utf-8

# In[137]:


# import libraries
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

# read in data
df_diabetes = pd.read_csv('C:\\Users\\Kevin Sharp\\Desktop\\Springboard\\2014_BRFSS_encoded.csv')
df_diabetes.head()


# In[116]:


from sklearn.preprocessing import OneHotEncoder

nonordinals = df_diabetes.loc[:,['last_checkup', 'race', 'mscode', 'employed', 'marital', 'sleep_time',
                               'health_coverage', 'rent_or_own_home', 'smoker']].values

onehotencoder = OneHotEncoder(sparse=False)
X = onehotencoder.fit_transform(nonordinals)


# In[117]:


df_onehot = pd.DataFrame(X)
df_onehot.columns = ['checkup_1', 'checkup_2', 'checkup_5', 'checkup_5plus', 'checkup_never', 
                     'White','Black','American Indian/Alaska Native','Asian','Hawaiian/Pacific Islander','Other (race)','Multiracial','Hispanic',
                     'MSA_center', 'MSA_outer', 'MSA_suburb', 'MSA_none',
                     'employed_wage', 'employed_self', 'unemployed_>1', 'unemployed_<1', 'homemaker', 'student', 'retired', 'unable to work',
                     'Married', 'Divorced', 'Widowed', 'Separated', 'Never_married', 'Unmarried_couple',
                     'sleep_low', 'sleep_normal', 'sleep_high',
                     'Plan from employer/union','Plan from marketplace','Medicare','Medicaid/state program','TRICARE (formerly CHAMPUS), VA, or Military','Alaska Native, Indian Health Service, Tribal Health Services',
                     'Other (ins)', 'None',
                     'Own', 'Rent', 'Other (home)',
                     'Smokes every day', 'Smokes some days','Former smoker', 'Never smoked'
                    ]
df_onehot


# In[118]:


df_diabetes = df_diabetes.drop(['last_checkup', 'race', 'mscode', 'employed', 'marital', 'sleep_time',
                               'health_coverage', 'rent_or_own_home', 'smoker'], axis=1)
df_diabetes = pd.concat([df_diabetes, df_onehot], axis=1)


# In[119]:


df_diabetes['general_health'] = df_diabetes['general_health'] - 1
df_diabetes['10yr_age_group'] = df_diabetes['10yr_age_group'] - 1
df_diabetes['income'] = df_diabetes['income'].replace(2,0)
df_diabetes['flushot'] = df_diabetes['flushot'].replace(2,0)
df_diabetes['sex'] = df_diabetes['sex'].replace(2,0)
df_diabetes['education'] = df_diabetes['education'] - 1
df_diabetes['angina_coronary_heart_disease'] = df_diabetes['angina_coronary_heart_disease'].replace(2,0)
df_diabetes['mental_health_days_per_month'] = df_diabetes['mental_health_days_per_month'] - 1
df_diabetes['kidney_disease'] = df_diabetes['kidney_disease'].replace(2,0)
df_diabetes['depressive_disorder'] = df_diabetes['depressive_disorder'].replace(2,0)
df_diabetes['uses_medical_equipment'] = df_diabetes['uses_medical_equipment'].replace(2,0)
df_diabetes['any_exercise'] = df_diabetes['any_exercise'].replace(2,0)
df_diabetes['blindness'] = df_diabetes['blindness'].replace(2,0)
df_diabetes['trouble_concentrating'] = df_diabetes['trouble_concentrating'].replace(2,0)


# In[120]:


df_diabetes


# For this project, I will be build logistic regression model to predict the risk of diabetes in responants to the 2014 BRFSS. Those with diabetes make up a minority of both the general population and the sample data, which will affect how I approach contructing the model.

# In[121]:


#"no" response was encoded as 3; change these to 0 for clarity
df_diabetes.loc[(df_diabetes['diabetes'] == 3), 'diabetes'] = 0

#verify unbalanced nature of the data
df_diabetes['diabetes'].value_counts()/df_diabetes.shape[0]


# First, we split the data into training and testing sets. I use the default 25/75 ratio of training data to testing data.

# In[122]:


#create train-test split
from sklearn.model_selection import train_test_split

X = df_diabetes.drop('diabetes', axis=1)
y = df_diabetes['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)


# In[123]:


from yellowbrick.classifier import ClassificationReport
from sklearn.model_selection import cross_val_score

def plot_classification_report(model, version_number):
    visualizer = ClassificationReport(model, classes=['diabetes-negative', 'diabetes-positive'], support=False,
                                     title=f'Logistic Regression Classification Report {version_number}')

    visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()                       # Finalize and show the figure
    
    y_pred = model.predict(X_test)
    score = np.mean(cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=5))
    print(f'roc_auc score: {score}')
    
    score = np.mean(cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5))
    print(f'accuracy score: {score}')


# Initially, I use a basic logistic regression model with no additional parameters to use as a control for the tuning I expect to perform at a later step. As a reminder, 0 indicates a diabetes negative response, while 1 indicates a diabetes positive response.

# In[124]:


#establishing baseline naive model to use as a control for further tuning

model = LogisticRegression(max_iter=1000)
#model.fit(X_train, y_train)

plot_classification_report(model, 1)


# The printed table of metrics shows us the effect of modeling unbalanced data. Although the model's accuracy is a fine 85%, we find a recall of only 16% for those with diabetes; in other words, the remaining 84% of those with diabetes were reported as false negatives by the model, which has a strong negative effect on the $f_1$ score. Since we are primarily interested in identifying those with diabetes, I will retune the model to assign class weights proportional to their frequency in the data.

# In[125]:


model = LogisticRegression(max_iter=1000, class_weight='balanced')
#model.fit(X_train, y_train)

plot_classification_report(model, 2)


# Already, we see a marked improvement in the rate of recall and the $f_1$ score for diabetes-positive, and the ROC AUC score remains about the same at roughly 0.80. Although the $f_1$ score for diabetes-negative drops slightly, I more strongly consider regard the increased performance on diabetes-positive and consider this a net improvement. I will next attempt to tune the model further by adjusting the regularization parameter $C$. I also provide a range of possible weights to allow the model more flexibility.

# In[126]:


#create hyperparameter grid for grid search CV 
from sklearn.model_selection import GridSearchCV

Cs = [0.001, 0.1, 1, 10, 100]
Ws = [{1:84.4994, 0:15.5006}, {1:84, 0:16}, {1:80, 0:20}, {1:75, 0:25}]
hyperparam_grid = {"C":Cs, "class_weight":Ws}


# In[127]:


model = LogisticRegression(max_iter=1000)

grid = GridSearchCV(model, hyperparam_grid, scoring='f1_weighted')
grid.fit(X_train, y_train)
print(f'best score: {grid.best_score_}\nbest parameters: {grid.best_params_}')


# In[128]:


C = grid.best_params_['C']
class_weight = grid.best_params_['class_weight']

model = LogisticRegression(max_iter=1000, class_weight=class_weight)
#model.fit(X_train, y_train)
plot_classification_report(model, 3)


# This updated model uses a value of 100 for $C$ instead of the default 1 and adjusts the class weights a bit. The ROC AUC score is once again about 0.80, and accuracy is increased from 72% to 76%. Both $f_1$ scores are slightly higher in this version of the model compared to the version that considers class weights but not the regularization parameter. I believe this version of the model is sufficient to move forward with a final analysis of the predictor variables.

# In[129]:


df_coef = pd.DataFrame()
df_coef['coefficients'] = model.coef_[0]
df_coef['odds_ratios'] = np.exp(df_coef['coefficients'])
df_coef['features'] = X.columns
df_coef = df_coef.sort_values(['odds_ratios'])


# In[135]:


fig = plt.figure(figsize=(20,10))

plt.bar(df_coef['features'][-20:], df_coef['odds_ratios'][-20:])
plt.xticks(rotation=90, fontsize=16)
plt.title("Odds ratios of the Logistic Regression Model", fontsize=24)
plt.xlabel("features", fontsize=20)
plt.ylabel("OR value", fontsize=16)
plt.show()


# In[148]:


df_checkup = df_coef.loc[df_coef['features'].isin(['checkup_1', 'checkup_2', 'checkup_5', 'checkup_5plus', 'checkup_never'])]
plt.bar(df_checkup['features'], df_checkup['odds_ratios'])
plt.xticks(rotation=90, fontsize=16)
plt.title("Odds Ratios for Years Since Last Checkup", fontsize=24)
plt.xlabel("features", fontsize=20)
plt.ylabel("OR value", fontsize=16)
plt.show()


# In[153]:


df_race = df_coef.loc[df_coef['features'].isin(['White','Black','American Indian/Alaska Native','Asian','Hawaiian/Pacific Islander','Other (race)','Multiracial','Hispanic'])]
plt.bar(df_race['features'], df_race['odds_ratios'])
plt.xticks(rotation=90, fontsize=16)
plt.title("Odds ratios for Race", fontsize=24)
plt.xlabel("features", fontsize=20)
plt.ylabel("OR value", fontsize=16)
plt.show()


# Conclusions
# ---
# 
# As a reminder, the model itself does not establish the direction of causality between the predictors and the incidence of diabetes; conclusions about causality are derived from general medical knowledge about the causes and outcomes associated with diabetes.
# 
# Based on the results from the final version of thge model, the four strongest predictors for diabetes are `bmi_category`, `kidney_disease`, `general_health`, and `angina_coronary_heart_disease`. Predictably, as a person's weight increases and their general health worsens, do does their risk of diabetes. Kidney disease, angina, and coronary heart disease in general are negative health outcomes shown to be strong predictors of a prior onset of diabetes. Finally, as discussed in the section on exploratory data analysis, those with diabetes are perhaps more likely to vist their doctor frequently in order to monitor their condition.

# In[ ]:




