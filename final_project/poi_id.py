
# coding: utf-8

# In[2]:

#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")
import numpy as np
import pandas as pd
import matplotlib.pyplot
from sklearn.preprocessing import Imputer
from feature_format import featureFormat, targetFeatureSplit
import tester
from tester import dump_classifier_and_data

### features selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

### Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

### tuning the model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# In[3]:

### Define the various feature lists needed

# all the intital featurs in dataset except for email address
initial_features_list = ['poi',
                'salary',
                'bonus', 
                'long_term_incentive', 
                'deferred_income', 
                'deferral_payments',
                'loan_advances', 
                'other',
                'expenses', 
                'director_fees',
                'total_payments',
                'exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred',
                'total_stock_value',
                'to_messages',
                'from_messages',
                'from_this_person_to_poi',
                'from_poi_to_this_person']
# non-stock payments
payment_feature_list = ['salary',
            'bonus', 
            'long_term_incentive', 
            'deferred_income', 
            'deferral_payments',
            'loan_advances', 
            'other',
            'expenses', 
            'director_fees']

# stock payments
stock_feature_list = ['exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred']

# to/from email frequency features
email_feature_list = [  'to_messages',
                'from_messages',
                'from_this_person_to_poi',
                'from_poi_to_this_person']

# features list for the model
features_list = ['poi', 'salary']

# In[4]:
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

my_dataset = data_dict

### Exploring the dataset: 146 people, 21 features
print "Number of people in the dataset: {0}".format(len(my_dataset))
print "Number of features per person in the dataset: {0}".format(len(my_dataset.values()[0]))
print "features included in the dataset: " 
my_dataset.values()[0].keys()

# In[6]:
### Transform data from dictionary to the Pandas DataFrame
df = pd.DataFrame.from_dict(data_dict, orient = 'index')

#Order columns in DataFrame, excluding email
df = df[initial_features_list]
df = df.replace('NaN', np.nan)

print "total NaN values in the dataset: {0}" .format (df.isnull().sum().sum())
print "total values in the datase: {0}" .format (sum (df.count () + df.isnull().sum().sum()))
print " total number of rows with NaN values: {0}"\
.format (sum([True for idx,row in df.iterrows() if any(row.isnull())]))
df.info()

# In[7]:
### impute the "NaN" values

# Replace "NaN" values in financial dataset with 0
df.iloc [:,:15] = df.iloc [:,:15].fillna(0)

# Replace "NaN" values in email feautre set with median values for two categories POI = 1 and POI = 0
email_features = ['to_messages', 'from_messages', 'from_this_person_to_poi', 'from_poi_to_this_person']
imp = Imputer(missing_values='NaN', strategy='median', axis=0)

df.loc[df[df.poi == 1].index,email_features] = imp.fit_transform(df[email_features][df.poi == 1])
df.loc[df[df.poi == 0].index,email_features] = imp.fit_transform(df[email_features][df.poi == 0])

# Reviewing to see if we still have NaN values
print "total NaN values in the dataset is {0}" .format (df.isnull().sum().sum())
df.info()

# In[9]:

### Task 2: Data cleansing: check for typos and miscalculations

# Task 2.1: check for the accuracy of money payments data: summing payments features and comparing with total_payments
df[df[payment_feature_list].sum(axis='columns') != df.total_payments]

# Task 2.2: check for the accuracy of stock payments: summing stock payments and comparing with total_stock_value
df[df[stock_feature_list].sum(axis='columns') != df.total_stock_value]

# Task 2.3: Correct the errors for total_payments and total_stock_value based on the data in the PDF file
df.loc['BELFER ROBERT','total_payments'] = 3285
df.loc['BELFER ROBERT','deferral_payments'] = 0
df.loc['BELFER ROBERT','restricted_stock'] = 44093
df.loc['BELFER ROBERT','restricted_stock_deferred'] = -44093
df.loc['BELFER ROBERT','total_stock_value'] = 0
df.loc['BELFER ROBERT','director_fees'] = 102500
df.loc['BELFER ROBERT','deferred_income'] = -102500
df.loc['BELFER ROBERT','exercised_stock_options'] = 0
df.loc['BELFER ROBERT','expenses'] = 3285
df.loc['BELFER ROBERT',]
df.loc['BHATNAGAR SANJAY','expenses'] = 137864
df.loc['BHATNAGAR SANJAY','total_payments'] = 137864
df.loc['BHATNAGAR SANJAY','exercised_stock_options'] = 1.54563e+07
df.loc['BHATNAGAR SANJAY','restricted_stock'] = 2.60449e+06
df.loc['BHATNAGAR SANJAY','restricted_stock_deferred'] = -2.60449e+06
df.loc['BHATNAGAR SANJAY','other'] = 0
df.loc['BHATNAGAR SANJAY','director_fees'] = 0
df.loc['BHATNAGAR SANJAY','total_stock_value'] = 1.54563e+07
df.loc['BHATNAGAR SANJAY',]

# Reviewing to see if the totals are correct now
df[df[payment_feature_list].sum(axis='columns') != df.total_payments]
df[df[stock_feature_list].sum(axis='columns') != df.total_stock_value]

# Task 2.4: identify & remove the outliers using interquantile range (IQR) in descriptive statistics
# IQR = df.quantile(.75)-df.quantile(.25)
# Upper outliers definition: df.quantile(.75) + (1.5 * IQR)
# lower outliers definition: df.quantile(.25) - (1.5 * IQR)

# determine the number of lower outliers for each row/person => we will ignore this based on the results   
lower_outliers = df.quantile(.25) - 1.5 * (df.quantile(.75)-df.quantile(.25))
pd.DataFrame((df[1:] < lower_outliers[1:]).sum(axis = 1), columns = ['# of lower outliers']).\
    sort_values('# of lower outliers',  ascending = [0]).head(7)

# determine the number of upper outliers for each row/person 
upper_outliers = df.quantile(.5) + 1.5 * (df.quantile(.75)-df.quantile(.25))
pd.DataFrame((df[1:] > upper_outliers[1:]).sum(axis = 1), columns = ['# of upper outliers']).\
    sort_values('# of upper outliers',  ascending = [0]).head(7)

# "TOTAL" doesn't add much value to the set so we will remove it.
# Kenneth Lay and Jeffrey Skilling are very important personas in Enron case 
# We will leave the rest of the outliers since they maybe anomalies vs outliers
df = df.drop(['TOTAL'],0)

# In[10]:

### Task 3: Create new feature(s) & store in the dataframe

#feature scaling:fraction of person's email to POI to all sent messages
df['to_poi_message_ratio'] = df['from_this_person_to_poi']/df['from_messages']
#clean all 'inf' values which we got if the person's from_messages = 0
df = df.replace('inf', 0)

#feature scaling: fraction of person's email from POI to all messages received
df['from_poi_message_ratio'] = df['from_poi_to_this_person']/df['to_messages']
#clean all 'inf' values which we got if the person's to_messages = 0
df = df.replace('inf', 0)

initial_features_list.extend(['to_poi_message_ratio', 'from_poi_message_ratio'])
features_list.extend(['to_poi_message_ratio', 'from_poi_message_ratio'])

# In[11]:

### Task 4.1: Trying GaussianNB
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### normalize the training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_norm = df[initial_features_list]
df_norm = scaler.fit_transform(df_norm.iloc[:,1:])

# Trying GaussianNB => eliminated for final model based on results
clf = GaussianNB()
temp_features_list = ['poi']+range(7)

my_dataset_GNB = pd.DataFrame(SelectKBest(f_classif, k = 7).fit_transform(df_norm, df.poi), index = df.index)
my_dataset_GNB.insert(0, "poi", df.poi)
my_dataset_GNB = my_dataset_GNB.to_dict(orient = 'index')

dump_classifier_and_data(clf, my_dataset_GNB, temp_features_list)
tester.main()


# In[12]:

# Trying PCA + GaussianNB => eliminated for final model based on results
pca = PCA(n_components=3)
temp_features_list = ['poi']+range(3)
my_dataset_GNB = pd.DataFrame(SelectKBest(f_classif, k=8).fit_transform(df_norm, df.poi), index = df.index)
PCA_dataset = pd.DataFrame(pca.fit_transform(my_dataset_GNB),  index=df.index)
PCA_dataset.insert(0, "poi", df.poi)
PCA_dataset = PCA_dataset.to_dict(orient = 'index')  

dump_classifier_and_data(clf, PCA_dataset, temp_features_list)
tester.main()

# In[13]:

# Trying Decision tree => eliminated for final model based on the results
clf = DecisionTreeClassifier(random_state = 75)
my_dataset_DT = df[initial_features_list].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, my_dataset_DT, initial_features_list)
tester.main() 

# In[15]:

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. 

### Trying random forest along with grid_search for tuning the parameters => reasonable results but slow
clf = RandomForestClassifier()
my_dataset_RF = df[initial_features_list].to_dict(orient = 'index')

# Searchgrid for random forest: specify parameters and distributions to sample from
param_grid = {'bootstrap': [False],
 'criterion': ['entropy'],
 'max_depth': [None],
 'max_features': [1],
 'min_samples_leaf': [1],
 'min_samples_split': [9]}

grid_search = GridSearchCV(clf,param_grid=param_grid)
dump_classifier_and_data(clf, my_dataset_RF, initial_features_list)
tester.main()

# In[412]:

# Trying Decision tree with feature importance => reasonable results for this assignment
clf = DecisionTreeClassifier(random_state = 75)
my_dataset_DT = df[initial_features_list].to_dict(orient = 'index')
clf.fit(df_norm, df['poi'])

# create and sort features_list of non-null importance features for the model
features_importance = []
len (clf.feature_importances_)
for i in range(len(clf.feature_importances_)):
   if clf.feature_importances_[i] > 0:
       features_importance.append([df.columns[i+1], clf.feature_importances_[i]])
features_importance.sort(key=lambda x: x[1], reverse = True)
for f_i in features_importance:
    print f_i
features_list = [x[0] for x in features_importance]
features_list.insert(0, 'poi')

# Searchgrid for tuning parameters
param_grid = {'bootstrap': [False],
 'criterion': ['entropy'],
 'max_depth': [None],
 'max_features': [1],
 'min_samples_leaf': [1],
 'min_samples_split': [9]}

grid_search = GridSearchCV(clf,param_grid=param_grid)

my_dataset_DT = df[features_list].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, my_dataset_DT, features_list)
tester.main() 

# In[410]:

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
### dump_classifier_and_data(clf, my_dataset_DT, features_list)


# In[ ]:



