# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 20:11:23 2020

@author: arpit
"""

import numpy as np
import pandas as pd
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

#combine data
train['source'] = "train"
test['source'] = "test"

data = pd.concat([train,test])

train.shape
test.shape
data.shape

#check for null values
data.apply(lambda x: sum(x.isnull()))
data.describe()

#check for unique values under each column
data.apply(lambda x:len(x.unique()))

data['Item_Fat_Content'].unique()
data['Item_Type'].unique()
data['Outlet_Establishment_Year'].unique()
data['Outlet_Identifier'].unique()
data['Outlet_Location_Type'].unique()
data['Outlet_Size'].unique()

#categorical columns
data.dtypes
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier', 
                                                                   'Outlet_Identifier',
                                                                   'source']]
for col in categorical_columns:
    print('\n'+'The value counts for %s ' %col)
    print(data[col].value_counts())
    
#imputing missing values
pd.pivot_table(data, values = 'Item_Weight', index = 'Item_Identifier' )

#missing boolean for Item_Weight
miss_bool = data['Item_Weight'].isnull()
item_avg_weight = data['Item_Weight'].mean()
data.loc[miss_bool, 'Item_Weight'] = data.loc[miss_bool, 'Item_Identifier'].apply(lambda x:item_avg_weight)
    
#imputing values for outlet_Size
''' Since it is categorical data we'll use the mode'''

'''from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy = 'most_frequent')'''

from scipy.stats import mode
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',
                                    aggfunc=(lambda x:mode(x).mode[0]) )
print('Mode for each Outlet_Type:')
print(outlet_size_mode)

#imputing values in Outlet_Size
miss_bool = data['Outlet_Size'].isnull()
data.loc[miss_bool, 'Outlet_Size'] = data.loc[miss_bool, 'Outlet_Type'].apply(
        lambda x: outlet_size_mode[x])


''' we have imputed the value is Outlet_size using the most frequent value with 
 respect to their outlet_type'''
 
 #imputing the values in Item_Visibility where the values are 0 with the average of
 # the values of same item_identifier
 
miss_bool = (data['Item_Visibility']==0)
visibility_avg = data.pivot_table(values='Item_Visibility', index = 'Item_Identifier')
data.loc[miss_bool, 'Item_Visibility'] = data.loc[miss_bool, 'Item_Identifier'].apply(lambda x: visibility_avg.loc[x])


''' In order to predict the sales we'll create some columns for more accuracy'''

data['Item_Visibility_MeanRatio'] = data.apply(lambda x:
    x.loc['Item_Visibility']/visibility_avg.loc[x['Item_Identifier']], axis=1)
    
''' the values in Item_Fat_content are repeating. for ex: LF, low fat is equal to Low Fat. So we need to fix it '''

''' data.loc[data['Item_Fat_Content']=='LF']['Item_Fat_Content']= 'Low Fat'
data.loc[data['Item_Fat_Content']=='low fat'] = 'Low Fat'
data.loc[data['Item_Fat_Content']=='reg'] = 'Regular'

Not working '''

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
        'LF' : 'Low Fat',
        'low fat' : 'Low_Fat',
        'reg' : 'Regular'})

#we'll add the column outlet_years to determine the age od the store
#older the store greater the sales

data['Outlet_Years'] = (data.loc[:,'Outlet_Establishment_Year'].astype(str).astype(int)).apply(lambda x : 2013 - x)

# creating a new column for the item type food, drink and non consumable using Item_Identifier

data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_Type_Combined'] = data['Item_Type_Combined'].replace({
        'FD' : 'Food',
        'DR' : 'Drink',
        'NC' : 'Non Consumable'})
    
    
# set new category as non edible in Item_Fat_Content for Non consumable category in Item_type_combined
    

data.loc[data['Item_Type_Combined']=='Non Consumable',  'Item_Fat_Content'] = 'Non Edible'


#converting categorical variable into variable for further processing us sklearn labelEncoder

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])

cat_var=['Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type', 
         'Item_Type_Combined', 'Outlet' ]
for i in cat_var:
    data[i] = le.fit_transform(data[i])
    
    
    
#One Hot Encoding
data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type',
                                     'Item_Type_Combined', 'Outlet'])

data.dtypes
    
#remove the excess columns
data.drop(['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)

#splitting the data into train and test
train = data[data['source'] == 'train']
test = data[data['source'] == 'test']

train.drop(['source'], axis=1, inplace=True)
test.drop(['source', 'Item_Outlet_Sales'], axis=1, inplace=True)
    










    