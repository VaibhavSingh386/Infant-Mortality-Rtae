#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as seabornInstance

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("IMR4_state_IMR.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


data.isnull().sum()/data.shape[0]*100


# In[9]:


data.duplicated().sum()


# In[10]:


for i in data.select_dtypes(include="object").columns:
    print(data[i].value_counts())
    print("***"*10)


# In[11]:


data.describe().T


# In[12]:


data.describe(include="object")


# In[13]:


import warnings
warnings.filterwarnings("ignore")
sns.distplot(data['Infant mortality rate - 1971'])


# In[14]:


#Deviate from the normal distribution.
#Have appreciable positive skewness.
#Show peakedness.
print('Skewness: %f' % data['Infant mortality rate - 1971'].skew())
print('Kurtsis: %f' %data['Infant mortality rate - 1971'].kurt())


# In[15]:


import warnings
warnings.filterwarnings("ignore")
for i in data.select_dtypes(include="number").columns:
    sns.histplot(data=data,x=i)
    plt.show()


# In[16]:


import warnings
warnings.filterwarnings("ignore")
for i in data.select_dtypes(include="number").columns:
    sns.boxplot(data=data,x=i)
    plt.show()


# In[17]:


import warnings
warnings.filterwarnings("ignore")
for i in data.select_dtypes(include="number").columns:
    plt.figure(figsize=(14,5))
    sns.kdeplot(data=data,x=i)
    plt.show()


# In[18]:


for i in ['Infant mortality rate - 1971', 'Infant mortality rate - 1972', 'Infant mortality rate - 1974',
       'Infant mortality rate - 1975', 'Infant mortality rate - 1976',
       'Infant mortality rate - 1977', 'Infant mortality rate - 1978',
       'Infant mortality rate - 1979', 'Infant mortality rate - 1980',
       'Infant mortality rate - 1981', 'Infant mortality rate - 1982',
       'Infant mortality rate - 1983', 'Infant mortality rate - 1984',
       'Infant mortality rate - 1985', 'Infant mortality rate - 1986',
       'Infant mortality rate - 1987', 'Infant mortality rate - 1988',
       'Infant mortality rate - 1989', 'Infant mortality rate - 1990',
       'Infant mortality rate - 1991', 'Infant mortality rate - 1992',
       'Infant mortality rate - 1993', 'Infant mortality rate - 1994',
       'Infant mortality rate - 1995', 'Infant mortality rate - 1996',
       'Infant mortality rate - 1997', 'Infant mortality rate - 1998',
       'Infant mortality rate - 1999', 'Infant mortality rate - 2000',
       'Infant mortality rate - 2001', 'Infant mortality rate - 2002',
       'Infant mortality rate - 2003', 'Infant mortality rate - 2004',
       'Infant mortality rate - 2005', 'Infant mortality rate - 2006',
       'Infant mortality rate - 2007', 'Infant mortality rate - 2008',
       'Infant mortality rate - 2009', 'Infant mortality rate - 2010',
       'Infant mortality rate - 2011', 'Infant mortality rate - 2012']:
    sns.scatterplot(data=data,x=i,y='Infant mortality rate - 1973')
    plt.show()


# In[19]:


data.select_dtypes(include="number").columns


# In[20]:


s=data.select_dtypes(include="number").corr()


# In[21]:


plt.figure(figsize=(30,30))
sns.heatmap(s,annot=True)


# In[22]:


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[23]:


for i in ["Infant mortality rate - 1975","Infant mortality rate - 1986","Infant mortality rate - 1997","Infant mortality rate - 2003"]:
    data[i].fillna(data[i].median(),inplace=True)


# In[24]:


from sklearn.impute import KNNImputer
impute=KNNImputer()


# In[25]:


for i in data.select_dtypes(include="number").columns:
    data[i]=impute.fit_transform(data[[i]])


# In[26]:


data.isnull().sum()


# In[27]:


def wisker(col):
    q1,q3=np.percentile(col,[25,75])
    iqr=q3-q1
    lw=q1-1.5*iqr
    uw=q3+1.5*iqr
    return lw,uw


# In[28]:


wisker(data['Infant mortality rate - 1992'])


# In[29]:


for i in ['Infant mortality rate - 1972','Infant mortality rate - 1976','Infant mortality rate - 1977','Infant mortality rate - 1978','Infant mortality rate - 1980','Infant mortality rate - 1983','Infant mortality rate - 1984','Infant mortality rate - 1988','Infant mortality rate - 1989','Infant mortality rate - 1990']:
    lw,uw=wisker(data[i])
    data[i]=np.where(data[i]<lw,lw,data[i])
    data[i]=np.where(data[i]>uw,uw,data[i])


# In[30]:


for i in ['Infant mortality rate - 1972','Infant mortality rate - 1976','Infant mortality rate - 1977','Infant mortality rate - 1978','Infant mortality rate - 1980','Infant mortality rate - 1983','Infant mortality rate - 1984','Infant mortality rate - 1988','Infant mortality rate - 1989','Infant mortality rate - 1990']:
    sns.boxplot(data[i])
    plt.show()


# In[31]:


class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        
        # for leaf node
        self.value = value


# In[32]:


class DecisionTreeRegressor():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["var_red"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["var_red"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_var_red = -float("inf")
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    # update the best split if needed
                    if curr_var_red>max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def variance_reduction(self, parent, l_child, r_child):
        ''' function to compute variance reduction '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction
    
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        val = np.mean(Y)
        return val
                
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
        
    def make_prediction(self, x, tree):
        ''' function to predict new dataset '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
    def predict(self, X):
        ''' function to predict a single data point '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions


# In[33]:


X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)


# In[34]:


regressor = DecisionTreeRegressor(min_samples_split=3, max_depth=3)
regressor.fit(X_train,Y_train)
regressor.print_tree()


# In[35]:


Y_pred = regressor.predict(X_test) 
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(Y_test, Y_pred))


# In[36]:


data.columns


# In[37]:


data.drop_duplicates()


# In[38]:


dummy=pd.get_dummies(data=data,columns=["Category","Country/ State/ UT Name"],drop_first=True)


# In[39]:


dummy


# In[40]:


dummy.dtypes


# In[41]:


plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.kdeplot(x=dummy['Infant mortality rate - 1973'])

plt.subplot(1,2,2)
sns.kdeplot(x=np.log(dummy['Infant mortality rate - 1973']))


# In[42]:


dummy['Infant mortality rate - 1973']=np.log(dummy['Infant mortality rate - 1973'])


# In[43]:


dummy.columns


# In[44]:


import statsmodels.api as sm
from sklearn.model_selection import train_test_split


# In[45]:


x=dummy.drop('Infant mortality rate - 1973',axis=1)
y=dummy['Infant mortality rate - 1973']


# In[46]:


x_c=sm.add_constant(x)


# In[47]:


x_train, x_test, y_train, y_test = train_test_split(x_c, y, test_size=30, random_state=42)


# In[48]:


model=sm.OLS(y_train,x_train.astype(float)).fit()


# In[49]:


import warnings
warnings.filterwarnings("ignore")
model.summary()


# In[50]:


y_predict=model.predict(x_test.astype(float))


# In[51]:


y_predict


# In[52]:


residual=y_test-y_predict
residual


# In[53]:


np.sqrt((residual**2).mean())


# In[54]:


y_test.mean()


# In[55]:


0.19615621233855746/4.753818534966868


# In[56]:


residual


# In[57]:


residual.mean()


# In[58]:


print(residual.skew())
sns.kdeplot(residual)


# In[59]:


sns.scatterplot(x=y_predict,y=residual)


# In[60]:


a=pd.Series(residual,name="residual")
b=pd.Series(y_predict,name="y_predict")


# In[61]:


ab=pd.DataFrame(a)


# In[62]:


rp=pd.merge(a,b,left_index=True,right_index=True)


# In[63]:


sns.lmplot(data=rp,x="y_predict",y="residual")


# In[64]:


Y_pred = regressor.predict(X_test) 
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(Y_test, Y_pred))


# In[65]:


data.shape


# In[66]:


data.info()


# In[67]:


data.nunique()


# In[68]:


data.columns


# In[69]:


new_data = data[['Infant mortality rate - 2003', 'Infant mortality rate - 2012']]
new_data


# In[70]:


new_data.plot(x='Infant mortality rate - 2003', y='Infant mortality rate - 2012', kind='scatter')	


# In[71]:


data.plot(x='Infant mortality rate - 1998', y='Infant mortality rate - 2012', kind='scatter')


# In[72]:


data.plot(x='Infant mortality rate - 2002', y='Infant mortality rate - 2012', kind='scatter')


# In[73]:


data.plot(x='Infant mortality rate - 2001', y='Infant mortality rate - 2012', kind='scatter')


# In[74]:


data.plot(x='Infant mortality rate - 2000', y='Infant mortality rate - 2012', kind='scatter')


# In[75]:


x = new_data['Infant mortality rate - 2012']
y = new_data['Infant mortality rate - 2012']


# In[76]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[77]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=30, random_state=42)


# In[78]:


print("X_train:",x_train.shape)
print("X_test:",x_test.shape)
print("Y_train:",y_train.shape)
print("Y_test:",y_test.shape)


# In[79]:


model = LinearRegression()


# In[80]:


model.fit(x_train.values.reshape(-1,1), y_train)


# In[81]:


model.coef_


# In[82]:


model.intercept_


# In[83]:


y_pred = model.predict(x_test.values.reshape(-1,1))


# In[84]:


mse = mean_squared_error(y_test, y_pred)
print("MSE --> ", mse)


# In[85]:


import math
rmse = math.sqrt(mse)
print("RMSE --> ", rmse)


# In[86]:


mae = mean_absolute_error(y_test, y_pred)
print("MAE --> ", mae)


# In[87]:


r2 = r2_score(y_test, y_pred)
print("R2 --> ", r2)


# In[88]:


print("MSE --> ", mse)
print("RMSE --> ", rmse)
print("MAE --> ", mae)
print("R2 --> ", r2)


# In[89]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[90]:


plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')


# In[91]:


sns.regplot(x=x, y=y, ci=None, color ='blue')


# In[92]:


data.plot(x='Infant mortality rate - 1971', y='Infant mortality rate - 2000', kind='scatter')


# In[93]:


x = data[['Infant mortality rate - 2003', 'Infant mortality rate - 1971']]
x


# In[94]:


y


# In[95]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[96]:


print("X_train:",x_train.shape)
print("X_test:",x_test.shape)
print("Y_train:",y_train.shape)
print("Y_test:",y_test.shape)


# In[97]:


model = LinearRegression()


# In[98]:


model.fit(x_train, y_train)


# In[99]:


model.coef_


# In[100]:


model.intercept_


# In[101]:


y_pred = model.predict(x_test)


# In[102]:


mse_2 = mean_squared_error(y_test, y_pred)
rmse_2 = math.sqrt(mse_2)
mae_2 = mean_absolute_error(y_test, y_pred)
r2_2 = r2_score(y_test, y_pred)


# In[103]:


print("MSE --> ", mse_2)
print("RMSE --> ", rmse_2)
print("MAE --> ", mae_2)
print("R2 --> ", r2_2)


# In[104]:


plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')


# In[105]:


sns.regplot(x=y_test, y=y_pred, ci=None, color ='blue')


# In[106]:


metrics = {
    'Model': ['First', 'Second'],
    'MSE' : [mse, mse_2],
    'RMSE' : [rmse, rmse_2],
    'MAE' : [mae, mae_2],
    'R2' : [r2, r2_2]
    }

metrics_data = pd.DataFrame(data=metrics)


# In[107]:


metrics_data


# In[108]:


sns.lmplot(data=rp,x="y_predict",y="residual")


# In[ ]:





# In[ ]:





# In[ ]:




