#!/usr/bin/env python
# coding: utf-8

# <h1> TASK 3 (NLP)
# 

# In[1]:


#Importing packages
import pandas as pd 
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <h2>Reading Data And Basic Analysis

# In[2]:


df= pd.read_csv(r"F:\vaishnav srivastava\Tableau projects\Flipkart\flipkart_com-ecommerce_sample - flipkart_com-ecommerce_sample - flipkart_com-ecommerce_sample.csv")


# In[3]:


df.head()


# In[4]:


df['product_category_tree']=df['product_category_tree'].str.replace('[', '').str.replace('"', '')
print(df.info())
print(df.describe())
df.head()


# <h3> Seprating Main Categories

# In[5]:


#Splitting category column 
df['main_category']=df['product_category_tree'].str.split('>>', expand=True)[0]
df['main_category'].value_counts()


# In[6]:


'''
    We have 266 different main categories but many of these are just names of the product
    So filtering out them on the basis of their count
    
    '''

value_count=df['main_category'].value_counts()
to_remove=value_count[value_count<=10].index
df=df[df.main_category.isin(to_remove)==False]
df.main_category.value_counts()


# <h3> Creating function to seprate sub categories

# In[7]:


def second_category(value):
    try:
        return value.split('>>')[1]
    except IndexError:
        return 'None'
def third_category(value):
    try:
        return value.split('>>')[2]
    except IndexError:
        return 'None'


# In[8]:


#applying function
df['sub_category']=df['product_category_tree'].apply(second_category)
df['sub-sub_category']=df['product_category_tree'].apply(third_category)
df.head()


# <h3> Deleting Extra Columns

# In[9]:


df.product_rating.value_counts()


# In[10]:


'''Rating column has many values with no rating availabale'''


df.drop(['uniq_id', 'product_url', 'pid', 'image', 'product_rating', 'overall_rating', 'product_specifications'], axis=1)


# <h1> EDA

# Discounted sales by category

# In[11]:


df['discounted_percentage']=round((df['retail_price']-df['discounted_price'])/df['retail_price']*100,2)
df.head()


# Creating new data frame

# In[12]:


main_category_discount_percentage=df.groupby('main_category').agg({'discounted_percentage':[np.mean],'main_category':['count']})
main_category_discount_percentage


# In[13]:


#combining column
main_category_discount_percentage.columns=['_' .join(column) for column in main_category_discount_percentage.columns]
main_category_discount_percentage


# In[14]:


main_category_discount_percentage.sort_values(by='main_category_count',ascending=False)


# In[15]:


plt.figure(figsize=(12,8))
main_category_discount_percentage['discounted_percentage_mean'].sort_values(ascending=True).plot(kind='barh')
plt.title('Main Category by Discount', fontsize=15)
plt.ylabel('Main Category', fontsize=13)
plt.show()


# In[16]:


## The Brand column has lots of null values.

plt.figure(figsize =(10,8))
sns.heatmap(df.isnull(),yticklabels=False,cmap='plasma',cbar=True)
plt.show()


# <h3> Seprating Year and Month

# In[17]:


#converting to datetime object
df.crawl_timestamp=pd.to_datetime(df.crawl_timestamp)


# In[18]:


df['Year']=df.crawl_timestamp.apply(lambda x: x.year)
df['Month']=df.crawl_timestamp.apply(lambda x: x.month)


# In[19]:


df.head()


# In[20]:


#December has most number of sales
plt.figure(figsize=(8,6))
df.groupby('Month')['Month'].count().plot(kind='bar')
plt.title('Sales Count by Month',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel('Month',fontsize=12)
plt.ylabel('Sales Count',fontsize=12)
plt.show()
print(df.groupby('Month')['Month'].count())


# In[21]:


#2015 has slightly more number of sales
plt.figure(figsize=(8,6))
df.groupby('Year')['Year'].count().plot(kind='bar')
plt.title('Sales Count by Year',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel('Year',fontsize=12)
plt.ylabel('Sales Count',fontsize=12)
plt.show()
print(df.groupby('Year')['Year'].count())


# <h3> Sales with different categories

# In[22]:


#Clothing category has most number of sales
plt.figure(figsize=(12,8))
df['main_category'].value_counts().sort_values(ascending=True).plot(kind='barh')
plt.title('Main Category',fontsize=15)
plt.yticks(fontsize=10)
plt.xticks(fontsize=12)
plt.show()


# In[23]:


plt.figure(figsize=(12,8))
df['sub_category'].value_counts()[:20].sort_values(ascending=True).plot(kind='barh')
plt.title('Sub-Category',fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()
print('Top Ten Sub-Categories by Sales.\n')
print(df['sub_category'].value_counts()[:10])


# In[24]:


plt.figure(figsize=(12,8))
df['sub-sub_category'].value_counts()[:20].sort_values(ascending=True).plot(kind='barh')
plt.title('Sub-Sub Category',fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()
print('Top Sub-Sub Categories by Sales.\n')
print(df['sub-sub_category'].value_counts()[:10])


# In[25]:


#Most Expensive item
df[df['discounted_price']==df['discounted_price'].max()]


# In[26]:


#Most cheap item
df[df['discounted_price']==df['discounted_price'].min()]


# <h3> Cleaning Data 

# In[27]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text
    


# In[28]:


#applying function to description column by converting to string
df['description'] = df['description'].astype(str)


# In[29]:


df['description'] = df['description'].apply(clean_text)


# In[30]:


df.info()


# <h1>Model Training

# In[31]:


# Creating X, y Variables
X, y = df['description'], df['main_category']
# Setting up train test split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# # Pipeline and Grid Search Setup

# In[32]:


#CountVectorizer Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('clf', MultinomialNB())])

#TfidVectorizer Pipeline
tvc_pipe = Pipeline([
 ('tvec', TfidfVectorizer()),
 ('clf1', MultinomialNB())
])
## Hyperparameter Tuning
tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'clf__alpha': [1, 1e-1, 1e-2]}

tf_params = {
 'tvec__ngram_range': [(1, 1), (1, 2), (2, 2)],
 'tvec__stop_words': [None, 'english'],
 'clf1__alpha': [1, 1e-1, 1e-2]
 
}


# In[33]:


# Setting up GridSearch for CountVectorizer
clf = GridSearchCV(text_clf, tuned_parameters, cv=5, verbose =1, n_jobs = -1)
clf.fit(X_train, y_train)
# Setting up GridSearch for TfidVectorizer
clf2 = GridSearchCV(tvc_pipe, tf_params, cv=5, verbose =1, n_jobs = -1)
clf2.fit(X_train, y_train)


# In[34]:


print(clf.score(X_test, y_test))# Scoring Training data on CountVectorizer
print(clf.score(X_train, y_train))# Scoring Testing data on CountVectorizer
print(clf2.score(X_test, y_test))# Scoring Training data on TFIDFVectorizer
print(clf2.score(X_train, y_train))# Scoring Testing data on TFIDFVectorizer


# In[35]:


#Classification report of CountVEctorizer
predicted_clf=clf.predict(X_test)
print(classification_report(y_test,predicted_clf))


# In[36]:


#Classification report of TfidVectorizer
predicted_clf2=clf2.predict(X_test)
print(classification_report(y_test,predicted_clf2))


# In[ ]:




