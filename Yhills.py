#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import nltk
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from wordcloud import WordCloud
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score, recall_score, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


# In[2]:


df = pd.read_csv('fake_job_postings.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df.fraudulent.nunique()


# * 4 columns named as job_id, telecommuting, has_company_logo and has_questions features have numerical data and can be removed since they are of no use in text classification problems.
# * The 'fraudulent' is the column on which our model will be trained and predicted.

# In[9]:


columns=['job_id', 'telecommuting', 'has_company_logo', 'has_questions', 'salary_range', 'employment_type']


# In[10]:


df_f = df[df['fraudulent'] == 1]  #fraudulent
df_nf = df[df['fraudulent'] == 0]  # non fraudulent
plt.figure(figsize=(10,5))

plt.pie([df_f.shape[0], df_nf.shape[0]], 
        labels=['fraudulent', 'non fraudulent'], 
        explode=(0, 0.2),
        autopct='%1.2f%%',
       colors=['red','blue'])
plt.title(' fraudulent vs non fraudulent')
plt.show()


# In[11]:


#It is the highly imbalance Dataset


# In[12]:


#Replacing the Null Values with " "(space)
df.fillna(' ', inplace=True)


# In[13]:


#checking which country posts most number of jobs
def split(location):
    l = location.split(',')
    return l[0]

df['country'] = df['location'].apply(split)


# **Imputing nulls values**

# In[60]:


for idx in (df[df['department'].isna()]['title'].index) :
    if 'Marketing' in df.at[idx ,'title' ] :
        df.at[idx , 'department'] = 'Marketing'
    elif 'Sales' in df.at[idx ,'title' ] :
        df.at[idx , 'department'] = 'Sales'
    elif ('Accountant' in df.at[idx ,'title' ])|('Accounting' in df.at[idx ,'title' ] ) :
        df.at[idx , 'department'] = 'Accounting'
    elif ('Engineer' in df.at[idx ,'title' ] )|('Engineering' in df.at[idx ,'title' ] ) :
        df.at[idx , 'department'] = 'Engineering'
    else :
        df.at[idx , 'department'] = df.at[idx , 'title']


# In[61]:


for idx in (df['salary_range'].dropna()).index :
    Range = df.at[idx , 'salary_range'].split('-')
    try :
        start = int(Range[0])
        if start < 1000 :
            df.at[idx ,'salary_range' ] = 0
        else :
            df.at[idx ,'salary_range' ] = start
            
    except ValueError :
        df.at[idx ,'salary_range' ] = 0
        
df['salary_range'] = df['salary_range'].fillna(0)


# In[62]:


df.head()


# **Exploratory Data Analysis**

# In[14]:


country = dict(df.country.value_counts()[:11])
del country[' ']
plt.figure(figsize=(8,6))
plt.title('No. of job postings country wise', size=20)
plt.bar(country.keys(), country.values())
plt.ylabel('No. of jobs', size=10)
plt.xlabel('Countries', size=10)


# In[15]:


df[['title','fraudulent']].groupby(['fraudulent'])['title'].value_counts()


# In[35]:


#Industry Having most number of fake jobs
df_fraud = df[df['fraudulent']==1]
df_fraud[['fraudulent','industry']].groupby(['industry']).sum().reset_index().sort_values(by='fraudulent',ascending = False).head(10)


# In[43]:


most_fraud_industry =df_fraud[['fraudulent','industry']].groupby(['industry']).sum().reset_index().sort_values(by='fraudulent',ascending = False).head(10)[1:]
x = most_fraud_industry['industry']
y= most_fraud_industry['fraudulent']
plt.figure(figsize=(10, 6))
plt.barh(x,y)
plt.xlabel('Fraud count')
plt.ylabel('Industry')
plt.title('Top Most Fraud Industry')
plt.show()


# In[44]:


# Top 10 most common job titles
job_title_counts = df['title'].value_counts().head(10)  
job_title_counts.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
plt.ylabel('')
plt.title('Top 10 Most Common Job Titles')
plt.show()


# In[45]:


#Most Number of Fake Jobs Department
df[df['fraudulent']==1]['department'].value_counts()[1:]


# In[46]:


most_fakes_job = df[df['fraudulent']==1]['department'].value_counts()[1:11]
most_fakes_job.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Job Title')
plt.ylabel('Frequency')
plt.title('Top 10 Most Fake Job Profile')
plt.xticks(rotation=45)
plt.show()


# It was observed that the highest number of fake job profiles originate from the **Engineering** department.

# In[47]:


def process_text(text):
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    return words

df['profile'] = df['company_profile'].apply(process_text)

# Count word frequencies
word_freq = Counter()
for text in df['profile']:
    word_freq.update(text)

# Find the top 3 most common words
top_words = word_freq.most_common(3)

print("Top 3 most common words in Company Profile:")
for word, freq in top_words:
    print(f"{word}: {freq}")


# In[48]:


columns_to_drop = ['profile']
df.drop(columns=columns_to_drop, inplace=True)


# In[28]:


df.head(2)


# In[66]:


UK_job = df[(df['location'].str.contains(r'\bUK\b', case=False, na=False) | df['location'].str.contains('United Kingdom', case=False, na=False))]


# In[87]:


highest_paid_department = UK_job.loc[UK_job['salary_range'].idxmax()]['department']
print(f"The department or function with the highest-paying jobs in the UK is: {highest_paid_department}")


# In[80]:


df.loc[df['salary_range'].idxmax()][['department','function','location','salary_range']]


# **Machine Learning Modelling**

# In[9]:


#Combining the all columns into the single Text
df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['department'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits'] + ' ' + df['employment_type'] + ' ' + df['required_education'] + ' ' + df['industry'] + ' ' + df['function']


# In[10]:


df['text'] = df['text'].str.lower()

df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
data = df[['fraudulent','text']]


# In[40]:


from sklearn.utils import resample

# Separate the majority and minority classes
majority_class = data[data['fraudulent'] == 0]
minority_class = data[data['fraudulent'] == 1]

# Determine the number of samples to randomly select (e.g., same as the minority class size)
desired_sample_size = len(minority_class)

# Randomly sample the majority class to match the desired size
majority_sampled = resample(majority_class, replace=False, n_samples=desired_sample_size, random_state=42)

# Combine the minority class and the randomly sampled majority class
balanced_df = pd.concat([minority_class, majority_sampled])


# In[41]:


fraudulent_counts = balanced_df['fraudulent'].value_counts()

plt.figure(figsize=(6, 6)) 
plt.pie(fraudulent_counts, labels=['Non-Fraudulent', 'Fraudulent'], autopct='%1.1f%%', startangle=90)
plt.title('Class Distribution in balanced_df')

# Show the chart
plt.show()


# In[42]:


# Split the balanced dataset into features (X) and target (y)
X = balanced_df['text']
y = balanced_df['fraudulent']


# In[43]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[44]:


# Creating a TF-IDF vectorizer
vectorizer = TfidfVectorizer(strip_accents='unicode',
                             analyzer='word',
                             ngram_range=(1, 2),
                             max_features=15000,
                             smooth_idf=True,
                             sublinear_tf=True)


# In[46]:


# Creating pipelines for different classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Support Vector Classifier': SVC()
}

results = {}

# Iterate through classifiers and evaluate
for classifier_name, classifier in classifiers.items():
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculating confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"Classifier: {classifier_name}")
    print(f"Accuracy: {accuracy}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Store results
    results[classifier_name] = {
        'Accuracy': accuracy,
    }

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraudulent', 'Fraudulent'],
                yticklabels=['Non-Fraudulent', 'Fraudulent'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.show()

print("Results:")
print(results)


# **Result** 
# 
# **Support Vector Classifier (SVC)** is the best-fit model compare to the other classifiers. SVC achieved the highest accuracy score on our balanced dataset, demonstrating its capability to effectively classify fraudulent and non-fraudulent cases. Additionally, when reviewing the confusion matrices and classification reports, SVC consistently demonstrated balanced precision and recall rates for both classes.
