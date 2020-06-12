#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:27:13 2020

@author: kaushik
"""

import numpy as np
import pandas as pd
import seaborn as sns
import random
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv('passdata.csv',',',error_bad_lines=False)

data.head()

data[data['password'].isnull()]
data.dropna(inplace=True)

password_tuple = np.array(data)

random.shuffle(password_tuple)

y = [labels[1] for labels in password_tuple]

X = [labels[0] for labels in password_tuple]

def word_divide_char(inputs):
    characters = []
    for i in inputs:
        characters.append(i)
    return characters

vectorizer = TfidfVectorizer(tokenizer=word_divide_char)
X = vectorizer.fit_transform(X)
X.shape
vectorizer.vocabulary_

sns.set_style('whitegrid')
sns.countplot(x='strength',data=data,palette='pastel')

data.iloc[0,0]

feature_names = vectorizer.get_feature_names()
first_document_vector = X[0]

df = pd.DataFrame(first_document_vector.T.todense(),index=feature_names
                  ,columns=['tfidf'])
df.sort_values(by=['tfidf'],ascending=False)

print('-----------------------------------------------')

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

log_class = LogisticRegression(penalty='l2',multi_class='ovr')
log_class.fit(X_train,y_train)
print(log_class.score(X_test,y_test))

cl = LogisticRegression(random_state=0,multi_class='multinomial',solver='newton-cg')
cl.fit(X_train,y_train)
print(cl.score(X_test,y_test))


X_predict=np.array(["%@123abcd"])
X_predict=vectorizer.transform(X_predict)
y_pred=log_class.predict(X_predict)
print(y_pred)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train,y_train)
nb_classifier.score(X_test,y_test)


X_predict=np.array(["%@123abcd"])
X_predict=vectorizer.transform(X_predict)
y_pred=nb_classifier.predict(X_predict)
print(y_pred)