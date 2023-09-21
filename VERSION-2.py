#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

training_data = pd.read_excel('training (2) (1).xlsx')
testing_data = pd.read_excel('testing (2) (1).xlsx')

training_data['input'].fillna('', inplace=True)
testing_data['Equation'].fillna('', inplace=True)

Tr_X = training_data['input']
Tr_y = training_data['Classification']

Te_X = testing_data['Equation']
Te_y = testing_data['Classification']

tfidf_vectorizer = TfidfVectorizer()
Tr_X = tfidf_vectorizer.fit_transform(Tr_X)
Te_X = tfidf_vectorizer.transform(Te_X)

feature_names = tfidf_vectorizer.get_feature_names_out().tolist()

model = DecisionTreeClassifier()
model.fit(Tr_X, Tr_y)

train_accuracy = model.score(Tr_X, Tr_y)
test_accuracy = model.score(Te_X, Te_y)

print("Training Set Accuracy:", train_accuracy)
print("Test Set Accuracy:", test_accuracy)

plt.figure(figsize=(70, 20))
plot_tree(model, filled=True, feature_names=feature_names, class_names=["0", "1"])
plt.show()


# In[2]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

training_data = pd.read_excel('training (2) (1).xlsx')
testing_data = pd.read_excel('testing (2) (1).xlsx')

training_data['input'].fillna('', inplace=True)
testing_data['Equation'].fillna('', inplace=True)

Tr_X = training_data['input']
Tr_y = training_data['Classification']

Te_X = testing_data['Equation']
Te_y = testing_data['Classification']

tfidf_vectorizer = TfidfVectorizer()
Tr_X = tfidf_vectorizer.fit_transform(Tr_X)
Te_X = tfidf_vectorizer.transform(Te_X)

model = DecisionTreeClassifier(max_depth=5)
model.fit(Tr_X, Tr_y)

train_accuracy = model.score(Tr_X, Tr_y)
test_accuracy = model.score(Te_X, Te_y)

print("Training Set Accuracy (max_depth=5):", train_accuracy)
print("Test Set Accuracy (max_depth=5):", test_accuracy)

plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=list(tfidf_vectorizer.get_feature_names_out()), class_names=["0", "1"], fontsize=8)
plt.show()


# In[ ]:




