#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math

def entropy(probabilities):
    return -sum(p * math.log2(p) if p > 0 else 0 for p in probabilities)

def information_gain(data, attribute_index, target_index):
    total_instances = len(data)
    target_values = set(data[i][target_index] for i in range(total_instances))
    target_probabilities = [sum(1 for row in data if row[target_index] == value) / total_instances for value in target_values]
    entropy_before = entropy(target_probabilities)
    
    attribute_values = set(data[i][attribute_index] for i in range(total_instances))
    weighted_entropy_after = 0
    
    for value in attribute_values:
        subset = [row for row in data if row[attribute_index] == value]
        subset_size = len(subset)
        subset_target_probabilities = [sum(1 for row in subset if row[target_index] == target_value) / subset_size for target_value in target_values]
        weighted_entropy_after += (subset_size / total_instances) * entropy(subset_target_probabilities)
    
    information_gain = entropy_before - weighted_entropy_after
    return information_gain

data = [
    ["<=30", "high", "no", "fair", "no"],
    ["<=30", "high", "no", "excellent", "no"],
    ["31…40", "high", "no", "fair", "yes"],
    [">40", "medium", "no", "fair", "yes"],
    [">40", "low", "yes", "fair", "yes"],
    [">40", "low", "yes", "excellent", "no"],
    ["31…40", "low", "yes", "excellent", "yes"],
    ["<=30", "medium", "no", "fair", "no"],
    ["<=30", "low", "yes", "fair", "yes"],
    [">40", "medium", "yes", "fair", "yes"],
    ["<=30", "medium", "yes", "excellent", "yes"],
    ["31…40", "medium", "no", "excellent", "yes"],
    ["31…40", "high", "yes", "fair", "yes"],
    [">40", "medium", "no", "excellent", "no"]
]

target_index = 4
attributes = ["age", "income", "student", "credit_rating"]

information_gains = {}
for attribute_index, attribute in enumerate(attributes):
    gain = information_gain(data, attribute_index, target_index)
    information_gains[attribute] = gain

root_attribute = max(information_gains, key=information_gains.get)
root_information_gain = information_gains[root_attribute]

print(f"The root node is '{root_attribute}' with Information Gain of {root_information_gain:.3f}")


# In[5]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = [
    ["<=30", "high", "no", "fair", "no"],
    ["<=30", "high", "no", "excellent", "no"],
    ["31…40", "high", "no", "fair", "yes"],
    [">40", "medium", "no", "fair", "yes"],
    [">40", "low", "yes", "fair", "yes"],
    [">40", "low", "yes", "excellent", "no"],
    ["31…40", "low", "yes", "excellent", "yes"],
    ["<=30", "medium", "no", "fair", "no"],
    ["<=30", "low", "yes", "fair", "yes"],
    [">40", "medium", "yes", "fair", "yes"],
    ["<=30", "medium", "yes", "excellent", "yes"],
    ["31…40", "medium", "no", "excellent", "yes"],
    ["31…40", "high", "yes", "fair", "yes"],
    [">40", "medium", "no", "excellent", "no"]
]

columns = ["age", "income", "student", "credit_rating", "buys_computer"]

df = pd.DataFrame(data, columns=columns)

X = df.drop("buys_computer", axis=1)
y = df["buys_computer"]

categorical_features = ["age", "income", "student", "credit_rating"]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), categorical_features)],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier())
])

pipeline.fit(X, y)

tree_depth = pipeline.named_steps["classifier"].get_depth()

print(f"Tree depth: {tree_depth}")


# In[10]:


import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = [
    ["<=30", "high", "no", "fair", "no"],
    ["<=30", "high", "no", "excellent", "no"],
    ["31…40", "high", "no", "fair", "yes"],
    [">40", "medium", "no", "fair", "yes"],
    [">40", "low", "yes", "fair", "yes"],
    [">40", "low", "yes", "excellent", "no"],
    ["31…40", "low", "yes", "excellent", "yes"],
    ["<=30", "medium", "no", "fair", "no"],
    ["<=30", "low", "yes", "fair", "yes"],
    [">40", "medium", "yes", "fair", "yes"],
    ["<=30", "medium", "yes", "excellent", "yes"],
    ["31…40", "medium", "no", "excellent", "yes"],
    ["31…40", "high", "yes", "fair", "yes"],
    [">40", "medium", "no", "excellent", "no"]
]

columns = ["age", "income", "student", "credit_rating", "buys_computer"]

df = pd.DataFrame(data, columns=columns)

X = df.drop("buys_computer", axis=1)
y = df["buys_computer"]

categorical_features = ["age", "income", "student", "credit_rating"]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), categorical_features)],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier())
])

pipeline.fit(X, y)

feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out(input_features=categorical_features)) + list(X.columns.drop(categorical_features))

plt.figure(figsize=(70, 20))
plot_tree(pipeline.named_steps["classifier"], filled=True, feature_names=feature_names, class_names=['no', 'yes'])
plt.show()


# In[17]:


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


# In[18]:


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


# In[19]:


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

model = DecisionTreeClassifier(criterion="entropy")
model.fit(Tr_X, Tr_y)

train_accuracy = model.score(Tr_X, Tr_y)
test_accuracy = model.score(Te_X, Te_y)

print("Training Set Accuracy (Entropy Criterion):", train_accuracy)
print("Test Set Accuracy (Entropy Criterion):", test_accuracy)

plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=list(tfidf_vectorizer.get_feature_names_out()), class_names=["0", "1"], fontsize=8)
plt.show()


# In[20]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(Tr_X, Tr_y)

decision_tree_predictions = decision_tree_model.predict(Te_X)

decision_tree_accuracy = accuracy_score(Te_y, decision_tree_predictions)
decision_tree_classification_report = classification_report(Te_y, decision_tree_predictions)
decision_tree_confusion_matrix = confusion_matrix(Te_y, decision_tree_predictions)

random_forest_model = RandomForestClassifier()
random_forest_model.fit(Tr_X, Tr_y)

random_forest_predictions = random_forest_model.predict(Te_X)

random_forest_accuracy = accuracy_score(Te_y, random_forest_predictions)
random_forest_classification_report = classification_report(Te_y, random_forest_predictions)
random_forest_confusion_matrix = confusion_matrix(Te_y, random_forest_predictions)

print("Decision Tree Classifier Metrics:")
print("Accuracy:", decision_tree_accuracy)
print("Classification Report:\n", decision_tree_classification_report)
print("Confusion Matrix:\n", decision_tree_confusion_matrix)

print("\nRandom Forest Classifier Metrics:")
print("Accuracy:", random_forest_accuracy)
print("Classification Report:\n", random_forest_classification_report)
print("Confusion Matrix:\n", random_forest_confusion_matrix)


# In[ ]:




