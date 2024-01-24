import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import re  # used for pattern matching and text manipulation.
import string 
import nltk #a powerful library for working with human language data.
import os
from nltk.corpus import stopwords #for cleaning 
from nltk.stem import LancasterStemmer ##for cleaning 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

script_dir = os.path.dirname(os.path.abspath(__file__))

train_path= "kaggle/Genre Classification Dataset/train_data.txt"
training_data_path = os.path.join(script_dir, train_path)
train_data = pd.read_csv(training_data_path, sep=":::", names=["TITLE", "GENRE", "DESCRIPTION"], engine="python")

train_data

train_data.info()

train_data.describe()

train_data.isnull().sum()

test_path= "kaggle/Genre Classification Dataset/test_data.txt"
testing_data_path = os.path.join(script_dir, test_path)
test_data = pd.read_csv(testing_data_path, sep=":::", names=["ID","TITLE","DESCRIPTION"], engine="python")

test_data

test_data.info()

plt.figure(figsize=(10,15))
sns.countplot(data=train_data, y="GENRE", order= train_data["GENRE"].value_counts().index)
plt.show()

plt.figure(figsize=(27,7))
sns.countplot(data=train_data, x="GENRE", order= train_data["GENRE"].value_counts().index, palette = "YlGnBu")
plt.show()

stemmer = LancasterStemmer()
stop_words = set(stopwords.words("english"))  # Stopwords set

def cleaning_data(text):
    text = text.lower()
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'.pic\S+', '', text)
    text = re.sub(r'[^a-zA-Z+]', ' ', text)  # Change to replace non-characters with a space
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    # Use the predefined stop_words variable instead of redefining it inside the function
    text = " ".join([i for i in words if i not in stop_words and len(i) > 2])
    text = re.sub(r"\s+", " ", text).strip()  # Replace multiple spaces with a single space
    return text

train_data["TextCleaning"] = train_data["DESCRIPTION"].apply(cleaning_data)
test_data["TextCleaning"] = test_data["DESCRIPTION"].apply(cleaning_data)

train_data

vectorize = TfidfVectorizer()

X_train = vectorize.fit_transform(train_data["TextCleaning"])

X_test = vectorize.transform(test_data["TextCleaning"])

X = X_train
y = train_data["GENRE"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

model = SVC()
model.fit(X_train, Y_train)

model.score(X_train, Y_train)

y_pred = model.predict(X_test)
y_pred

accuracy = accuracy_score(Y_test, y_pred)
print("Validation Accuracy:", accuracy)