import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score ,confusion_matrix
from sklearn.preprocessing import LabelEncoder

script_dir = os.path.dirname(os.path.abspath(__file__))

train_path= "kaggle/input/Churn_Modelling.csv"
training_data_path = os.path.join(script_dir, train_path)
df = pd.read_csv(training_data_path)
df

LE = LabelEncoder()
df["Gen"] = LE.fit_transform(df["Gender"])
df["Geo"] = LE.fit_transform(df["Geography"])

X = df[["RowNumber","CreditScore", "Geo", "Gen", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard","IsActiveMember","EstimatedSalary"]]
Y = df["Exited"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=42)

# Calculate counts of unique values in the "Exited" and "NumOfProducts" columns
exit_counts = df["Exited"].value_counts()
num_counts = df["NumOfProducts"].value_counts()

# Create a pie chart for "Exited" counts
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)  # Subplot for the pie chart
plt.pie(exit_counts, labels=["No", "YES"], autopct="%0.0f%%")
plt.title("Exited Counts")

# Create a bar chart for "NumOfProducts" counts
plt.subplot(1, 2, 2)  # Subplot for the bar chart
plt.bar(num_counts.index, num_counts.values, width=0.4)
plt.xlabel("Number of Products")
plt.ylabel("Count")
plt.title("Number of Products Counts")
plt.xticks(np.arange(0,5,1))
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

model = RandomForestClassifier()
model.fit(X_train, Y_train)

model.score(X_train, Y_train)

y_pred = model.predict(X_test)
y_pred

accuracy = accuracy_score(Y_test, y_pred)
print("Validation Accuracy:", accuracy)