# CodSoft_MachineLearning
Welcome to the repository containing my solutions for the **"Machine Learning Internship"** offered by CodSoft. Here, you'll find a comprehensive collection of my completed tasks, showcasing the skills and knowledge acquired throughout the internship.

## Task - 1 : Movie Genre Classification
### Objective
Develop a robust machine learning model capable of accurately predicting a movie's genre based on textual information.

### Techniques
Utilize advanced text processing techniques, including TF-IDF and word embeddings, along with well-established classifiers like Naive Bayes, Logistic Regression, or Support Vector Machines.

### Dataset
The project utilizes the [Genre Classification Dataset from IMDb](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb) available on Kaggle. This dataset provides the necessary information for training and testing the genre classification model.

## Task - 2 : Credit Card Fraud Detection
### Objective
Build a robust machine learning model for detecting fraudulent credit card transactions. The objective is to implement algorithms, such as Logistic Regression and Decision Trees, to classify transactions as legitimate or fraudulent. The project includes exploratory data analysis, data preprocessing, feature engineering, upsampling, model building, and hyperparameter tuning.

### Techniques
* **Exploratory Data Analysis (EDA)**
> In-depth exploration of the dataset, analyzing features like gender, state, city, job, etc.

> Visualizations to understand patterns in fraudulent transactions.

* **Data Preprocessing**
> Conversion of date-related columns to the appropriate format.

> Removal of unnecessary columns and encoding of categorical variables.

* **Feature Engineering**
> Creation of additional features (age, trans_month, latitudinal_distance, longitudinal_distance) to enhance model performance.

* **Upsampling**
> Upsampling of the minority class (fraudulent transactions) to address class imbalance.

* **Model Building**
> Implementation of Logistic Regression and Decision Tree Classifier models.

> Calculation of model performance metrics: accuracy, mean absolute error, F1 score.

* **Model Evaluation**
> Confusion matrices and classification reports for a detailed overview of model performance.

> Visualization of Receiver Operating Characteristic (ROC) curves.

* **Hyperparameter Tuning**
> Utilization of Randomized Search to find optimal hyperparameters for the Decision Tree model.

### Dataset
The project employs [Credit Card Transactions Fraud Detection Dataset.](https://www.kaggle.com/datasets/kartik2112/fraud-detection) The dataset includes information about credit card 
transactions, and the goal is to experiment with algorithms for accurate classification of transactions as either fraudulent or legitimate.

## Task - 3 : Customer Churn Prediction
### Objective
Develop a predictive model for customer churn in a subscription-based service or business. The goal is to utilize historical customer data, including usage behavior and demographic features, to forecast customer churn accurately. The project employs algorithms such as Logistic Regression, Random Forests, or Gradient Boosting to predict churn and enhance customer retention strategies.

### Techniques
* **Data Preprocessing**
> Label encoding for categorical features (Gender, Geography) to convert them into numeric format.

> Selection of relevant features for model training.

* **Exploratory Data Analysis (EDA)**
> Visualization of churn counts and distribution of the number of products.

> Insights into the distribution of exited customers.

* **Model Building**
> Implementation of a Random Forest Classifier to predict customer churn.

> Evaluation of model performance using accuracy scores.

### Dataset
The project employs [Bank Customer Churn Prediction.](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction) The dataset contains information about customers, including features like credit score, geography, gender, age, tenure, balance, number of products, credit card status, activity status, and estimated salary. The aim is to leverage this data to develop a robust model for predicting customer churn in subscription-based services or businesses.

## Task - 4 : Spam SMS Detection
### Objective
Develop an AI model for classifying SMS messages as either spam or legitimate. The project aims to employ techniques such as TF-IDF or word embeddings along with classifiers like Naive Bayes, Logistic Regression, or Support Vector Machines to effectively identify spam messages. The primary objective is to enhance SMS filtering systems and improve user experience by minimizing unwanted messages.

### Techniques
* **Data Preprocessing**
> Removal of unnecessary columns (Unnamed: 2, Unnamed: 3, Unnamed: 4) from the dataset.

> Renaming columns to "label" and "text" for clarity.

> Mapping labels ("spam" and "ham") to binary values for model training.

* **Exploratory Data Analysis (EDA)**
> Visualization of spam and ham counts.

> Analysis of message lengths for spam and ham messages.

* **Text Processing**
> Tokenization, removal of punctuation, and removal of stopwords to clean and prepare text data.

> Word cloud visualization for understanding common words in spam and ham messages.

* **Model Building**
> Implementation of classifiers such as Multinomial Naive Bayes, Support Vector Machines, K-Nearest Neighbors, Stochastic Gradient Descent, and Gradient Boosting.

> Model evaluation using accuracy scores, confusion matrices, and classification reports.

### Dataset
The project utilizes [SMS Spam Collection Dataset.](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) The dataset contains SMS messages labeled as "spam" or "ham" (legitimate). The objective is to leverage this data to build a robust AI model capable of accurately classifying SMS messages and distinguishing between spam and legitimate content.
