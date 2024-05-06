import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# load the dataset
data_path = '/home/jiajun/project/data/data/data/data_fakes.csv'
data = pd.read_csv(data_path)

# display the detail of dataset, exploration of data
# print(data.head())
# print(data.shape)
# print(data.info())

# check for missing values in the dataset
missing_values = data.isnull().sum()

# check data types 
data_types = data.dtypes
# print(missing_values, data_types)

#data cleanning 
# fill missing values for AveragePrice and ConfirmedPrice with their mean
data['AveragePrice'] = data['AveragePrice'].fillna(data['AveragePrice'].mean())
data['ConfirmedPrice'] = data['ConfirmedPrice'].fillna(data['ConfirmedPrice'].mean())
# fill null Brand , DeliveryCategory, top_review with 'Unknown'
data['Brand'] = data['Brand'].fillna('Unknown')
data['DeliveryCategory'] = data['DeliveryCategory'].fillna('Unknown')
data['top_review'] = data['top_review'].fillna('')
# drop rows with missing values in the PurchDate column
data = data.dropna(subset=['PurchDate'])
data['PurchDate'] = pd.to_datetime(data['PurchDate'], errors='coerce')

# encoding categorical variables to numerical values
label_encoder = LabelEncoder()
categorical_columns = ['VNST', 'Category', 'Brand', 'FullfillmentType', 'DeliveryCategory']

for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# print(data.head())
# dropping columns that are not needed for training the model, (use feature selection techniques to identify the most important features in the future)!
model_data = data.drop(columns=['RefId', 'PurchDate', 'ProductName', 'product_description', 'top_review'])

#split the data into features and target variable
X = model_data.drop('Fake', axis=1)
y = model_data['Fake']


# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# initializing and training the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# predict the target variable
y_pred = rf_classifier.predict(X_test)

# evaluate the model
# accuracy
accuracy = accuracy_score(y_test, y_pred)
# precision, recall, fscore, support
print(classification_report(y_test, y_pred))

# show aoc curve of the rf model
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# use SVM to see this performance
svm_classifier = SVC(probability=True)  
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

# Evaluation
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print('SVM Accuracy:', accuracy_svm)
report_svm = classification_report(y_test, y_pred_svm)
print(report_svm)


# sorry I have no more time to do more and improve it
# here are some ideas for further improvement
# 1. do feature selection (Filter methods,Wrapper methods,Embedded methods)
# 2. do grid search for get best hyperparameters of models
# 3. show more chart to show distrusbution of data / performance of models
# 4. try more diferent models or even other emsemble learning models, becasue these 2 are still to simple to find the best trade-off of bias and variance 



