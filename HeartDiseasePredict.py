# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
disease_df = pd.read_csv("/Users/anish/Downloads/framingham.csv")
disease_df.drop(['education'], inplace=True, axis=1)
disease_df.rename(columns={'male':'Sex_male'}, inplace=True)
disease_df.dropna(axis=0, inplace=True)

# Define features and target variable
X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])

# Normalize the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# Visualize the data using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Visualization of Heart Disease Data')
plt.colorbar()
plt.show()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

# Train a Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# Evaluate the Logistic Regression model
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print('Accuracy of Logistic Regression model is =', accuracy_logreg)

# Confusion matrix and classification report for Logistic Regression
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
conf_matrix_logreg = pd.DataFrame(data=cm_logreg, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix_logreg, annot=True, fmt='d', cmap="Blues")
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

print('The details for confusion matrix of Logistic Regression is =')
print(classification_report(y_test, y_pred_logreg))

# Train a Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print('Accuracy of Random Forest model is =', accuracy_rf)

# Confusion matrix and classification report for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
conf_matrix_rf = pd.DataFrame(data=cm_rf, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap="Greens")
plt.title('Confusion Matrix for Random Forest')
plt.show()

print('The details for confusion matrix of Random Forest is =')
print(classification_report(y_test, y_pred_rf))
