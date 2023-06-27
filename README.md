I used creditcard dataset from kaggle.

Project Report: Credit Card Fraud Detection using Support Vector Machine (SVM)

1. Introduction
In this project, we aim to develop a credit card fraud detection model using Support Vector Machine (SVM). The dataset used for training and testing the model is sourced from a CSV file containing credit card transactions.

2. Data Pre-processing
   
2.1 Loading the Data
The dataset is loaded using the Pandas library, and the CSV file is read into a DataFrame.

df = pd.read_csv('/content/creditcard.csv')

2.2 Handling Missing Values
We check for missing values in the target variable 'Class' (indicating fraud or normal transactions) and drop the corresponding rows from both the feature matrix (X) and the target variable (y).

if y.isnull().sum() > 0:
    missing_indices = y[y.isnull()].index
    X = X.drop(missing_indices)
    y = y.drop(missing_indices)
    
2.3 Handling Missing Values in Features
We use SimpleImputer from the scikit-learn library to replace missing values in the feature matrix (X) with the mean of the available values.

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

2.4 Feature Scaling
To ensure that all features are on the same scale, we use StandardScaler from scikit-learn to standardize the feature matrix (X).

scalar = StandardScaler()
X = scalar.fit_transform(X)

2.5 Train-Test Split
The pre-processed data is split into training and testing sets using the train_test_split function from scikit-learn. The testing set size is set to 20% of the total data, and a random seed is used for reproducibility.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

3. Model Training and Evaluation
   
3.1 Support Vector Machine (SVM)
We train a Support Vector Machine (SVM) model using the SVC class from scikit-learn.

model_svc = SVC()
model_svc.fit(X_train, y_train)

3.2 Model Evaluation
We evaluate the trained SVM model by calculating the training and testing accuracies.

train_score = model_svc.score(X_train, y_train)
test_score = model_svc.score(X_test, y_test)
print("Training Accuracy:", train_score)
print("Testing Accuracy:", test_score)

4. Results and Analysis
   
4.1 Prediction
We use the trained SVM model to make predictions on the testing set.

y_predict = model_svc.predict(X_test)

4.2 Confusion Matrix and Classification Report
We generate a confusion matrix and a classification report to assess the performance of the model.

cm = confusion_matrix(y_test, y_predict, labels=[1, 0])
confusion = pd.DataFrame(cm, index=['is Fraud', 'is Normal'], columns=['Predicted Fraud', 'Predicted Normal'])
sns.heatmap(confusion, annot=True, fmt='d')
plt.show()
print(classification_report(y_test, y_predict))

The confusion matrix provides information about the true positive, false positive, true negative, and false negative predictions. The classification report provides
