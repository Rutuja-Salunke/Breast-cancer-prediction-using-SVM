

### 1. Importing Packages
```python
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```
This section imports the necessary libraries and modules for the project. Key packages include scikit-learn for machine learning, Matplotlib for plotting, and Pandas for data manipulation.

### 2. Data Gathering
```python
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["Target"] = cancer.target
```
Here, the breast cancer dataset is loaded using scikit-learn's `load_breast_cancer` function. The data is then stored in a Pandas DataFrame (`df`), where features are taken from `cancer.data` and the target variable is added as a new column named "Target" using `cancer.target`.

### 3. Exploratory Data Analysis (EDA)
```python
print("DataFrame Shape:", df.shape)
print("\nDataFrame Info:")
df.info()
print("\nMissing Values:")
print(df.isna().sum())
```
The code prints information about the DataFrame, including its shape, general information, and the number of missing values in each column.

### 4. Model Training
```python
x = df.iloc[:, :30]
y = df["Target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

svm = SVC()
svm_model = svm.fit(x_train, y_train)
```
The dataset is split into training and testing sets using the `train_test_split` function. An SVM (Support Vector Machine) model is then created (`svm`) and trained on the training data.

### 5. Model Evaluation on Training Set
```python
y_train_pred = svm_model.predict(x_train)
print("\nTraining Set Metrics:")
print("Mean Squared Error =", mean_squared_error(y_train, y_train_pred))
print("Mean Absolute Error =", mean_absolute_error(y_train, y_train_pred))
print("R2 Score =", r2_score(y_train, y_train_pred))
```
The code evaluates the SVM model on the training set, calculating metrics such as mean squared error, mean absolute error, and R2 score.

### 6. Model Evaluation on Test Set
```python
y_test_pred = svm_model.predict(x_test)
print("\nTest Set Metrics:")
print("Mean Squared Error =", mean_squared_error(y_test, y_test_pred))
print("Mean Absolute Error =", mean_absolute_error(y_test, y_test_pred))
print("R2 Score =", r2_score(y_test, y_test_pred))
```
Similarly, the code evaluates the SVM model on the test set and prints the corresponding metrics.

### 7. Train model using two features and visualize decision boundary
```python
X_2d = cancer.data[:, :2]
svm_2d = SVC(kernel="rbf", gamma=0.5, C=1.0)
svm_2d.fit(X_2d, y)

# Plot Decision Boundary
DecisionBoundaryDisplay.from_estimator(
    svm_2d, X_2d, response_method="predict", cmap=plt.cm.Spectral, alpha=0.8,
    xlabel=cancer.feature_names[0], ylabel=cancer.feature_names[1],
)

# Scatter plot
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=20, edgecolors="k")
plt.show()
```
This section trains an SVM model with only two features (`cancer.data[:, :2]`) and visualizes the decision boundary using the `DecisionBoundaryDisplay` class. The scatter plot shows how the data points are distributed in the feature space, with different colors indicating different classes
