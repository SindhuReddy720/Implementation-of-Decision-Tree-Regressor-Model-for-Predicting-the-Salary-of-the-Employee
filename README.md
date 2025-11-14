# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas 

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset. 

4.calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Pelleti Sindhu Sri 
RegisterNumber: 212224240113
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import files

# Upload the file
uploaded = files.upload()

# Load the dataset
df = pd.read_csv('Salary.csv')

# Check the data structure
print(df.head())

# Use 'Level' as feature and 'Salary' as target
X = df[['Level']]  # Feature(s) - Independent variable(s)
y = df['Salary']   # Target variable

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=42)

# Train the model
regressor.fit(X_train, y_train)

# Predict on test set
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Visualize results
plt.scatter(df['Level'], y, color='blue', label='Actual Salary')
plt.scatter(X_test, y_pred, color='red', label='Predicted Salary (Test Set)')
plt.title('Decision Tree Regressor - Salary Prediction')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Predict salary for a new employee level (example: level 3.5)
new_level = np.array([[3.5]])
predicted_salary = regressor.predict(new_level)
print(f"Predicted salary for employee with level {new_level[0][0]} is: {predicted_salary[0]:.2f}")


```

## Output:
<img width="949" height="863" alt="image" src="https://github.com/user-attachments/assets/540736c1-96c7-41ba-87f0-dcfeef19fe30" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
