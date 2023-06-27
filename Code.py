import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
# Read data from CSV
data = pd.read_csv('data.csv')
# Split data into training and testing sets
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Perform Decision Tree Regression
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
predicted_y_train = regressor.predict(X_train)
predicted_y_test = regressor.predict(X_test)
# Visualize the results
plt.scatter(range(len(y_train)), y_train, color='blue', label='Actual (Train)')
plt.scatter(range(len(y_train)), predicted_y_train, color='red', label='Predicted (Train)')
plt.scatter(range(len(y_train), len(y_train) + len(y_test)), y_test, color='green', label='Actual (Test)')
plt.scatter(range(len(y_train), len(y_train) + len(y_test)), predicted_y_test, color='orange', label='Predicted (Test)')
plt.xlabel('Sample')
plt.ylabel('Crop Yield')
plt.title('Decision Tree Regression')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
