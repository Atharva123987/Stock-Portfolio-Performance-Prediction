import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Download data for Apple (AAPL) from 2010-01-01 to 2022-04-10
aapl = yf.download("AAPL", start="2010-01-01", end="2022-04-10")

# Create a new dataframe with only the 'Close' price column
df = pd.DataFrame(aapl['Close'])
print(df.head())

# Add columns for previous days' closing prices (lags)
for i in range(1, 6):
    df[f"lag_{i}"] = df["Close"].shift(i)

# Drop rows with missing values (due to lags)
df = df.dropna()

# Split the data into training and testing sets
X = df.drop("Close", axis=1)
y = df["Close"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate a linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"Linear Regression R^2: {lr.score(X_test, y_test):.4f}")

# Train and evaluate a decision tree regressor model
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
print(f"Decision Tree Regressor R^2: {dt.score(X_test, y_test):.4f}")

# Train and evaluate a random forest regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest Regressor R^2: {rf.score(X_test, y_test):.4f}")

# Plot predicted and actual values for linear regression model
y_pred_lr = lr.predict(X_test)
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred_lr, label='Predicted')
plt.title('Linear Regression Model')
plt.legend()
plt.show()

# Plot predicted and actual values for decision tree regressor model
y_pred_dt = dt.predict(X_test)
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred_dt, label='Predicted')
plt.title('Decision Tree Regressor Model')
plt.legend()
plt.show()

# Plot predicted and actual values for random forest regressor model
y_pred_rf = rf.predict(X_test)
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred_rf, label='Predicted')
plt.title('Random Forest Regressor Model')
plt.legend()
plt.show()

# Create a confusion matrix for the random forest regressor model
y_pred_binary = [1 if pred > 0 else 0 for pred in y_pred_rf] # Convert predicted values to binary 0/1
y_test_binary = [1 if val > 0 else 0 for val in y_test] # Convert actual values to binary 0/1


cm = confusion_matrix(y_test_binary, y_pred_binary)
print(f"Confusion Matrix:\n{cm}")
