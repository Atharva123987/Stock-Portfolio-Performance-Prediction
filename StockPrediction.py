import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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

# Plot actual vs predicted values for the last 2 years of data
last_two_years = aapl.iloc[-504:, :]
last_two_years["lr_predicted"] = lr.predict(X)[-504:]
last_two_years["dt_predicted"] = dt.predict(X)[-504:]
last_two_years["rf_predicted"] = rf.predict(X)[-504:]

plt.figure(figsize=(12, 6))
plt.plot(last_two_years["Close"], label="Actual")
plt.plot(last_two_years["lr_predicted"], label="Linear Regression")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(last_two_years["Close"], label="Actual")
plt.plot(last_two_years["dt_predicted"], label="Decision Tree Regressor")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(last_two_years["Close"], label="Actual")
plt.plot(last_two_years["rf_predicted"], label="Random Forest Regressor")
plt.legend()
plt.show()

