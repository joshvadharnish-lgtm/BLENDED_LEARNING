# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Dataset
Import required libraries and read the car price dataset.
2. Preprocess Data
Handle missing values, encode categorical data, and separate features and target variable.
3. 
Split Data
Divide the dataset into training and testing sets.
4. Train Model
Create and fit the Linear Regression model using training data.
5.Predict & Evaluate
Predict car prices and evaluate using MSE, RMSE, and R² score.
6.
Test Assumptions
Verify linearity, normality, homoscedasticity, and independence using residual plots.

## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

x=df[["enginesize","horsepower","citympg","highwaympg"]]
y=df["price"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
model = LinearRegression()
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)
print('Name: G.DHARNISH')
print('Reg NO: 25004380')
print("MODEL COEFFICIENTS:")
for feature, coef in zip(x.columns, model.coef_):
    print(f"{feature:}: {coef:}")
print(f"Intercept: {model.intercept_:}") 

print("\nMODEL PERFORMANCE:")
print(f"MSE: {mean_squared_error(y_test,y_pred):}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test,y_pred)):}")
print(f"R-squared: {r2_score(y_test,y_pred):}")
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Linearity Check: Actual vs Predicted Price")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()
residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-watson statistic:{dw_test:.2f}",
      "\n(values close to 2 indicate no autocorrelation)")
      
plt.figure(figsize=(10, 5)) 
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted") 
plt.xlabel("Predicted Price ($)") 
plt.ylabel("Residuals ($)")
plt.grid(True) 
plt.show() 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) 
sns.histplot(residuals, kde=True, ax=ax1) 
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()

```

## Output:
<img width="433" height="298" alt="image" src="https://github.com/user-attachments/assets/7d135a83-5442-4c6c-aec3-c7cc61858cbd" />
<img width="1230" height="562" alt="image" src="https://github.com/user-attachments/assets/4d4010e7-c095-44a6-89cc-3ecbf3379e0b" />
<img width="1347" height="592" alt="image" src="https://github.com/user-attachments/assets/56104948-2895-4e94-8eb5-65fb53cd56e4" />
<img width="1299" height="507" alt="image" src="https://github.com/user-attachments/assets/4387d62b-12ad-4a68-8495-dbbb2a068c8a" />






## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
