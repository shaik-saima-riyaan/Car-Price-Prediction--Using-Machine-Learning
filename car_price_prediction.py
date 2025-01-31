import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('used_cars.csv')

# Visualizing missing values before preprocessing
missing_values_before = data.isnull().sum()
plt.figure(figsize=(10, 6))
missing_values_before[missing_values_before > 0].plot(kind='bar', color='salmon')
plt.title("Missing Values Before Preprocessing")
plt.ylabel("Count of Missing Values")
plt.xlabel("Columns")
plt.show()

# Filling missing values
data = data.fillna("NA")

# Converting 'milage' and 'price' to numeric
data['milage'] = data['milage'].str.replace(' mi.', '', regex=False).str.replace(',', '', regex=False).astype(float)
data['price'] = data['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

# Visualizing missing values after preprocessing
missing_values_after = data.isnull().sum()
plt.figure(figsize=(10, 6))
missing_values_after.plot(kind='bar', color='lightgreen')
plt.title("Missing Values After Preprocessing")
plt.ylabel("Count of Missing Values")
plt.xlabel("Columns")
plt.show()

# Heatmap of missing values
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap After Preprocessing")
plt.show()

# Price distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], kde=True, color="skyblue")
plt.title("Car Price Distribution")
plt.xlabel("Price")
plt.show()

# Mileage vs Price scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['milage'], y=data['price'], color="purple")
plt.title("Mileage vs Price")
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.show()

# Fuel Type vs Price boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='fuel_type', y='price', data=data)
plt.title("Fuel Type vs Price")
plt.xlabel("Fuel Type")
plt.ylabel("Price")
plt.show()

# Pairplot of Price, Mileage, and Model Year
sns.pairplot(data[['price', 'milage', 'model_year']])
plt.suptitle("Pairplot of Price, Mileage, and Model Year", y=1.02)
plt.show()

# Correlation matrix heatmap
numeric_data = data.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Encoding categorical variables
data = pd.get_dummies(data, drop_first=True)

# Splitting data into features and target
X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dictionary to store model results
results = {}

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
results['Linear Regression'] = {
    'MSE': mean_squared_error(y_test, y_pred_lr),
    'R2 Score': r2_score(y_test, y_pred_lr)
}

# Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
results['Decision Tree'] = {
    'MSE': mean_squared_error(y_test, y_pred_dt),
    'R2 Score': r2_score(y_test, y_pred_dt)
}

# Support Vector Regressor (SVR) - Replacing Random Forest
svr_model = SVR()
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)
results['Support Vector Regressor'] = {
    'MSE': mean_squared_error(y_test, y_pred_svr),
    'R2 Score': r2_score(y_test, y_pred_svr)
}

# Gradient Boosting Regressor (GBR) - Replacing XGBoost
gbr_model = GradientBoostingRegressor(random_state=42)
gbr_model.fit(X_train, y_train)
y_pred_gbr = gbr_model.predict(X_test)
results['Gradient Boosting'] = {
    'MSE': mean_squared_error(y_test, y_pred_gbr),
    'R2 Score': r2_score(y_test, y_pred_gbr)
}

# Model Performance Comparison
print("Model Performance Comparison:")
for model, metrics in results.items():
    print(f"{model}:")
    print(f"  Mean Squared Error (MSE): {metrics['MSE']:.2f}")
    print(f"  R-squared Score (R2): {metrics['R2 Score']:.2f}")
    print(f"  Accuracy: {metrics['Accuracy']:.2f}%")
    print()

# Bar plots for MSE and R2 Score comparison
mse_scores = [metrics['MSE'] for metrics in results.values()]
r2_scores = [metrics['R2 Score'] for metrics in results.values()]
model_names = list(results.keys())

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(model_names, mse_scores, color=['blue', 'green', 'orange', 'purple'])
plt.title("Mean Squared Error (MSE) Comparison")
plt.ylabel("MSE")

plt.subplot(1, 2, 2)
plt.bar(model_names, r2_scores, color=['blue', 'green', 'orange', 'purple'])
plt.title("R-squared Score (R2) Comparison")
plt.ylabel("R2 Score")

plt.suptitle("Model Comparison")
plt.tight_layout()
plt.show()

# Scatterplot of actual vs predicted prices for different models
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5, label='Linear Regression', color='blue')
plt.scatter(y_test, y_pred_dt, alpha=0.5, label='Decision Tree', color='green')
plt.scatter(y_test, y_pred_svr, alpha=0.5, label='Support Vector Regressor', color='orange')
plt.scatter(y_test, y_pred_gbr, alpha=0.5, label='Gradient Boosting', color='purple')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal Fit')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Car Prices for Different Models")
plt.legend()
plt.show()

# Function to calculate accuracy within a certain percentage of the actual values
def calculate_accuracy(y_true, y_pred, threshold=0.10):
    within_threshold = np.abs(y_pred - y_true) <= (threshold * y_true)
    accuracy = np.mean(within_threshold) * 100  # Convert to percentage
    return accuracy

# Adding accuracy calculation to results
for model_name, y_pred in [('Linear Regression', y_pred_lr),
                           ('Decision Tree', y_pred_dt),
                           ('Support Vector Regressor', y_pred_svr),
                           ('Gradient Boosting', y_pred_gbr)]:
    accuracy = calculate_accuracy(y_test, y_pred)
    results[model_name]['Accuracy'] = accuracy

# Printing results with accuracy
print("Model Performance with Accuracy Comparison:")
for model, metrics in results.items():
    print(f"{model}:")
    print(f"  Mean Squared Error (MSE): {metrics['MSE']:.2f}")
    print(f"  R-squared Score (R2): {metrics['R2 Score']:.2f}")
    print(f"  Accuracy: {metrics['Accuracy']:.2f}%")
    print()

# Plotting the accuracy comparison
accuracies = [metrics['Accuracy'] for metrics in results.values()]

plt.figure(figsize=(10, 6))

plt.bar(model_names, accuracies, color=['blue', 'green', 'orange', 'purple'])
plt.title("Accuracy Comparison of Different Models")
plt.ylabel("Accuracy (%)")
plt.show()
