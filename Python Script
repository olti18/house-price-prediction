# House Price Prediction - Python Script

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Simulimi i datasetit
np.random.seed(0)
df = pd.DataFrame({
    'OverallQual': np.random.randint(1, 10, 100),
    'GrLivArea': np.random.randint(500, 2500, 100),
    'GarageCars': np.random.randint(0, 4, 100),
    'TotalBsmtSF': np.random.randint(0, 1500, 100),
    'FullBath': np.random.randint(1, 4, 100),
    'YearBuilt': np.random.randint(1900, 2020, 100),
    'SalePrice': np.random.randint(50000, 500000, 100)
})

# Përgatitja e të dhënave
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelet
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Trajnimi dhe vlerësimi
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\n{name}")
    print(f"MSE: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
