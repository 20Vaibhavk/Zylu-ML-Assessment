# train_models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load and preprocess data
df = pd.read_csv('customer_data.csv')

df['TotalSpent'] = df['Quantity'] * df['Price']
customer_features = df.groupby('CustomerID').agg({
    'TotalSpent': 'sum',
    'ProductID': 'count'
}).rename(columns={'ProductID': 'NumPurchases'}).reset_index()

# Train model for customer return prediction
customer_features['Return'] = np.random.choice([0, 1], size=len(customer_features), p=[0.7, 0.3])
X = customer_features[['TotalSpent', 'NumPurchases']]
y = customer_features['Return']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Return Prediction Model")
print(classification_report(y_test, y_pred))

# Save the return prediction model
with open('return_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Train model for repeat purchase prediction
df['RepeatPurchase'] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])
X = df[['CustomerID', 'ProductID', 'Quantity', 'Price']]
y = df['RepeatPurchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_repeat = RandomForestClassifier(n_estimators=100, random_state=42)
model_repeat.fit(X_train, y_train)
y_pred = model_repeat.predict(X_test)
print("Repeat Purchase Prediction Model")
print(classification_report(y_test, y_pred))

# Save the repeat purchase prediction model
with open('repeat_purchase_model.pkl', 'wb') as f:
    pickle.dump(model_repeat, f)
