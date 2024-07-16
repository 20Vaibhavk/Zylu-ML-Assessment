# generate_data.py

import pandas as pd
import numpy as np

def generate_data(num_customers=1000, num_products=50):
    np.random.seed(42)
    customers = np.arange(num_customers)
    products = np.arange(num_products)
    
    data = []
    for customer in customers:
        for product in products:
            purchase = np.random.choice([0, 1], p=[0.95, 0.05])
            if purchase == 1:
                data.append([customer, product, np.random.randint(1, 5), np.random.randint(10, 100)])
    
    df = pd.DataFrame(data, columns=['CustomerID', 'ProductID', 'Quantity', 'Price'])
    return df

df = generate_data()
df.to_csv('customer_data.csv', index=False)
print(df.head())
