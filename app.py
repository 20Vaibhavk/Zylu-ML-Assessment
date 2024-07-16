# app.py

from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Load models and data
model = pickle.load(open('return_model.pkl', 'rb'))
model_repeat = pickle.load(open('repeat_purchase_model.pkl', 'rb'))
df = pd.read_csv('customer_data.csv')

app = Flask(__name__)

def recommend_products(customer_id, top_n=5):
    customer_data = df[df['CustomerID'] == customer_id]
    product_revenue = df.groupby('ProductID')['TotalSpent'].sum().sort_values(ascending=False)
    recommendations = product_revenue.index[:top_n].tolist()
    return recommendations

@app.route('/predict_return', methods=['POST'])
def predict_return():
    customer_data = request.json
    features = pd.DataFrame(customer_data, index=[0])
    prediction = model.predict(features)
    return jsonify({'return': int(prediction[0])})


@app.route('/predict_repeat_purchase', methods=['POST'])
def predict_repeat_purchase():
    purchase_data = request.json
    features = pd.DataFrame(purchase_data, index=[0])
    prediction = model_repeat.predict(features)
    return jsonify({'repeat_purchase': int(prediction[0])})

@app.route('/recommend_products', methods=['POST'])
def recommend():
    customer_id = request.json['customer_id']
    recommendations = recommend_products(customer_id)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
