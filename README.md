# Customer Prediction and Recommendation API

## Overview

This project provides a Flask API for predicting the likelihood of a customer returning to a departmental store, the likelihood of them buying the same product/service again, and recommending products that will increase the store revenue.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- flask

## Installation

1. Clone the repository and navigate to the project directory.
2. Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Generate Synthetic Data:**

    ```bash
    python generate_data.py
    ```

    This will create a `customer_data.csv` file with synthetic customer purchase data.

2. **Train the Models:**

    ```bash
    python train_models.py
    ```

    This will train the models and save them as `return_model.pkl` and `repeat_purchase_model.pkl`.

3. **Run the Flask API:**

    ```bash
    python app.py
    ```

    The API will be available at `http://127.0.0.1:5000`.

## API Endpoints

### Predict Customer Return

- **URL:** `/predict_return`
- **Method:** `POST`
- **Request Body:**

    ```json
    {
        "TotalSpent": 300,
        "NumPurchases": 5
    }
    ```

- **Response:**

    ```json
    {
        "return": 1
    }
    ```

### Predict Repeat Purchase

- **URL:** `/predict_repeat_purchase`
- **Method:** `POST`
- **Request Body:**

    ```json
    {
        "CustomerID": 1,
        "ProductID": 10,
        "Quantity": 2,
        "Price": 30
    }
    ```

- **Response:**

    ```json
    {
        "repeat_purchase": 0
    }
    ```

### Recommend Products

- **URL:** `/recommend_products`
- **Method:** `POST`
- **Request Body:**

    ```json
    {
        "customer_id": 1
    }
    ```

- **Response:**

    ```json
    {
        "recommendations": [1, 2, 3, 4, 5]
    }
    ```

## Testing the API

### Using Postman

1. Set the request method to `POST`.
2. Enter the appropriate request URL.
3. Go to the `Body` tab and select `raw`.
4. Set the format to `JSON`.
5. Enter the request body JSON.
6. Click `Send`.

## Conclusion

This project demonstrates how to build a machine learning model to predict customer behavior and expose it via a Flask API.
