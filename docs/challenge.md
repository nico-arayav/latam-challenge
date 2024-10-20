# API Documentation

## Overview

This API is designed to predict flight delays using a machine learning model. The API is built using FastAPI and provides endpoints for health checks and making predictions.

## Endpoints

### Health Check Endpoint

- **URL**: `/health`
- **Method**: `GET`
- **Description**: This endpoint is used to check the health status of the API.
- **Response**:
  - **Status Code**: `200 OK`
  - **Body**:
    ```json
    {
        "status": "OK"
    }
    ```

### Prediction Endpoint

- **URL**: `/predict`
- **Method**: `POST`
- **Description**: This endpoint is used to predict flight delays.
- **Request Body**:
  - **Content-Type**: `application/json`
  - **Schema**:
    ```json
    {
        "flights": [
            {
                "OPERA": "Aerolineas Argentinas",
                "TIPOVUELO": "N",
                "MES": 3
            }
        ]
    }
    ```
  - **Fields**:
    - [`OPERA`](string): The airline operating the flight. Must be one of the valid airlines.
    - [`TIPOVUELO`](string): The type of flight. Must be either 'N' (national) or 'I' (international).
    - [`MES`](integer): The month of the flight. Must be between 1 and 12.
- **Response**:
  - **Status Code**: `200 OK`
  - **Body**:
    ```json
    {
        "predict": [0]
    }
    ```
- **Errors**:
  - **Status Code**: `400 Bad Request`
  - **Body**:
    ```json
    {
        "detail": [
            {
                "loc": ["body", "flights", 0, "OPERA"],
                "msg": "Invalid OPERA",
                "type": "value_error"
            }
        ]
    }
    ```

## Exceptions

- **RequestValidationError**: Raised when the request body does not conform to the expected schema.
  - **Status Code**: `400 Bad Request`
  - **Response**:
    ```json
    {
        "detail": [
            {
                "loc": ["body", "flights", 0, "OPERA"],
                "msg": "Invalid OPERA",
                "type": "value_error"
            }
        ]
    }
    ```

# Model Documentation

## Overview

The model used in this API is designed to predict flight delays. It is implemented using the `LogisticRegression` algorithm from scikit-learn. The model is trained on historical flight data and uses various features to make predictions.

## Features

The model uses the following top 10 features for prediction:
1. `OPERA_Latin American Wings`
2. `MES_7`
3. `MES_10`
4. `OPERA_Grupo LATAM`
5. `MES_12`
6. `TIPOVUELO_I`
7. `MES_4`
8. `MES_11`
9. `OPERA_Sky Airline`
10. `OPERA_Copa Air`

These features were determined by analyzing feature importance using a plot generated with the following code:
```python
plt.figure(figsize=(10, 5))
plot_importance(xgb_model)
```
This plot helped identify the most significant features contributing to the model's predictions.

## Preprocessing

The preprocessing steps include:
1. **Feature Engineering**: Creating new features such as `period_day`, `high_season`, and `min_diff`.
2. **One-Hot Encoding**: Converting categorical variables into numerical format using one-hot encoding.
3. **Handling Missing Values**: Setting default values for missing features.

## Training

The model is trained using the `LogisticRegression` algorithm with class weights to handle imbalanced data. The training process involves:
1. Splitting the data into training and validation sets.
2. Training the model on the training set.
3. Saving the trained model to disk.

## Prediction

The prediction process involves:
1. Loading the trained model from disk.
2. Preprocessing the input data to match the format used during training.
3. Making predictions using the trained model.

# Why LogisticRegression with Balance Instead of XGBoost with Balance

1. **Simplicity**: Logistic Regression is simpler and easier to interpret compared to XGBoost. It provides clear insights into the relationship between features and the target variable.
2. **Performance**: For this specific problem, Logistic Regression with balanced class weights performs adequately. It is computationally less expensive and faster to train compared to XGBoost.
3. **Imbalanced Data Handling**: Logistic Regression with class weights effectively handles imbalanced data, which is common in flight delay prediction. It ensures that the model does not bias towards the majority class.
4. **Deployment**: Logistic Regression models are generally smaller and require fewer resources to deploy, making them suitable for environments with limited computational resources.

By choosing Logistic Regression with balanced class weights, we achieve a good balance between model performance, interpretability, and computational efficiency.