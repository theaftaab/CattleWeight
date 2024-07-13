import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

data = pd.read_csv("train_data.csv")  # Replace with your data loading method

# Define the features (independent variables) and target (dependent variable)
features = ["side_length_shoulderbone", "side_f_girth", "side_r_girth", "rear_width", "cow_pixels", "sticker_pixels"]
target = "weight"
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

for i in range(1, 5):
    # Define the degree of the polynomial (experiment with different values)
    degree = i  # This creates quadratic terms

    # Create a PolynomialFeatures object
    poly = PolynomialFeatures(degree=degree)

    # Transform the training data to include polynomial terms
    X_train_poly = poly.fit_transform(X_train)

    # Create the polynomial regression model
    model_poly = LinearRegression()

    # Train the model on the transformed training data
    model_poly.fit(X_train_poly, y_train)

    # Use the trained model for prediction on test data (already transformed)
    y_pred_poly = model_poly.predict(poly.transform(X_test))
    r_square_poly = r2_score(y_test, y_pred_poly)
    print(f"R-squared for polynomial regression (degree {degree}): {r_square_poly}")
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

    # Train the XGBoost model
    xgb_model.fit(X_train, y_train)

    # Use the trained model for prediction on test data
    y_pred_xgb = xgb_model.predict(X_test)

    # Evaluate the XGBoost model
    r_square_xgb = r2_score(y_test, y_pred_xgb)
    print(f"R-squared for XGBoost regression: {r_square_xgb}")

