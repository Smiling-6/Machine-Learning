# train_model.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # you can change to other models
from sklearn.metrics import mean_squared_error, r2_score

# Load Data
df = pd.read_csv("data.csv")
print("First 5 rows:")
print(df.head())



# Basic Info
print("\nInfo:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Drop non-numeric and irrelevant columns
df_numeric = df.drop(columns=["date", "street", "city", "statezip", "country"])

# Compute correlation matrix
correlation_matrix = df_numeric.corr().abs()

# Select upper triangle of correlation matrix
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Identify columns with correlation > 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

# Drop highly correlated features
df_reduced = df_numeric.drop(columns=to_drop)

# Define features and target
X = df_reduced.drop(columns=["price"])
y = df_reduced["price"]

print(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Output results
print("Highly Correlated Columns Removed:", to_drop)
print("R² Score:", r2)
print("Root Mean Squared Error (RMSE):", rmse)

# Page title
st.title("🏠 House Price Predictor (Linear Regression)")

st.markdown("Enter the details of the house below to predict its price.")

# Input features — ensure these match the features used in training
bedrooms = st.number_input("Number of Bedrooms", min_value=0.0, value=3.0)
bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, value=2.0)
sqft_living = st.number_input("Living Area (sqft)", min_value=0, value=1500)
sqft_lot = st.number_input("Lot Area (sqft)", min_value=0, value=4000)
floors = st.number_input("Number of Floors", min_value=0.0, value=1.0)
waterfront = st.selectbox("Waterfront View", options=[0, 1])
view = st.slider("View Rating", 0, 4, 0)
condition = st.slider("Condition Rating", 1, 5, 3)
sqft_above = st.number_input("Above Ground Area (sqft)", min_value=0, value=1500)
sqft_basement = st.number_input("Basment Area (sqft)", min_value=0, value=4000)
yr_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0, max_value=2025, value=0)

# Create DataFrame for prediction
input_df = pd.DataFrame({
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'sqft_living': [sqft_living],
    'sqft_lot': [sqft_lot],
    'floors': [floors],
    'waterfront': [waterfront],
    'view': [view],
    'condition': [condition],
    'sqft_above': [sqft_above],
    'sqft_basement':[sqft_basement],
    'yr_built': [yr_built],
    'yr_renovated': [yr_renovated]
})

# Predict
if st.button("Predict Price"):
    predicted_price = model.predict(input_df)[0]
    st.success(f"💰 Estimated House Price: ₹{predicted_price:,.0f}")

