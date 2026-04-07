import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("train.csv")

# Use only required columns
data = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

# Remove empty values
data = data.dropna()

# Inputs and output
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict price
result = model.predict([[2000, 3, 2]])

print("Predicted House Price:", result)