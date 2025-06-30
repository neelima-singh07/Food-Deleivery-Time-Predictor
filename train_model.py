import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
data = pd.read_csv("Food_Delivery_Times.csv")

# Rename column to match code (if needed)
data.rename(columns={"Delivery_Time_min": "Delivery_Time"}, inplace=True)

# Drop rows with missing target or features
data.dropna(subset=["Delivery_Time"], inplace=True)

# Define features and target
X = data.drop(columns=["Order_ID", "Delivery_Time"])
y = data["Delivery_Time"]

# Define categorical features
cat_features = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_features)
    ],
    remainder='passthrough'
)

# Pipeline: Preprocessing + Model
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit model
pipeline.fit(X_train, y_train)

# Save the pipeline
joblib.dump(pipeline, "delivery_model.pkl")

print("✅ Model trained and saved as delivery_model.pkl")
