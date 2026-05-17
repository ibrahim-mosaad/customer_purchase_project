from data_preprocessing import load_data, clean_data
from feature_engineering import create_features
from train_model import train_model
from evaluate import evaluate_model

# Load
df = load_data("data/online_shoppers.csv")

# Clean
df = clean_data(df)

# Features
df = create_features(df)

# Train
model, X_test, y_test = train_model(df)

# Evaluate
evaluate_model(model, X_test, y_test)