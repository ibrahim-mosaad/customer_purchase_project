import pandas as pd
import os

# URL of dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"

# Load dataset from URL
df = pd.read_csv(url)

# Show first rows
print(df.head())

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save locally
file_path = "data/online_shoppers.csv"
df.to_csv(file_path, index=False)

print(f"Dataset saved successfully at: {file_path}")