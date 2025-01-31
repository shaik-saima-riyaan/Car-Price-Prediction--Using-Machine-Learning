# First, ensure you import the necessary libraries
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

# Assuming you have already loaded your dataset 'data'
data = pd.read_csv('used_cars.csv')

# Step 1: Clean 'milage' column by removing ' mi.' and commas
data['milage'] = data['milage'].str.replace(' mi.', '', regex=False).str.replace(',', '', regex=False)

# Step 2: Convert 'milage' to numeric, after cleaning
data['milage'] = pd.to_numeric(data['milage'], errors='coerce')

# Step 3: Handle missing values in 'milage' using SimpleImputer
imputer_num = SimpleImputer(strategy='mean')
data['milage'] = imputer_num.fit_transform(data[['milage']])

# Now continue with the rest of the preprocessing
