import pandas as pd
import numpy as np
import os

def generate_indian_housing_data(num_samples=2500):
    np.random.seed(42)
    
    locations = ['Mumbai', 'Bangalore', 'Delhi', 'Chennai', 'Pune', 'Hyderabad']
    property_types = ['Apartment', 'Villa', 'Independent House']
    parking_options = ['Yes', 'No']
    
    data = {
        'Location': np.random.choice(locations, num_samples),
        'Property_Type': np.random.choice(property_types, num_samples, p=[0.7, 0.15, 0.15]),
        'Area_sqft': np.random.randint(500, 4500, num_samples),
        'Age_years': np.random.randint(0, 30, num_samples),
        'Parking': np.random.choice(parking_options, num_samples, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Sensible constraints
    df['Bedrooms'] = df['Area_sqft'].apply(lambda x: max(1, min(5, int(x / 600) + np.random.randint(-1, 2))))
    df['Bathrooms'] = df['Bedrooms'].apply(lambda x: max(1, x - np.random.randint(0, 2)))
    df['Floors'] = df['Property_Type'].apply(lambda x: 1 if x == 'Apartment' else np.random.randint(1, 4))
    
    # Price Generation Logic (Realistic base multipliers in INR)
    location_multiplier = {'Mumbai': 15000, 'Bangalore': 9000, 'Delhi': 11000, 'Chennai': 7000, 'Pune': 6500, 'Hyderabad': 7500}
    type_multiplier = {'Apartment': 1.0, 'Villa': 1.5, 'Independent House': 1.3}
    
    prices = []
    for _, row in df.iterrows():
        base = row['Area_sqft'] * location_multiplier[row['Location']] * type_multiplier[row['Property_Type']]
        parking_premium = 500000 if row['Parking'] == 'Yes' else 0
        age_depreciation = row['Age_years'] * 0.01 * base
        noise = np.random.normal(0, base * 0.1) # 10% market variance
        
        final_price = base + parking_premium - age_depreciation + noise
        prices.append(max(1500000, final_price)) # Minimum 15 Lakhs
        
    df['Price_INR'] = prices
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/house_prices.csv', index=False)
    print("✅ Synthetic dataset created at 'data/house_prices.csv'")

if __name__ == "__main__":
    generate_indian_housing_data()
