import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Remove any fixed seed for truly dynamic randomness
np.random.seed(None)

def generate_sample_data(num_rows=30000):
    # Base parameters
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days
    
    # Generate customer base (3000 unique customers)
    customer_ids = np.arange(1, 3001)
    customer_segments = np.random.choice(['High-Value', 'At-Risk', 'Budget', 'Seasonal'], 
                                       size=3000, p=[0.1, 0.3, 0.4, 0.2])
    
    # Create lists for storing values
    dates = []
    amounts = []
    customer_list = []
    
    # Generate transactional data
    for _ in range(num_rows):
        # Select a customer
        cust_id = np.random.choice(customer_ids)
        segment = customer_segments[cust_id-1]  # IDs start at 1
        
        # Generate a purchase date with a beta distribution for recency trends
        days_offset = np.random.beta(a=1.5, b=3) * date_range
        purchase_date = start_date + timedelta(days=days_offset)
        
        # Base amount based on customer segment
        base_amount = {
            'High-Value': np.random.normal(300, 50),
            'At-Risk': np.random.normal(80, 20),
            'Budget': np.random.normal(50, 15),
            'Seasonal': np.random.normal(150, 40)
        }[segment]
        
        # Add yearly growth (3% per year)
        years_since_start = (purchase_date - start_date).days / 365
        growth_factor = 1.03 ** years_since_start
        
        # Add monthly seasonality (peak in December)
        month_factor = 1 + 0.3 * np.sin(2 * np.pi * purchase_date.month / 12)
        
        # Weekend effect: slightly higher amounts on weekends
        weekend_factor = 1.1 if purchase_date.weekday() >= 5 else 1.0
        
        # Final amount calculation with added noise
        final_amount = base_amount * growth_factor * month_factor * weekend_factor
        final_amount += np.random.normal(0, 10)  # noise
        
        # Save the values
        dates.append(purchase_date)
        amounts.append(abs(final_amount))
        customer_list.append(cust_id)
    
    # Create and sort the DataFrame by PurchaseDate
    df = pd.DataFrame({
        'CustomerID': customer_list,
        'PurchaseDate': dates,
        'AmountSpent': np.round(amounts, 2)
    })
    
    df = df.sort_values('PurchaseDate').reset_index(drop=True)
    return df

# Generate and save sample data dynamically
sample_data = generate_sample_data(30000)
sample_data.to_csv('retail_transactions_sample.csv', index=False)
print("Sample dataset generated with 30,000 rows!")
