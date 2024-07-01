import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Load the dataset
file_path = 'data/annual_notifs_2023-07-24.csv'
nepal_data = pd.read_csv(file_path)

# Filter data for Nepal (iso3 code for Nepal is 'NPL')
nepal_data = nepal_data[nepal_data['iso3'] == 'NPL']

# Prepare the data
nepal_data = nepal_data[['year', 'c_newinc', 'e_pop_num']].dropna()

# Feature and target variables
X = nepal_data[['year']]
y = nepal_data['c_newinc']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict future values (next 10 years)
future_years = np.array([2024 + i for i in range(10)]).reshape(-1, 1)
predictions = model.predict(future_years)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(nepal_data['year'], nepal_data['c_newinc'], marker='o', label='Actual Data')
plt.plot(future_years, predictions, marker='x', linestyle='--', color='r', label='Predicted Data')
plt.title('Annual TB Notifications in Nepal')
plt.xlabel('Year')
plt.ylabel('Number of New TB Cases')
plt.ylim(min(nepal_data['c_newinc'].min(), predictions.min()) - 1000, 
         max(nepal_data['c_newinc'].max(), predictions.max()) + 1000)
plt.legend()
plt.grid(True)
plt.show()

# Display future predictions
future_predictions = pd.DataFrame({'Year': future_years.flatten(), 'Predicted TB Cases': predictions})
print(future_predictions)
