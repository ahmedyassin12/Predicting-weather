import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
dataset_url = 'weatherHistory.csv'  # Include the file extension
df = pd.read_csv(dataset_url)

# Convert the 'Formatted Date' column to datetime format
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'])

# Extract numerical representation of datetime (Unix timestamp)
df['Timestamp'] = df['Formatted Date'].apply(lambda x: x.timestamp())

# Train the Linear Regression model
model = LinearRegression()
model.fit(df[['Timestamp']], df['Temperature (C)'])

# Streamlit app
st.title("Weather Prediction App")

# User input for date
user_date = st.date_input("Enter a date:")

# Convert the input date to datetime format
user_date = datetime.combine(user_date, datetime.min.time())

# Make prediction
user_date_timestamp = user_date.timestamp()
prediction = model.predict([[user_date_timestamp]])[0]

# Display the prediction
st.write(f"Predicted Temperature on {user_date.strftime('%Y-%m-%d')}: {prediction:.2f}°C")

# Visualize historical data
st.header("Historical Temperature Data")

# Display a line chart of historical temperature data
st.line_chart(df.set_index('Formatted Date')['Temperature (C)'])

# Show descriptive statistics of the dataset
st.subheader("Dataset Statistics")
st.write(df.describe())

# Show correlation heatmap (excluding non-numeric columns)
st.subheader("Correlation Heatmap")
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_columns].corr()

# Create a figure and axes
fig, ax = plt.subplots()

# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5, ax=ax)

# Display the figure using st.pyplot()
st.pyplot(fig)

# Add a disclaimer
st.info("Note: This is a simple example using linear regression. Real-world weather prediction involves more complex models and considerations.")
