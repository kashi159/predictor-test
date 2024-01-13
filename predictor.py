import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load your combined dataset with score and rank ranges
combined_data = pd.read_csv('data.csv')

# Split the range values into minimum and maximum
combined_data[['score_min', 'score_max']] = combined_data['score_range'].str.split('-', expand=True)
combined_data[['rank_min', 'rank_max']] = combined_data['rank_range'].str.split('-', expand=True)

# Convert the new columns to numeric values
combined_data['score_min'] = pd.to_numeric(combined_data['score_min'])
combined_data['score_max'] = pd.to_numeric(combined_data['score_max'])
combined_data['rank_min'] = pd.to_numeric(combined_data['rank_min'])
combined_data['rank_max'] = pd.to_numeric(combined_data['rank_max'])

# Encode categorical variables
combined_data = pd.get_dummies(combined_data, columns=['category', 'academic_year'], drop_first=True)

# Assuming 'score_min', 'score_max', and the encoded features are relevant
X = combined_data[['score_min', 'score_max', 'category_SC', 'category_OBC', 'academic_year_2022-2023']]
y = combined_data['rank_min']  # Use 'rank_min' instead of 'rank_min', 'rank_max'

# Train-test split for time series data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model (adjust for rank range prediction)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# Now, you can use the trained model to predict rank for new data
new_data = pd.DataFrame({'score_min': [75], 'score_max': [75], 'category_SC': [1], 'category_OBC': [0], 'academic_year_2022-2023': [1]})
predicted_rank = model.predict(new_data)

# Ensure the shape of the predicted_rank array
if predicted_rank.ndim == 1:
    # Single prediction
    print(f'Predicted Rank: {predicted_rank[0]}')
else:
    print(f'Unexpected Prediction Format: {predicted_rank}')