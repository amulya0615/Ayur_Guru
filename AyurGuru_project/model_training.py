import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
df = pd.read_csv('/C:/Users/rahul/OneDrive/Desktop/AyurGuru_project/model/data_filtered.csv')

# Encode categorical features
encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        encoders[column] = LabelEncoder()
        df[column] = encoders[column].fit_transform(df[column])

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Save the model and encoders to a .pkl file
with open('/C:/Users/rahul/OneDrive/Desktop/AyurGuru_project/model/model.pkl', 'wb') as f:
    pickle.dump({'model': rf_model, 'encoders': encoders, 'feature_names': X.columns.tolist()}, f)

print("Model training complete and saved to model.pkl")