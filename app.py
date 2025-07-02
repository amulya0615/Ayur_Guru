from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError("GROQ_API_KEY not found in environment variables.")
groq_client = Groq(api_key=api_key)

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("model/data_filtered.csv")

# Initialize label encoders for each categorical column
encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':  # If the column contains strings
        encoders[column] = LabelEncoder()
        df[column] = encoders[column].fit_transform(df[column])

# Split features and target
X = df.drop('class', axis=1)
y = df['class']

# Store feature names
feature_names = X.columns.tolist()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
}

metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': []
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
    metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
    metrics['f1_score'].append(f1_score(y_test, y_pred, average='weighted'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.get_json()
    
    # Create a DataFrame with the input features
    input_df = pd.DataFrame([features])
    
    # Transform input features using the saved encoders
    encoded_features = {}
    for column in feature_names:
        if column in encoders:
            encoded_features[column] = encoders[column].transform([features[column]])[0]
        else:
            encoded_features[column] = features[column]
    
    # Convert to DataFrame with correct feature names
    encoded_df = pd.DataFrame([encoded_features], columns=feature_names)
    
    # Make prediction
    rf_model = models['Random Forest']
    prediction = rf_model.predict(encoded_df)

    # Decode the prediction back to original class name
    predicted_class = encoders['class'].inverse_transform(prediction)[0]
    
    # Generate personalized recommendations using Groq API
    prompt = f"Generate personalized health and lifestyle recommendations for someone with {predicted_class} prakriti type, considering these characteristics: {features}"
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )
        recommendations = chat_completion.choices[0].message.content
    except Exception as e:
        recommendations = f"Failed to fetch recommendations: {str(e)}"
    
    # Instead of returning JSON, redirect to result page with query params
    import urllib.parse
    params = urllib.parse.urlencode({
        'prakriti': predicted_class,
        'recommendations': recommendations
    })
    return redirect(url_for('result') + '?' + params)

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/ai-doctor', methods=['GET', 'POST'])
def ai_doctor():
    if request.method == 'GET':
        return render_template('ai_doctor.html')
    if request.method == 'POST':
        data = request.get_json()
        question = data.get('question', '')
        prompt = f"You are an expert Ayurvedic doctor. Answer the following user question in a friendly, clear, and practical way, referencing Ayurvedic principles where relevant. If the question is not related to Ayurveda, politely decline.\n\nUser: {question}\n\nAI Doctor:" 
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
            )
            answer = chat_completion.choices[0].message.content
        except Exception as e:
            answer = f"Failed to fetch answer: {str(e)}"
        return jsonify({'answer': answer})

@app.route('/about')
def about():
    return render_template('about.html')
    
@app.route('/vata')
def vata():
    return render_template('vata.html')

@app.route('/pitta')
def pitta():
    return render_template('pitta.html')

@app.route('/kapha')
def kapha():
    return render_template('kapha.html')

@app.route('/metrics')
def metrics_page():
    return render_template('metrics.html')

@app.route('/metrics_data')
def metrics_data():
    return jsonify({
        'models': list(models.keys()),
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score']
    })

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=8100)