import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('../Notebooks/heart_cleveland_upload.csv')

# Nominal columns
nominal_cols = ['cp', 'restecg', 'slope', 'thal']

# Encode
df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

# Split
X = df_encoded.drop('condition', axis=1)
y = df_encoded['condition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Models
models = {
    'lr': LogisticRegression(max_iter=1000, random_state=42),
    'dt': DecisionTreeClassifier(max_depth=5, random_state=42),
    'rf': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train models
for key, model in models.items():
    model.fit(X_train_sc, y_train)

def predict_patient(patient_dict, model_key):
    """
    Predict for a single patient.
    
    patient_dict: dict with keys: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    model_key: 'lr', 'dt', or 'rf'
    """
    patient_df = pd.DataFrame([patient_dict])
    
    # Encode nominal columns
    patient_encoded = pd.get_dummies(patient_df, columns=nominal_cols, drop_first=True)
    
    # Align columns to match training data
    patient_encoded = patient_encoded.reindex(columns=X.columns, fill_value=0)
    
    # Scale
    patient_sc = scaler.transform(patient_encoded)
    
    # Predict
    model = models[model_key]
    pred = model.predict(patient_sc)[0]
    prob = model.predict_proba(patient_sc)[0][1]  # Probability of positive class
    
    risk_label = "🔴 HIGH RISK" if prob > 0.5 else "🟢 LOW RISK"
    
    return {
        'model': model_key.upper(),
        'prediction': int(pred),
        'probability': round(prob, 2),
        'risk_label': risk_label
    }

def evaluate_all(csv_file, model_key):
    """
    Evaluate model on a new CSV file.
    
    csv_file: path to CSV with same structure as training data, including 'condition' column
    model_key: 'lr', 'dt', or 'rf'
    """
    new_df = pd.read_csv(csv_file)
    
    # Encode
    new_encoded = pd.get_dummies(new_df.drop('condition', axis=1), columns=nominal_cols, drop_first=True)
    
    # Align columns
    new_encoded = new_encoded.reindex(columns=X.columns, fill_value=0)
    
    # Scale
    new_sc = scaler.transform(new_encoded)
    
    y_true = new_df['condition']
    
    model = models[model_key]
    y_pred = model.predict(new_sc)
    
    print(f"Evaluation on {csv_file} using {model_key.upper()}:")
    print(classification_report(y_true, y_pred))