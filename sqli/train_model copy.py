import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np


# Load dataset
df = pd.read_csv('csic_database.csv')

# Check column existence and handle missing values
if 'URL' not in df.columns or 'classification' not in df.columns:
    raise ValueError("The CSV file must contain 'URL' and 'classification' columns.")

# Correct Column
X = df['URL'].astype(str) # URL column instead of Query
X = np.nan_to_num(X.values, nan='', copy=False) # Handle missing values

# Convert strings to lowercase
X = [text.lower() for text in X]

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['classification'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Added random_state for reproducibility

# Pipeline: vectorizer + classifier
model = Pipeline([
    ('tfidf', TfidfVectorizer()), #You can add ngram_range=(1,2)
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42)) # Added random_state for reproducibility
])

# Train and save model
model.fit(X_train, y_train)
print("Model accuracy:", model.score(X_test, y_test))
joblib.dump(model, 'sql_injection_model.pkl')
