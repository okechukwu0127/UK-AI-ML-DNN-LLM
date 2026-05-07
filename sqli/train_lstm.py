import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder

# Load and prepare data
df = pd.read_csv('csic_database.csv')
texts = df['URL'].astype(str).values   # Use 'URL' column
# Handle missing values in 'texts' by filling with an empty string
texts = np.nan_to_num(texts, nan='', copy=False)

# Convert strings to lowercase
texts = [text.lower() for text in texts]

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['classification']) #Numericalise the labels

# Tokenize
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, padding='post', maxlen=50)

# Save tokenizer
joblib.dump(tokenizer, 'tokenizer.pkl')

X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2)

# Define model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=50))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('sql_injection_lstm_model.h5')
