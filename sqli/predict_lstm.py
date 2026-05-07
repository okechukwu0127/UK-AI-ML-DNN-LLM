import sys
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('sql_injection_lstm_model.h5')
tokenizer = joblib.load('tokenizer.pkl')


if len(sys.argv) < 2:
    print("Usage: python3 predict_lstm.py '<sql_query>'")
    sys.exit(1)

query = sys.argv[1]

#query = sys.argv[1]
seq = tokenizer.texts_to_sequences([query])
padded = pad_sequences(seq, padding='post', maxlen=50)

pred = model.predict(padded)
print(1 if pred[0][0] > 0.5 else 0)
