import sys
import joblib

print("Starting script...", flush=True)
if len(sys.argv) < 2:
    print(0)
    sys.exit(1)

query = sys.argv[1]
print(f"Received query: {query}", flush=True)

model = joblib.load('sql_injection_model.pkl')
print("Model loaded.", flush=True)

pred = model.predict([query])
print(f"Prediction: {pred[0]}", flush=True)
print(pred[0])
