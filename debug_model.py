import numpy as np
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model

def debug_prediction():
    print("Loading model and scaler...")
    try:
        model = load_model('cnn_ecg_model.h5')
        scaler = joblib.load('scaler.pkl')
    except Exception as e:
        print(f"Error loading: {e}")
        return

    # Test cases: [RR, P, QRS, T, QT]
    test_cases = [
        ("User V-Beat",   [0.60, 0.08, 0.18, 0.25, 0.50]),
        ("Extreme PVC",   [0.50, 0.00, 0.20, 0.30, 0.40]), # Wide QRS, No P
        ("Normal Typical",[0.85, 0.10, 0.09, 0.18, 0.40]), # Standard
        ("Tachycardia",   [0.40, 0.08, 0.08, 0.15, 0.25]), # Fast heart
        ("Bradycardia",   [1.20, 0.12, 0.10, 0.20, 0.45]), # Slow heart
    ]

    print(f"\n{'Case Name':<15} | {'Raw Values':<35} | {'Scaled':<35} | {'Prob':<6} | {'Result'}")
    print("-" * 110)

    for name, values in test_cases:
        input_data = np.array([values])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data, verbose=0)[0][0]
        result = "Arrhythmia" if prediction > 0.5 else "Normal"
        
        raw_str = str(values)
        scaled_str = str(np.round(scaled_data[0], 2))
        
        print(f"{name:<15} | {raw_str:<35} | {scaled_str:<35} | {prediction:.4f} | {result}")

if __name__ == "__main__":
    debug_prediction()
