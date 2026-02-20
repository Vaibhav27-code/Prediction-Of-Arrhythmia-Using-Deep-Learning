"""
train_model.py
--------------
Downloads the MIT-BIH Arrhythmia Database from PhysioNet and trains a
neural network on real ECG feature data.

Uses fast geometric feature extraction from R-peaks (no slow delineation).

Outputs:
    cnn_ecg_model.h5  – trained Keras model
    scaler.pkl         – fitted StandardScaler

Usage:
    python train_model.py
"""

import os, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = './mitbih_data'
MODEL_PATH  = './cnn_ecg_model.h5'
SCALER_PATH = './scaler.pkl'
FS          = 360   # MIT-BIH sample rate (Hz)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ARRHYTHMIA_LABELS = {'A','a','J','S','V','F','f','!','E','/','x','Q','[',']'}
NORMAL_LABELS     = {'N','L','R','e','j'}

RECORDS = [
    '100','101','102','103','104','105','106','107',
    '108','109','111','112','113','114','115','116',
    '117','118','119','121','122','123','124','200',
    '201','202','203','205','207','208','209','210',
    '212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]


# ── Signal helpers ────────────────────────────────────────────────────────────
def bandpass(signal, low=0.5, high=40.0, fs=FS):
    b, a = butter(3, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, signal)


def fast_features(signal, ann_samples, ann_symbols, fs=FS):
    """
    Compute 5 features per beat using fast geometric estimation from R-peaks.
    No slow delineation — all durations are inferred from RR and beat morphology.
    """
    rows = []
    n = len(ann_samples)

    for i, (r_sample, symbol) in enumerate(zip(ann_samples, ann_symbols)):
        if symbol not in NORMAL_LABELS and symbol not in ARRHYTHMIA_LABELS:
            continue
        label = 1 if symbol in ARRHYTHMIA_LABELS else 0

        # ── RR interval ───────────────────────────────────────────────────
        if i > 0 and ann_symbols[i-1] in (NORMAL_LABELS | ARRHYTHMIA_LABELS):
            rr = (r_sample - ann_samples[i-1]) / fs
        elif i < n-1 and ann_symbols[i+1] in (NORMAL_LABELS | ARRHYTHMIA_LABELS):
            rr = (ann_samples[i+1] - r_sample) / fs
        else:
            rr = 0.80
        rr = float(np.clip(rr, 0.3, 2.0))

        # ── Beat window: 100 ms before R, 300 ms after ────────────────────
        w_start = max(0, r_sample - int(0.10 * fs))
        w_end   = min(len(signal), r_sample + int(0.30 * fs))
        beat    = signal[w_start:w_end]

        if len(beat) < 20:
            continue

        r_idx = r_sample - w_start  # R-peak index within window

        # ── QRS complex: width of R-peak above 50% threshold ─────────────
        r_amp   = signal[r_sample]
        thresh  = 0.5 * abs(r_amp)
        above   = np.where(np.abs(beat) > thresh)[0]
        if len(above) >= 2:
            qrs = (above[-1] - above[0]) / fs
        else:
            qrs = 0.09
        qrs = float(np.clip(qrs, 0.05, 0.30))

        # ── P-wave: energy in window 200–50 ms before R ───────────────────
        p_start = max(0, r_sample - int(0.20 * fs))
        p_end   = max(0, r_sample - int(0.05 * fs))
        p_seg   = signal[p_start:p_end]
        if len(p_seg) > 5:
            p_energy = np.max(np.abs(p_seg)) / (abs(r_amp) + 1e-9)
            p_wave   = float(np.clip(0.08 + 0.05 * p_energy, 0.05, 0.25))
        else:
            p_wave = 0.10

        # ── T-wave: 150–350 ms after R peak ──────────────────────────────
        t_start = min(len(signal), r_sample + int(0.15 * fs))
        t_end   = min(len(signal), r_sample + int(0.35 * fs))
        t_seg   = signal[t_start:t_end]
        if len(t_seg) > 5:
            t_energy = np.max(np.abs(t_seg)) / (abs(r_amp) + 1e-9)
            t_wave   = float(np.clip(0.10 + 0.08 * t_energy, 0.05, 0.40))
        else:
            t_wave = 0.16

        # ── QT interval: Bazett with correction ──────────────────────────
        qt = float(np.clip(0.39 * np.sqrt(rr), 0.20, 0.70))

        rows.append({
            'rr_interval': rr,
            'p_wave':      p_wave,
            'qrs_complex': qrs,
            't_wave':      t_wave,
            'qt_interval': qt,
            'label':       label
        })

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=" * 60)
    print("  MIT-BIH Arrhythmia Model Trainer  (fast mode)")
    print("=" * 60)

    all_rows = []

    for i, rec_id in enumerate(RECORDS, 1):
        print(f"[{i:2d}/{len(RECORDS)}] Record {rec_id}...", end=' ', flush=True)
        rec_path = os.path.join(DATA_DIR, rec_id)

        # Download if needed
        if not os.path.exists(rec_path + '.dat'):
            try:
                wfdb.dl_files('mitdb', DATA_DIR,
                    [f'{rec_id}.dat', f'{rec_id}.hea', f'{rec_id}.atr'])
            except Exception as e:
                print(f"SKIP ({e})")
                continue

        try:
            record = wfdb.rdrecord(rec_path)
            ann    = wfdb.rdann(rec_path, 'atr')
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        signal = bandpass(record.p_signal[:, 0])
        rows   = fast_features(signal, ann.sample, ann.symbol)
        all_rows.extend(rows)
        print(f"{len(rows)} beats")

    if not all_rows:
        print("\n[ERROR] No data extracted. Check internet and try again.")
        return

    # ── Create DataFrame ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    print(f"\nTotal beats extracted: {len(df)}")

    # ── Balance Classes (Undersample Normal) ──────────────────────────────────
    df_normal = df[df['label'] == 0]
    df_arr = df[df['label'] == 1]

    print(f"\nOriginal counts:")
    print(f"Normal: {len(df_normal)}")
    print(f"Arrhythmia: {len(df_arr)}")

    # Balance to 1:1 ratio
    n_samples = len(df_arr)
    if n_samples > 0:
        df_normal_balanced = df_normal.sample(n=n_samples, random_state=RANDOM_SEED)
        df_balanced = pd.concat([df_normal_balanced, df_arr])
        df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        print("[ERROR] No arrhythmia samples found!")
        return

    print(f"\nBalanced counts:")
    print(f"Normal: {(df_balanced['label']==0).sum()}")
    print(f"Arrhythmia: {(df_balanced['label']==1).sum()}")

    X = df_balanced[['rr_interval', 'p_wave', 'qrs_complex', 't_wave', 'qt_interval']].values
    y = df_balanced['label'].values

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved  → {SCALER_PATH}")

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Sequential([
        Dense(128, activation='relu', input_shape=(5,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
    ]

    print("\nTraining…")
    model.fit(X_train, y_train,
              epochs=100, batch_size=256,
              validation_split=0.15,
              callbacks=callbacks, verbose=1)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Test Results")
    print("=" * 60)
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    print(classification_report(y_test, y_pred,
                                target_names=['Normal','Arrhythmia']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\nModel saved   → {MODEL_PATH}")
    print("Done! Restart app.py to use the new model.")


if __name__ == '__main__':
    main()
