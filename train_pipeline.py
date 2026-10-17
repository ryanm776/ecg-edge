import os
import numpy as np
import scipy.signal as signal
import wfdb
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.regularizers import l2

# ==========================================
# 1. DATA ACQUISITION & 2D STFT PROCESSING
# ==========================================
# Expanded 30+ patient split for zero-cost edge generalization
train_records = ['101', '106', '108', '112', '114', '115', '118', '119', '200', '201', '203', '205', '208', '209', '210', '212', '213', '215', '219', '221', '222', '223']
val_records   = ['103', '109', '111', '124', '220', '230', '231', '232']
test_records  = ['100', '105', '113', '121', '123', '202', '214', '233']

# Binary mapping: 0 = Normal, 1 = Abnormal (Absorbs both V and S classes)
symbol_map = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  
    'V': 1, 'E': 1,                         
    'A': 1, 'a': 1, 'J': 1, 'S': 1           
}

def load_dual_input_spectrogram_data(record_list, max_normal_per_rec=300):
    X_spec_list, X_rr_list, y_list = [], [], []
    WINDOW_SIZE = 250
    HALF_WINDOW = WINDOW_SIZE // 2
    
    for rec in record_list:
        try:
            record = wfdb.rdrecord(rec, pn_dir='mitdb')
            annotation = wfdb.rdann(rec, pn_dir='mitdb', extension='atr')
        except ValueError:
            continue # Skip if record missing online
            
        fs = record.fs
        
        # Basic 0.5 - 40 Hz Bandpass Filter to remove baseline wander
        nyq = 0.5 * fs
        b, a = signal.butter(2, [0.5/nyq, 40.0/nyq], btype='band')
        filtered = signal.filtfilt(b, a, record.p_signal[:, 0])
        
        samples = annotation.sample
        symbols = annotation.symbol
        rr_intervals = np.diff(samples) / fs
        normal_count = 0
        
        for i in range(1, len(samples) - 1):
            sample_idx = samples[i]
            symbol = symbols[i]
            
            if symbol in symbol_map and (sample_idx - HALF_WINDOW >= 0) and (sample_idx + HALF_WINDOW < len(filtered)):
                target_class = symbol_map[symbol]
                
                # Cap Normal beats per record to maintain class balance
                if target_class == 0:
                    if normal_count >= max_normal_per_rec:
                        continue
                    normal_count += 1
                
                # Extract 1D Window
                window = filtered[sample_idx - HALF_WINDOW : sample_idx + HALF_WINDOW]
                
                # Compute 2D Spectrogram (STFT)
                # fs=360, nperseg=32, noverlap=16 -> yields a ~ (17 x 15) 2D matrix
                f, t, Sxx = signal.spectrogram(window, fs=fs, nperseg=32, noverlap=16)
                
                # Log-scale magnitude to mimic visual perception and normalize
                Sxx_log = np.log1p(Sxx * 1000) 
                s_min, s_max = np.min(Sxx_log), np.max(Sxx_log)
                norm_spec = (Sxx_log - s_min) / (s_max - s_min) if s_max - s_min > 0 else Sxx_log
                
                # Feature Engineering: Inter-beat Rhythm Timing
                pre_rr = rr_intervals[i - 1]
                post_rr = rr_intervals[i]
                start_k = max(0, i - 10)
                local_mean_rr = np.mean(rr_intervals[start_k:i])
                rr_ratio = pre_rr / local_mean_rr if local_mean_rr > 0 else 1.0
                
                X_spec_list.append(norm_spec)
                X_rr_list.append([pre_rr, post_rr, local_mean_rr, rr_ratio])
                y_list.append(target_class)
                
    # Add channel dimension for 2D Conv layer -> shape: (Samples, 17, 15, 1)
    return (np.array(X_spec_list, dtype=np.float32)[..., np.newaxis], 
            np.array(X_rr_list, dtype=np.float32), 
            np.array(y_list, dtype=np.int32))

print("Extracting 2D STFT Spectrograms and R-R timing features...")
X_train_spec, X_train_rr, y_train = load_dual_input_spectrogram_data(train_records)
X_val_spec, X_val_rr, y_val       = load_dual_input_spectrogram_data(val_records)
X_test_spec, X_test_rr, y_test    = load_dual_input_spectrogram_data(test_records)

print(f"Train Set: {len(X_train_spec)} | Val Set: {len(X_val_spec)} | Test Set: {len(X_test_spec)}")

# Balance loss function penalties
class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights_array))

# Determine exact spectrogram matrix shape dynamically
spec_shape = X_train_spec.shape[1:]

# ==========================================
# 2. DUAL-INPUT 2D-CNN ARCHITECTURE (ReLU6)
# ==========================================
def build_base_model(spec_shape):
    # Branch 1: 2D-CNN for Spectrogram Morphology
    spec_input = tf.keras.layers.Input(shape=spec_shape, name="spectrogram_input")
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu6', kernel_regularizer=l2(0.001))(spec_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu6', kernel_regularizer=l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)

    # Branch 2: Dense Sub-Network for R-R Rhythm Timing
    rr_input = tf.keras.layers.Input(shape=(4,), name="rr_input")
    y = tf.keras.layers.Dense(8, activation='relu6')(rr_input)

    # Merge Branches
    combined = tf.keras.layers.concatenate([x, y])
    z = tf.keras.layers.Dense(64, activation='relu6', kernel_regularizer=l2(0.001))(combined)
    z = tf.keras.layers.Dropout(0.40)(z)
    
    # Binary Output
    output = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(z)
    
    return tf.keras.models.Model(inputs=[spec_input, rr_input], outputs=output)

base_model = build_base_model(spec_shape)

# ==========================================
# 3. QUANTIZATION-AWARE TRAINING (QAT)
# ==========================================
# Wrap the base model with tfmot quantization nodes to simulate 8-bit precision during training
qat_model = tfmot.quantization.keras.quantize_model(base_model)

qat_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

qat_model.summary()

# ==========================================
# 4. TRAINING PIPELINE
# ==========================================
early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

print("\nStarting Quantization-Aware Training (QAT)...")
history = qat_model.fit(
    x=[X_train_spec, X_train_rr],
    y=y_train,
    validation_data=([X_val_spec, X_val_rr], y_val),
    epochs=100,
    batch_size=64, # Slightly larger batch size for stability with STFT data
    class_weight=class_weights_dict,
    callbacks=[early_stop_cb]
)

# ==========================================
# 5. EVALUATION & MATPLOTLIB PLOTS
# ==========================================
print("\n--- Model Evaluation on Unseen Test Patients ---")
y_pred_probs = qat_model.predict([X_test_spec, X_test_rr])
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Abnormal (1)'], zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, linestyle='--')
plt.title('QAT Model Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
plt.title('QAT Model Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# 6. INT8 EXPORT FOR ARDUINO NANO ESP32
# ==========================================
# Because we used QAT, the model weights are already simulated at INT8. 
# We just need to pass it through the converter to lock it in.
def representative_data_gen():
    for i in range(100):
        sample_spec = X_train_spec[i:i+1].astype(np.float32)
        sample_rr = X_train_rr[i:i+1].astype(np.float32)
        yield [sample_spec, sample_rr]

converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_int8_model)

def export_to_c_header(tflite_bytes, filename="model_data.h"):
    with open(filename, "w") as f:
        f.write("#ifndef MODEL_DATA_H_\n#define MODEL_DATA_H_\n\n")
        f.write(f"const unsigned int g_model_len = {len(tflite_bytes)};\n")
        f.write("const unsigned char g_model[] __attribute__((aligned(4))) = {\n  ")
        for i, byte in enumerate(tflite_bytes):
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n  ")
        f.write("\n};\n\n#endif  // MODEL_DATA_H_\n")

export_to_c_header(tflite_int8_model)
print(f"\nPipeline Complete! INT8 QAT 2D-CNN Model Size: {len(tflite_int8_model) / 1024:.2f} KB")