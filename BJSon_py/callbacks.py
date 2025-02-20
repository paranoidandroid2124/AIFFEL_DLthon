# callbacks.py
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def get_callbacks():
    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    return [es, lr]
