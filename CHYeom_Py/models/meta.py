# models/meta.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

def build_meta_model_final(input_dim, num_classes=5):
    model = Sequential()
    # 전달받은 input_dim을 사용
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=Adam(learning_rate=0.005), 
                  metrics=['accuracy'])
    return model
