# textCNN.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam


# 파라미터들은 필요에 맞게 수정
vocab_size = 10000
embedding_dim = 256
max_len = 150

def build_textcnn_model(kernel_size, dropout):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(Conv1D(192, kernel_size=kernel_size, activation='relu', kernel_regularizer=regularizers.l2(0.005)))    
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(GlobalMaxPooling1D())
    model.add(Dense(192, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(Dense(5, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_textcnn_ensemble(train_X, train_y, val_X, val_y, callbacks=None):
    # 모델 3개 생성
    from callbacks import get_callbacks
    callbacks = get_callbacks()
    
    model1 = build_textcnn_model(kernel_size=3, dropout=0.3)
    model2 = build_textcnn_model(kernel_size=4, dropout=0.4)
    model3 = build_textcnn_model(kernel_size=5, dropout=0.5)
    
    # 학습 및 callbacks 설정 등은 생략 (필요시 추가)
    model1.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=15, batch_size=32, callbacks=callbacks)
    model2.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=15, batch_size=32, callbacks=callbacks)
    model3.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=15, batch_size=32, callbacks=callbacks)
    
    # 예측 후 메타 데이터 생성
    pred_train1 = model1.predict(train_X)
    pred_train2 = model2.predict(train_X)
    pred_train3 = model3.predict(train_X)
    meta_train = np.concatenate([pred_train1, pred_train2, pred_train3], axis=1)
    
    pred_val1 = model1.predict(val_X)
    pred_val2 = model2.predict(val_X)
    pred_val3 = model3.predict(val_X)
    meta_val = np.concatenate([pred_val1, pred_val2, pred_val3], axis=1)
    
    return meta_train, meta_val, (model1, model2, model3)

def build_meta_model_textCNN(input_dim=15):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
    from tensorflow.keras import regularizers
    from tensorflow.keras.optimizers import Adam

    meta_model = Sequential()
    meta_model.add(Dense(128, activation='gelu', input_shape=(input_dim,), kernel_regularizer=regularizers.l2(0.01)))
    meta_model.add(BatchNormalization())
    meta_model.add(Dropout(0.4))
    meta_model.add(Dense(64, activation='gelu', kernel_regularizer=regularizers.l2(0.01)))
    meta_model.add(BatchNormalization())
    meta_model.add(Dropout(0.4))
    meta_model.add(Dense(32, activation='gelu', kernel_regularizer=regularizers.l2(0.01)))
    meta_model.add(BatchNormalization())
    meta_model.add(Dropout(0.4))
    meta_model.add(Dense(5, activation='softmax'))
    
    meta_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.005), metrics=['accuracy'])
    return meta_model

# 필요한 다른 함수들도 여기에 추가 가능

# 메타모델 객체를 전역 변수로 정의해두고 싶다면 아래와 같이 정의할 수도 있습니다.
meta_model_textCNN = build_meta_model_textCNN(input_dim=15)
