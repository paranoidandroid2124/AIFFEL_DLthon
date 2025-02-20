# BiLSTM.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

# 파라미터들 (textCNN과 동일하거나 필요에 따라 조정)
vocab_size = 10000
embedding_dim = 256
max_len = 130

def build_BiLSTM_model(lstm_units, num_layers):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    
    for i in range(num_layers):
        # 각 층마다 LSTM 유닛 수를 조정 (예: 감소하는 형태)
        model.add(Bidirectional(LSTM(int(lstm_units / (2 ** i)), return_sequences=(i < num_layers - 1))))
        model.add(Dropout(0.3))
    
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(5, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_bilstm_ensemble(train_X, train_y, val_X, val_y, callbacks=None):
    # 모델 3개 생성 (여기서는 num_layers나 lstm_units를 조금씩 달리할 수 있습니다)
    
    from callbacks import get_callbacks
    callbacks = get_callbacks()
    
    model1 = build_BiLSTM_model(lstm_units=128, num_layers=2)
    model2 = build_BiLSTM_model(lstm_units=128, num_layers=3)
    model3 = build_BiLSTM_model(lstm_units=128, num_layers=2)
    
    callbacks = get_callbacks()
    
    model1.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=15, batch_size=32, callbacks=callbacks)
    model2.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=15, batch_size=32, callbacks=callbacks)
    model3.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=15, batch_size=32, callbacks=callbacks)
    
    pred_train1 = model1.predict(train_X)
    pred_train2 = model2.predict(train_X)
    pred_train3 = model3.predict(train_X)
    meta_train = np.concatenate([pred_train1, pred_train2, pred_train3], axis=1)
    
    pred_val1 = model1.predict(val_X)
    pred_val2 = model2.predict(val_X)
    pred_val3 = model3.predict(val_X)
    meta_val = np.concatenate([pred_val1, pred_val2, pred_val3], axis=1)
    
    return meta_train, meta_val, (model1, model2, model3)

def build_meta_model_BiLSTM(input_dim=15):
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

# 전역 변수로 미리 정의할 수도 있습니다.
meta_model_BiLSTM = build_meta_model_BiLSTM(input_dim=15)
