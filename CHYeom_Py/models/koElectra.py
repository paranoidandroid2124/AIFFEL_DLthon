import numpy as np
import tensorflow as tf
from transformers import ElectraTokenizer, TFElectraForSequenceClassification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils as keras_utils

def custom_unpack_x_y_sample_weight(data):
    """
    data가 (x, y) 또는 (x, y, sample_weight) 형태일 때,
    각각 x, y, sample_weight (없으면 None)을 반환합니다.
    """
    if isinstance(data, (list, tuple)):
        if len(data) == 3:
            return data[0], data[1], data[2]
        elif len(data) == 2:
            return data[0], data[1], None
    return data, None, None

# 반드시 다른 keras 또는 transformers 임포트 전에 실행되어야 합니다.
keras_utils.unpack_x_y_sample_weight = custom_unpack_x_y_sample_weight

# koElectra 토크나이저 전역 객체
electra_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")

def electra_tokenize_function(texts, max_len):
    """
    주어진 텍스트 리스트를 koElectra 모델에 맞게 토큰화합니다.
    """
    return electra_tokenizer(
        texts.tolist(),  # 리스트 형태로 변환
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="tf",
        return_attention_mask=True
    )

def encode_tf_dataset(input_ids, attention_mask, labels):
    """
    입력 id, 어텐션 마스크, 라벨로 TensorFlow 데이터셋을 생성합니다.
    """
    return tf.data.Dataset.from_tensor_slices((
        {"input_ids": input_ids, "attention_mask": attention_mask},
        tf.convert_to_tensor(labels, dtype=tf.int32)
    ))

def build_electra_model():
    """
    koElectra 모델을 불러오고, 컴파일한 후 반환합니다.
    """
    model = TFElectraForSequenceClassification.from_pretrained(
        "monologg/koelectra-small-v3-discriminator",
        num_labels=5,
        from_pt=True  # PyTorch → TensorFlow 변환
    )
    optimizer = tf.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model

def train_electra_ensemble(train_X, train_y, val_X, val_y, max_len, epochs=5, batch_size=64):
    """
    koElectra 모델 앙상블 학습 함수.
    주어진 학습 및 검증 데이터를 토큰화하고,
    3개의 koElectra 모델을 개별 학습한 후, 각 모델의 예측 결과를 결합하여
    메타 모델 학습을 위한 메타 데이터(meta_train, meta_val)를 생성합니다.
    """
    # 데이터 토큰화
    train_encodings = electra_tokenize_function(train_X, max_len)
    val_encodings = electra_tokenize_function(val_X, max_len)
    
    # TensorFlow 데이터셋 생성
    train_dataset = encode_tf_dataset(train_encodings["input_ids"], train_encodings["attention_mask"], train_y)
    val_dataset = encode_tf_dataset(val_encodings["input_ids"], val_encodings["attention_mask"], val_y)
    
    train_dataset = train_dataset.shuffle(buffer_size=len(train_X)).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    
    # 3개의 개별 koElectra 모델 생성
    model1 = build_electra_model()
    model2 = build_electra_model()
    model3 = build_electra_model()
    
    # 각 모델 학습 (필요시 callbacks 추가 가능)
    model1.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=1)
    model2.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=1)
    model3.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=1)
    
    # 학습 후, 각 모델의 예측 결과(softmax 확률)를 생성
    pred_train1 = tf.nn.softmax(
        model1.predict({"input_ids": train_encodings["input_ids"],
                        "attention_mask": train_encodings["attention_mask"]}).logits
    ).numpy()
    pred_train2 = tf.nn.softmax(
        model2.predict({"input_ids": train_encodings["input_ids"],
                        "attention_mask": train_encodings["attention_mask"]}).logits
    ).numpy()
    pred_train3 = tf.nn.softmax(
        model3.predict({"input_ids": train_encodings["input_ids"],
                        "attention_mask": train_encodings["attention_mask"]}).logits
    ).numpy()
    
    meta_train = np.concatenate([pred_train1, pred_train2, pred_train3], axis=1)
    
    pred_val1 = tf.nn.softmax(
        model1.predict({"input_ids": val_encodings["input_ids"],
                        "attention_mask": val_encodings["attention_mask"]}).logits
    ).numpy()
    pred_val2 = tf.nn.softmax(
        model2.predict({"input_ids": val_encodings["input_ids"],
                        "attention_mask": val_encodings["attention_mask"]}).logits
    ).numpy()
    pred_val3 = tf.nn.softmax(
        model3.predict({"input_ids": val_encodings["input_ids"],
                        "attention_mask": val_encodings["attention_mask"]}).logits
    ).numpy()
    
    meta_val = np.concatenate([pred_val1, pred_val2, pred_val3], axis=1)
    
    return meta_train, meta_val, (model1, model2, model3)

def build_meta_model_koElectra(input_dim=15):
    """
    앙상블된 koElectra 모델의 예측 결과를 입력받아 최종 분류를 수행하는 메타 모델을 생성합니다.
    """
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

# 전역 메타 모델 객체 (필요 시 모듈 외부에서 임포트 가능)
meta_model_koElectra = build_meta_model_koElectra(input_dim=15)
