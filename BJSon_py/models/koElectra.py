# koElectra 모델 호출
from transformers import ElectraTokenizer
from transformers import TFElectraForSequenceClassification
from transformers import ElectraModel # base모델 불러올때 필요

import tensorflow as tf

# koElectra base 모델
# 필요시 바꿔서 학습시키면 아무래도 큰 모델이다보니깐 성능이 올라가지않을까? 예상합니다
# 단, small 모델보다 무겁기때문에 속도나 GPU 등 상황을 고려해봐야할듯
'''
electra_model = ElectraModel.from_pretrained(
    "monologg/koelectra-base-v3-discriminator",
    num_labels=5,
    from_pt=True  # PyTorch → TensorFlow 변환
)
electra_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
'''

# koElectra 토큰나이저
electra_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
# koElectra 모델 정의 (사전학습 모델 불러오기)
# ✅ KoElectra 모델 로드
electra_model = TFElectraForSequenceClassification.from_pretrained(
    "monologg/koelectra-small-v3-discriminator",
    num_labels=5,
    from_pt=True  # PyTorch → TensorFlow 변환
)


# 토큰화 함수 정의
def electra_tokenize_function(texts, max_len):
    return electra_tokenizer(
        texts.tolist(),  # 리스트 형태로 변환
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="tf",
        return_attention_mask=True
    )

def encode_tf_dataset(input_ids, attention_masks, labels):
    return tf.data.Dataset.from_tensor_slices((
        {"input_ids": input_ids, "attention_mask": attention_masks},  # 모델 입력
        tf.convert_to_tensor(labels, dtype=tf.int32)  # 정답 라벨
    ))


def train_electra_ensemble(train_X, train_y, val_X, val_y, max_len):
    """
    koElectra 모델을 활용한 학습 및 메타데이터 생성.
    """
    # koElectra용 Train 데이터 토큰화
    train_encodings = electra_tokenize_function(train_X, max_len)
    train_input_ids = train_encodings["input_ids"]
    train_attention_masks = train_input_ids["attention_mask"]

    # koElectra용 Validation 데이터 토큰화
    val_encodings = electra_tokenize_function(val_X, max_len)
    val_input_ids = val_encodings["input_ids"]
    val_attention_masks = val_input_ids["attention_mask"]

    # Train 데이터셋 변환
    train_dataset = encode_tf_dataset(train_input_ids, train_attention_masks, train_y)

    # Validation 데이터셋 변환
    val_dataset = encode_tf_dataset(val_input_ids, val_attention_masks, val_y)

    # 배치 크기 설정
    BATCH_SIZE = 64
    train_dataset = train_dataset.shuffle(len(train_input_ids)).batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)

    # ✅ 옵티마이저 고정 (최신 옵티마이저 사용 시 에러 방지)
    optimizer = tf.optimizers.Adam(learning_rate=2e-5)

    # ✅ 손실 함수 및 메트릭 설정
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ["accuracy"]

    # ✅ 모델 컴파일
    electra_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # ✅ 학습 시작 (EarlyStopping + 최적의 모델 저장)
    EPOCHS = 100
    patience = 5  # EarlyStopping 조건 (5번 연속 개선되지 않으면 중단)
    wait = 0  # EarlyStopping을 위한 카운터
    best_val_accuracy = 0.0
    best_epoch = 0

    # 모델 학습 루프
    for epoch in range(EPOCHS):
        print(f"\n🔹 Epoch {epoch + 1}/{EPOCHS} 시작...")

        # ✅ 1 epoch 학습 진행
        history = electra_model.fit(train_dataset, validation_data=val_dataset, epochs=1, verbose=1)

        # ✅ 현재 epoch의 검증 정확도 가져오기
        current_val_accuracy = history.history["val_accuracy"][0]

        # ✅ 최적의 모델 여부 확인
        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            best_epoch = epoch + 1
            wait = 0  # EarlyStopping 카운터 초기화
            print(f"✅ 새로운 최고 성능! Epoch {best_epoch}, val_accuracy={best_val_accuracy:.4f}")
        else:
            wait += 1
            print(f"⚠️ {wait}/{patience} - 검증 정확도 개선 없음.")

        # ✅ EarlyStopping 조건 충족 시 학습 중단
        if wait >= patience:
            print(f"\n⏹️ EarlyStopping 발동! {patience}번 연속 개선되지 않음 → 학습 종료")
            break

    # 메타데이터 생성 (softmax 확률값)
    pred_train_koElectra = tf.nn.softmax(
        electra_model.predict([train_encodings['input_ids'], train_encodings['attention_mask']]).logits
    ).numpy()

    pred_val_koElectra = tf.nn.softmax(
        electra_model.predict([val_encodings['input_ids'], val_encodings['attention_mask']]).logits
    ).numpy()

    return pred_train_koElectra, pred_val_koElectra, electra_model

def build_meta_model_koElectra(input_dim=15):
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