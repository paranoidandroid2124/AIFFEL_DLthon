#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Keras 전처리 및 모델 관련 모듈
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# gensim 및 konlpy
from gensim.models import Word2Vec
from konlpy.tag import Okt

# 콜백 설정 (callbacks.py)
from callbacks import get_callbacks

# 모델 관련 함수들 (각 Notebook을 py 모듈로 변환하여 models 디렉토리에 배치)
from models.textCNN import train_textcnn_ensemble, build_meta_model_textCNN
from models.BiLSTM import train_bilstm_ensemble, build_meta_model_BiLSTM
from models.meta import build_meta_model_final
from models.koElectra import custom_unpack_x_y_sample_weight, train_electra_ensemble, build_meta_model_koElectra, electra_tokenize_function
#-----
# 0. test
#-----
print('is everything ok?')
# ------------------------------------------------------------------------------
# 1. 데이터 준비
# ------------------------------------------------------------------------------

train_data = pd.read_csv("C:/Users/양자/Desktop/Hun_Works/AIFFEL_DLthon/DLThon01/CHYeom/data/sns_syn_sent_augment_small.csv")
# train_data = pd.read_csv("AIFFEL_DLthon\\BJSon_py\\data\\sns_syn_sent_augment.csv")

# ---------------------
# textCNN용 데이터 전처리
# ---------------------
# 형태소 분석 및 토큰화 (textCNN 전용)
okt = Okt()
tokenized_data = [okt.morphs(sentence) for sentence in train_data['conversation']]

# Word2Vec 임베딩 학습 (필요한 경우)
embedding_model = Word2Vec(
    sentences=tokenized_data,
    sg=1,
    vector_size=128,
    window=10,
    min_count=1,
    workers=4
)

max_len = 130

# Tokenizer를 통한 정수 인코딩 (textCNN 전용)
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(tokenized_data)
train_X = tokenizer.texts_to_sequences(tokenized_data)
train_X = pad_sequences(train_X, maxlen=max_len, padding='pre')
train_y = train_data['class'].values

# Train/Validation 분할 (textCNN)
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

# ---------------------
# Electra용 데이터 전처리
# ---------------------
# Electra 모델은 자체 토크나이저를 사용하므로, raw 텍스트 데이터를 그대로 사용합니다.
elec_train_X, elec_val_X, elec_train_y, elec_val_y = train_test_split(train_data["conversation"], train_data["class"], test_size=0.2, random_state=42)

# ------------------------------------------------------------------------------
# 2. 콜백 객체 생성
# ------------------------------------------------------------------------------
callbacks = get_callbacks()

# ------------------------------------------------------------------------------
# 3. 1차 스태킹: 각 앙상블(동질 스태킹)을 통해 메타 데이터를 생성
#    (각 함수 내에서 base 모델들을 학습한 후, 예측값들을 결합하여 메타 데이터를 반환합니다.)
# ------------------------------------------------------------------------------
meta_train_textCNN, meta_val_textCNN, textcnn_models = train_textcnn_ensemble(train_X, train_y, val_X, val_y, callbacks=callbacks)
meta_train_koElectra, meta_val_koElectra, koElectra_models = train_electra_ensemble(elec_train_X, elec_train_y, elec_val_X, elec_val_y, max_len)
meta_train_BiLSTM,  meta_val_BiLSTM, BiLSTM_models  = train_bilstm_ensemble(train_X, train_y, val_X, val_y, callbacks=callbacks)


# ------------------------------------------------------------------------------
# 4. 각 앙상블의 메타모델 학습 (1차 스태킹 결과에 대해)
# ------------------------------------------------------------------------------
# textCNN 메타모델 학습
meta_model_textCNN = build_meta_model_textCNN(input_dim=meta_train_textCNN.shape[1])
meta_model_textCNN.fit(meta_train_textCNN, train_y,validation_data=(meta_val_textCNN, val_y),epochs=15,batch_size=64,callbacks=callbacks)

# # BiLSTM 메타모델 학습
meta_model_BiLSTM = build_meta_model_BiLSTM(input_dim=meta_train_BiLSTM.shape[1])
meta_model_BiLSTM.fit(meta_train_BiLSTM, train_y,validation_data=(meta_val_BiLSTM, val_y),epochs=15,batch_size=64,callbacks=callbacks)

# koElectra 메타모델 학습
meta_model_koElectra = build_meta_model_koElectra(input_dim=meta_train_koElectra.shape[1])
meta_model_koElectra.fit(meta_train_koElectra, elec_train_y, validation_data=(meta_val_koElectra, elec_val_y),epochs=15,batch_size=64)

# ------------------------------------------------------------------------------
# 5. 2차 스태킹: Heterogeneous한 최종 메타모델 학습
# ------------------------------------------------------------------------------
# 각 메타모델의 예측값 생성
pred_meta_train_textCNN = meta_model_textCNN.predict(meta_train_textCNN)
pred_meta_train_BiLSTM   = meta_model_BiLSTM.predict(meta_train_BiLSTM)
pred_meta_train_koElectra = meta_model_koElectra.predict(meta_train_koElectra)

# stacked_X_train = np.concatenate([pred_meta_train_textCNN, pred_meta_train_BiLSTM], axis=1)
# stacked_X_train = np.concatenate([pred_meta_train_textCNN, pred_meta_train_koElectra], axis=1)
stacked_X_train = np.concatenate([pred_meta_train_textCNN, pred_meta_train_BiLSTM, pred_meta_train_koElectra], axis=1)

pred_meta_val_textCNN = meta_model_textCNN.predict(meta_val_textCNN)
pred_meta_val_BiLSTM   = meta_model_BiLSTM.predict(meta_val_BiLSTM)
pred_meta_val_koElectra = meta_model_koElectra.predict(meta_val_koElectra)

# stacked_X_val = np.concatenate([pred_meta_val_textCNN, pred_meta_val_BiLSTM], axis=1)
# stacked_X_val = np.concatenate([pred_meta_val_textCNN, pred_meta_val_koElectra], axis=1)
stacked_X_val = np.concatenate([pred_meta_val_textCNN, pred_meta_val_BiLSTM, pred_meta_val_koElectra], axis=1)

# 최종 메타모델 생성 (입력 차원은 메타모델의 출력 차원 합, 예: 5+5+5=15, 5+5=10)
meta_model_final = build_meta_model_final(input_dim=stacked_X_train.shape[1])
meta_model_final.fit(stacked_X_train, train_y, validation_data=(stacked_X_val, val_y),epochs=10, batch_size=64, callbacks=callbacks)

# ------------------------------------------------------------------------------
# 6. 최종 모델 평가
# ------------------------------------------------------------------------------
loss, acc = meta_model_final.evaluate(stacked_X_val, val_y)
print(f"Final meta-model accuracy: {acc:.4f}")
# ------------------------------------------------------------------------------
# 7. 시각화
#-------------------------------------------------------------------------------
# 1. Validation 데이터에 대한 최종 메타모델 예측 수행
val_pred_probs = meta_model_final.predict(stacked_X_val)
val_pred_labels = np.argmax(val_pred_probs, axis=1)

# 2. Weighted F1-score 계산 및 출력
val_f1 = f1_score(val_y, val_pred_labels, average='weighted')
print(f"Validation Weighted F1-score: {val_f1:.4f}")

# 3. Classification Report 출력
num_classes = 5
class_names = [f"Class {i}" for i in range(num_classes)]
report = classification_report(val_y, val_pred_labels, target_names=class_names)
print("Classification Report:\n", report)

# 4. Confusion Matrix 계산 및 시각화
cm = confusion_matrix(val_y, val_pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=class_names,yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Validation Confusion Matrix")
plt.show()
# ------------------------------------------------------------------------------
# 8. 테스트 데이터 예측
#-------------------------------------------------------------------------------
test_df = pd.read_csv("C:/Users/양자/Desktop/Hun_Works/AIFFEL_DLthon/DLThon01/CHYeom/data/test.csv")
test_texts = test_df['text'].tolist()  # 테스트 데이터의 원본 텍스트

# textCNN 테스트 데이터 전처리 (Okt + Tokenizer 사용)
tokenized_test = [okt.morphs(sentence) for sentence in test_texts]
test_X = tokenizer.texts_to_sequences(tokenized_test)
test_X = pad_sequences(test_X, maxlen=max_len, padding='pre')

pred_test1_cnn = textcnn_models[0].predict(test_X)
pred_test2_cnn = textcnn_models[1].predict(test_X)
pred_test3_cnn = textcnn_models[2].predict(test_X)
meta_test_textCNN = np.concatenate([pred_test1_cnn, pred_test2_cnn, pred_test3_cnn], axis=1)

pred_test1_bilstm1  = BiLSTM_models[0].predict(test_X)
pred_test2_bilstm2 = BiLSTM_models[1].predict(test_X)
pred_test3_bilstm3 = BiLSTM_models[2].predict(test_X)
meta_test_BiLSTM = np.concatenate([pred_test1_bilstm1, pred_test2_bilstm2, pred_test3_bilstm3], axis=1)

# Electra 테스트 데이터 전처리 (전용 electra_tokenize_function 사용)
# electra_tokenize_function은 Electra 모델의 토크나이저를 활용합니다.
test_electra_encodings = electra_tokenize_function(np.array(test_texts), max_len)
pred_test_koElectra_list = []
for model in koElectra_models:
    pred = model.predict({
        "input_ids": test_electra_encodings["input_ids"],
        "attention_mask": test_electra_encodings["attention_mask"]
    })
    pred = tf.nn.softmax(pred.logits).numpy()
    pred_test_koElectra_list.append(pred)
meta_test_koElectra = np.concatenate(pred_test_koElectra_list, axis=1)

# 최종 메타모델을 통한 예측 (각각 5차원)
final_test_pred_textCNN = meta_model_textCNN.predict(meta_test_textCNN)
final_test_pred_BiLSTM = meta_model_BiLSTM.predict(meta_test_BiLSTM)
final_test_pred_koElectra = meta_model_koElectra.predict(meta_test_koElectra)
meta_test = np.concatenate([final_test_pred_textCNN, final_test_pred_BiLSTM, final_test_pred_koElectra], axis=1)

test_pred = meta_model_final.predict(meta_test)
test_pred_labels = np.argmax(test_pred, axis=1)

# Submission 파일 생성
idx_column = [f"t_{i:03d}" for i in range(len(test_pred_labels))]
submission_df = pd.DataFrame({"idx": idx_column, "target": test_pred_labels})
submission_df.to_csv("submission_textCNN_BiLSTM_koElectra_stacking.csv", index=False)
print("CSV 저장 완료: submission_textCNN_BiLSTM_koElectra_stacking.csv")

# ----------------
# 9. 시각화
# ----------------
class_counts = pd.Series(test_pred_labels).value_counts().sort_index()
print(class_counts)
plt.figure(figsize=(8, 5))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Distribution of Test Predictions by Class")
plt.xticks(ticks=range(5), labels=[f"Class {i}" for i in range(5)])
plt.show()
