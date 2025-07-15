# 1. 필요한 라이브러리 가져오기
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 2. MNIST 데이터셋 로드 및 전처리
# MNIST 데이터셋은 60,000개의 학습 이미지와 10,000개의 테스트 이미지로 구성됨
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 이미지 데이터를 0~1 사이의 값으로 정규화
# 픽셀 값은 0~255 사이의 정수이므로 255로 나누어 줌
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. 딥러닝 모델 구성
model = keras.models.Sequential([
    # 입력층: 28x28 크기의 이미지를 784개의 1차원 배열로 펼침
    keras.layers.Flatten(input_shape=(28, 28)),
    
    # 첫 번째 은닉층: 128개의 뉴런, 활성화 함수로 ReLU 사용
    keras.layers.Dense(128, activation='relu'),
    
    # 드롭아웃: 학습 중 과적합을 방지하기 위해 20%의 뉴런을 랜덤하게 끔
    keras.layers.Dropout(0.2),
    
    # 출력층: 0~9까지 10개의 클래스를 분류하므로 10개의 뉴런, 활성화 함수로 Softmax 사용
    keras.layers.Dense(10, activation='softmax')
])

# 4. 모델 컴파일
# 최적화 알고리즘, 손실 함수, 평가 지표 설정
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 학습
# 학습 데이터로 모델을 5번 반복하여 학습(epochs=5)
history = model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# 6. 모델 평가
# 테스트 데이터로 모델의 최종 성능 평가
print("\n# 모델 평가 #")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"테스트 정확도: {test_acc:.4f}")


# 7. 학습 결과 시각화 (Loss & Accuracy)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.title("Training & Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.title("Training & Validation Accuracy")
plt.legend()

plt.show()

# 8. 예측 해보기
# 테스트 데이터의 첫 번째 이미지를 사용하여 예측
predictions = model.predict(x_test)
predicted_label = np.argmax(predictions[0])
true_label = y_test[0]

print(f"\n모델의 예측: {predicted_label}")
print(f"실제 정답: {true_label}")
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Predicted: {predicted_label}, True: {true_label}")
plt.show()