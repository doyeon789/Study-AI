"""
선형 회귀(Linear Regression) 기초 실습

목표:
y = 3x + 5라는 규칙에 노이즈를 섞어 가짜 데이터를 생성한 후,
모델이 그 규칙을 데이터만 보고 얼마나 잘 찾는지 확인한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# 1. 데이터 생성
np.random.seed(42) # 매번 같은 결과가 나오도록 시드값 고정

n_samples = 100
X = np.random.rand(n_samples, 1) * 10 # 0 ~ 10 사이 X값 100개 
noise = np.random.normal(loc=0, scale=2, size=(n_samples, 1)) # 평균 0, 표준편차 2인 정규분포 노이즈
y = (3 * X + 5 + noise).flatten() # y를 1차원으로 변경

print(f"+{'-'*50}+")
print(f"데이터 개수 : {n_samples}개")
print(f"X 샘플 5개: {X[:5].flatten()}")
print(f"y 샘플 5개: {y[:5]}")
print(f"+{'-'*50}+")


# 2. 학습용/테스트용 데이터 나누기

# 학습(train)과 평가(test) 8:2 비율로 사용 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n+{'-'*50}+")
print(f"학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")
print(f"+{'-'*50}+")


# 3. 모델 생성, 학습
model = MyLinearRegression(learning_rate=0.01, epochs=5000)
model.fit(X_train, y_train) # 학습 

print(f"\n+{'-' * 50}+")
print(f"모델이 찾아낸 기울기(slope): {model.coef_[0]:.3f}  (정답: 3)")
print(f"모델이 찾아낸 절편(intercept): {model.intercept_:.3f}  (정답: 5)")
print(f"+{'-' * 50}+")


# 4. 테스트 데이터로 예측, 성능 평가 
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred) # 평균 제곱 오차 (작을수록 good)
r2 = r2_score(y_test, y_pred) # 결정 계수 (1에 가까울수록 good)

print(f"\n+{'-'*50}+")
print(f"[성능 평가]")
print(f"MSE (평균 제곱 오차): {mse:.3f}")
print(f"R² (결정 계수): {r2:.3f}")
print(f"+{'-'*50}+")


# 5. 시각화 
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color="steelblue", alpha=0.6, label="Train data")
plt.scatter(X_test, y_test, color="orange", alpha=0.8, label="Test data")

# 학습된 모델이 그리는 직선
X_line = np.linspace(0, 10, 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color="red", linewidth=2, label="Fitted line")

plt.xlabel("X")
plt.ylabel("y")
plt.title(
    f"Linear Regression "
    f"(y = {model.coef_[0]:.2f}x + {model.intercept_:.2f})"
)
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("linear_regression_result.png", dpi=120, bbox_inches="tight")
plt.show()

print("\n그래프 저장: linear_regression_result.png")
