"""
선형 회귀(Linear Regression) 시초 실습
목표 : y = 3x + 5라는 규칙에 노이즈를 섰어서 가짜 데이터를 생성후, 모델이 그 규칙을 데이터만 보고 얼마나 잘 찾는지 확인
"""

import numpy as np
import matplotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# 1. 데이터 생성
np.random.seed(42) # 매번 같은 결과가 나오도록 시드값 고정

n_samples = 100
X = np.random.rand(n_samples, 1) # 0 ~ 10 사이 X값 100개 
noise = np.random.randn(n_samples, 1) * 2 #  평균 : 0, 표준편차가 2인 정규분포 노이즈 -> 데이터가 직선에서 ±2 정도 흩어짐
y = 3*X + 5 + noise

print(f"+{'-'*50}+")
print(f"데이터 개수 : {n_samples}개")
print(f"X 샘플 5개: {X[:5].flatten()}")
print(f"y 샘플 5개: {y[:5].flatten()}")
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
model = LinearRegression()
model.fit(X_train, y_train) # 학습 

print(f"\n+{'-'*50}+")
print(f"모델이 찾아낸 기울기(slope): {model.coef_[0][0]:.3f}  (정답: 3)")
print(f"모델이 찾아낸 절편(intercept): {model.intercept_[0]:.3f}  (정답: 5)")
print(f"+{'-'*50}+")
