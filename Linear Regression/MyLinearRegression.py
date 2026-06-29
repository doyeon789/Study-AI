import numpy as np

"""
선형 회귀(Linear Regression) 직접 구현
목표 : 경사하강법을 이용한 선형 회귀 모델 구현
"""

class MyLinearRegression:
    """
    경사 하강법을 이용한 선형 회귀 모델
    
    모델 기본 형태 : y_pred = w * x + b 
    
    w : 기울기(weight)
    b : 절편(bias)
    """
    
    def __init__(self, learning_rate=0.01, epochs=5000):
        """_summary_

        Args:
            learning_rate (float, optional): _description_. Defaults to 0.01.
            epochs (int, optional): _description_. Defaults to 5000.
        """
        # 한 번 학습할 때 기울기와 절편을 얼마나 수행할지 결정
        self.learning_rate = learning_rate
        
        # 전체 데이터를 몇 번이나 반복 학습할지 결정
        self.epochs = epochs
        
        # 학습을 통해 찾을 기울기, 절편
        self.coef_ = None
        self.intercept_ = 0.0
        
        # 학습 과정의 MSE를 저장할 리스트
        self.loss_history = []
        
    def fit(self, X, y):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        # y를 (100, 1) 형태에서 (100,) 형태로 변환
        y = y.flatten()
        
        n_samples, n_features = X.shape
        
        # 특성 개수만큼 기울기 생성
        self.coef_ = np.zeros(n_features)
        
        # 학습 전 절편 0
        self.intercept_ = 0.0
        
        for epoch in range(self.epochs):
            
            # 1. 현재 기울기와 절편으로 y값 예측
            y_pred = X @ self.coef_ + self.intercept_
            
            # 2. 예측값과 실제값의 차이 게산
            error = y_pred - y
            
            # 3. MSE를 기울기와 절편에 대해 미분
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)
            
            # 4. 경사 하강법으로 기울기와 절편 수정
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db
            
            # 5. 현재 MSE 기록
            mse = np.mean(error ** 2)
            self.loss_history.append(mse)
            
            # 1000번마다 학습 상태 출력
            if epoch % 1000 == 0:
                print(
                    f"Epoch {epoch:4d} | "
                    f"MSE: {mse:.4f} | "
                    f"기울기: {self.coef_[0]:.4f} | "
                    f"절편: {self.intercept_:.4f}"
                )
                
        return self
    
    def predict(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        return X @ self.coef_ + self.intercept_