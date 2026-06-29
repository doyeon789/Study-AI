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
        # 한 번 학습할 때 기울기와 절편을 얼마나 수행할지 결정
        self.learning_rate = learning_rate
        
        # 전체 데이터를 몇 번이나 반복 학습할지 결정
        self.epochs = epochs
        
        # 학습을 통해 찾을 기울기, 절편
        self.coef = None
        self.intercept_ = 0.0
        
        # 학습 과정의 MSE를 저장할 리스트
        self.loss_history = []