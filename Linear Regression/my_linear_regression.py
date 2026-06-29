import numpy as np

"""
선형 회귀(Linear Regression) 직접 구현

목표: 경사하강법(Gradient Descent)을 이용하여 선형 회귀 모델의 학습 과정을 직접 구현한다.
"""


class MyLinearRegression:
    """
    경사하강법을 이용한 선형 회귀 모델

    모델의 기본 형태:
        y_pred = X @ w + b
 
    Attributes:
        learning_rate (float):
            한 번의 학습에서 가중치와 절편을 수정하는 크기

        epochs (int):
            전체 학습 데이터를 반복해서 학습하는 횟수

        coef_ (numpy.ndarray):
            학습을 통해 찾아낸 가중치(기울기)

        intercept_ (float):
            학습을 통해 찾아낸 절편

        loss_history (list):
            각 학습 단계에서 계산된 MSE 기록
    """

    def __init__(self, learning_rate=0.01, epochs=5000):
        """
        선형 회귀 모델의 학습 설정을 초기화한다.

        Args:
            learning_rate (float, optional):
                경사하강법에서 가중치와 절편을 한 번에 수정하는 크기.
                값이 너무 크면 학습이 불안정해질 수 있고,
                너무 작으면 학습 속도가 느려질 수 있다.
                기본값은 0.01이다.

            epochs (int, optional):
                전체 학습 데이터를 반복해서 학습하는 횟수.
                기본값은 5000이다.
        """

        # 한 번 학습할 때 가중치와 절편을 얼마나 수정할지 결정
        self.learning_rate = learning_rate

        # 전체 데이터를 몇 번 반복해서 학습할지 결정
        self.epochs = epochs

        # 학습을 통해 찾을 가중치와 절편
        # fit()을 실행하기 전에는 아직 가중치가 결정되지 않았으므로 None
        self.coef_ = None
        self.intercept_ = 0.0

        # 각 epoch에서 계산된 MSE를 저장
        self.loss_history = []

    def fit(self, X, y):
        """
        학습 데이터를 이용해 가중치와 절편을 학습한다.

        경사하강법을 반복하면서 MSE가 작아지는 방향으로
        가중치와 절편을 수정한다.

        Args:
            X (numpy.ndarray):
                학습에 사용할 입력 데이터.
                형태는 (데이터 개수, 특성 개수)이다.

                예:
                    데이터 100개, 특성 1개 -> (100, 1)

            y (numpy.ndarray):
                각 입력 데이터에 대응하는 실제 정답값.
                (데이터 개수,) 또는 (데이터 개수, 1) 형태를 사용할 수 있다.

        Returns:
            MyLinearRegression:
                학습이 완료된 현재 모델 객체를 반환한다.
                따라서 model.fit(X, y).predict(X)와 같은 사용이 가능하다.
        """

        # y를 1차원 배열로 변환
        # 예: (100, 1) -> (100,)
        y = np.asarray(y).flatten()

        # n_samples: 데이터 개수
        # n_features: 입력 특성 개수
        n_samples, n_features = X.shape

        # 이전 학습 기록이 남지 않도록 초기화
        self.loss_history = []

        # 특성 개수만큼 가중치를 생성하고 모두 0으로 초기화
        # 특성이 1개라면 가중치도 1개 생성된다.
        self.coef_ = np.zeros(n_features)

        # 학습 전 절편을 0으로 초기화
        self.intercept_ = 0.0

        for epoch in range(self.epochs):

            # 1. 현재 가중치와 절편으로 y값 예측
            #
            # X @ self.coef_는 행렬 곱셈이다.
            # 특성이 1개인 경우에는 w * x와 같은 역할을 한다.
            y_pred = X @ self.coef_ + self.intercept_

            # 2. 예측값과 실제값의 차이 계산
            # 양수이면 실제값보다 크게 예측한 것이고,
            # 음수이면 실제값보다 작게 예측한 것이다.
            error = y_pred - y

            # 3. MSE를 가중치와 절편에 대해 미분
            #
            # dw: 가중치를 어느 방향으로 얼마나 수정할지 나타내는 기울기
            # db: 절편을 어느 방향으로 얼마나 수정할지 나타내는 기울기
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            # 4. 경사하강법으로 가중치와 절편 수정
            #
            # 손실이 작아지는 방향으로 이동하기 위해
            # 현재 값에서 learning_rate * gradient를 뺀다.
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db

            # 5. 현재 예측 오차의 평균 제곱값(MSE) 계산
            # MSE는 0에 가까울수록 예측이 실제값과 가깝다는 의미이다.
            mse = np.mean(error**2)
            self.loss_history.append(mse)

            # 1000 epoch마다 현재 학습 상태 출력
            if epoch % 1000 == 0:
                print(
                    f"Epoch {epoch:4d} | "
                    f"MSE: {mse:.4f} | "
                    f"기울기: {self.coef_[0]:.4f} | "
                    f"절편: {self.intercept_:.4f}"
                )

        # sklearn의 fit()과 비슷하게 학습된 모델 자신을 반환
        return self

    def predict(self, X):
        """
        학습된 가중치와 절편을 사용해 새로운 입력값을 예측한다.

        Args:
            X (numpy.ndarray):
                예측에 사용할 입력 데이터.
                형태는 (데이터 개수, 특성 개수)이다.

        Returns:
            numpy.ndarray:
                각 입력 데이터에 대한 예측값.
                형태는 (데이터 개수,)이다.

        Raises:
            ValueError:
                fit()을 실행하기 전에 predict()를 호출한 경우 발생한다.
        """

        # 아직 학습하지 않은 모델로 예측하는 것을 방지
        if self.coef_ is None:
            raise ValueError("predict()를 호출하기 전에 fit()을 먼저 실행하세요.")

        # 학습된 선형 회귀식으로 예측
        return X @ self.coef_ + self.intercept_