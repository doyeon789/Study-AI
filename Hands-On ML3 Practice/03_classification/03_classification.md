# 3장. 분류

> MNIST 손글씨 숫자 분류를 중심으로 분류 모델의 평가 방법, 다중 분류, 다중 레이블·다중 출력 분류와 연습 문제를 정리한 학습 노트입니다.
>
> 실습 노트북: [03_classification.ipynb](./03_classification.ipynb)

## 1. 학습 목표

이 장의 핵심은 단순히 정확도가 높은 분류기를 만드는 것이 아니라, 문제에 맞는 평가 지표를 선택하고 모델의 오류를 분석하는 것입니다.

- MNIST 데이터의 구조 이해
- 이진 분류기와 다중 분류기 훈련
- 정확도, 오차 행렬, 정밀도, 재현율, F1 점수 해석
- 결정 임곗값에 따른 정밀도·재현율의 변화 이해
- PR 곡선, ROC 곡선, ROC AUC를 이용한 모델 비교
- 다중 레이블 분류와 다중 출력 분류 이해
- KNN 튜닝과 데이터 증식 실습
- 타이타닉 생존자 및 스팸 이메일 분류 파이프라인 구성

전체 실습 흐름은 다음과 같습니다.

```text
데이터 준비
  → 이진 분류
  → 여러 평가 지표로 성능 측정
  → 임곗값 조절과 모델 비교
  → 다중 분류
  → 오차 분석
  → 다중 레이블·다중 출력 분류
  → 연습 문제 적용
```

---

## 2. MNIST 데이터셋

MNIST는 0부터 9까지의 손글씨 숫자 이미지 70,000개로 이루어진 데이터셋입니다.

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", as_frame=False)
X, y = mnist.data, mnist.target
```

| 항목 | 내용 |
|---|---|
| 샘플 수 | 70,000개 |
| 입력 특성 수 | 784개 |
| 이미지 크기 | 28 × 28 픽셀 |
| 픽셀 값 | 0~255 |
| 타깃 | 문자열 `"0"`~`"9"` |

각 이미지는 28 × 28 픽셀을 일렬로 펼친 길이 784의 벡터입니다. 시각화할 때는 다시 2차원 배열로 변환합니다.

```python
def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
```

MNIST는 앞의 60,000개가 훈련 세트, 뒤의 10,000개가 테스트 세트로 미리 구성되어 있습니다.

```python
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
```

테스트 세트는 모델과 하이퍼파라미터를 모두 정한 뒤 최종 성능을 확인할 때만 사용합니다.

---

## 3. 이진 분류기 훈련

먼저 숫자 5와 5가 아닌 이미지만 구별하는 **이진 분류(binary classification)** 문제로 단순화합니다.

```python
y_train_5 = (y_train == "5")
y_test_5 = (y_test == "5")
```

`SGDClassifier`는 확률적 경사 하강법을 사용하는 분류기입니다. 샘플을 하나씩 처리할 수 있어 큰 데이터셋과 온라인 학습에 적합합니다.

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([X_train[0]])
```

첫 번째 이미지는 숫자 5이고 모델도 `True`로 예측합니다. 그러나 한 샘플의 예측만으로 모델의 품질을 판단할 수는 없습니다.

---

## 4. 정확도와 교차 검증

```python
from sklearn.model_selection import cross_val_score

cross_val_score(
    sgd_clf,
    X_train,
    y_train_5,
    cv=3,
    scoring="accuracy"
)
```

SGD 분류기의 3-폴드 교차 검증 정확도는 약 `95.0%`, `96.0%`, `96.0%`입니다. 겉으로는 매우 좋아 보이지만, 비교를 위해 모든 이미지를 무조건 `5 아님`으로 예측하는 더미 분류기를 사용해 봅니다.

```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
cross_val_score(
    dummy_clf,
    X_train,
    y_train_5,
    cv=3,
    scoring="accuracy"
)
```

더미 분류기도 약 `90.97%`의 정확도를 얻습니다. 전체 데이터 중 숫자 5가 약 10%뿐이므로 언제나 다수 클래스만 예측해도 높은 정확도가 나오기 때문입니다.

> 클래스 비율이 불균형한 데이터에서는 정확도만으로 모델을 평가하면 성능을 잘못 판단할 수 있습니다.

---

## 5. 오차 행렬

테스트 세트를 사용하지 않고 각 훈련 샘플에 대한 검증 예측을 만들기 위해 `cross_val_predict()`를 사용합니다.

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(
    sgd_clf,
    X_train,
    y_train_5,
    cv=3
)
cm = confusion_matrix(y_train_5, y_train_pred)
```

노트북의 결과는 다음과 같습니다.

```text
[[53892,   687],
 [ 1891,  3530]]
```

오차 행렬은 행이 실제 클래스, 열이 예측 클래스입니다.

| 구분 | 의미 | 결과 |
|---|---|---:|
| TN | 실제 5 아님, 예측도 5 아님 | 53,892 |
| FP | 실제 5 아님, 예측은 5 | 687 |
| FN | 실제 5, 예측은 5 아님 | 1,891 |
| TP | 실제 5, 예측도 5 | 3,530 |

- FP(False Positive): 거짓 양성, 1종 오류
- FN(False Negative): 거짓 음성, 2종 오류
- 완벽한 분류기는 주대각선인 TN과 TP에만 값이 존재합니다.

---

## 6. 정밀도, 재현율, F1 점수

### 6.1 정밀도

정밀도(precision)는 모델이 양성이라고 예측한 샘플 중 실제 양성의 비율입니다.

$$
Precision = \frac{TP}{TP + FP}
$$

```python
from sklearn.metrics import precision_score

precision_score(y_train_5, y_train_pred)
```

결과는 약 `0.837`입니다. 모델이 5라고 판단한 이미지 중 약 83.7%가 실제 5입니다.

### 6.2 재현율

재현율(recall)은 실제 양성 샘플 중 모델이 찾아낸 양성의 비율입니다. 민감도(sensitivity) 또는 진짜 양성 비율(TPR)이라고도 합니다.

$$
Recall = \frac{TP}{TP + FN}
$$

결과는 약 `0.651`입니다. 실제 숫자 5 중 약 65.1%만 감지했습니다.

### 6.3 F1 점수

F1 점수는 정밀도와 재현율의 조화 평균입니다.

$$
F_1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

```python
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)
```

노트북의 F1 점수는 약 `0.733`입니다. 조화 평균은 낮은 값에 더 큰 영향을 받으므로 정밀도와 재현율이 모두 높아야 좋은 점수를 얻습니다.

| 지표 | 중요하게 보는 것 | 적합한 사례 |
|---|---|---|
| 정밀도 | 양성 예측을 얼마나 믿을 수 있는가 | 정상 메일을 스팸으로 차단하는 일을 줄여야 할 때 |
| 재현율 | 실제 양성을 얼마나 빠짐없이 찾는가 | 질병이나 위험 사례를 놓치지 않아야 할 때 |
| F1 | 정밀도와 재현율의 균형 | 두 종류의 오류를 함께 고려할 때 |

---

## 7. 정밀도·재현율 트레이드오프

`SGDClassifier`는 각 샘플의 결정 점수를 계산하고, 이 점수가 임곗값보다 크면 양성으로 예측합니다.

```python
y_scores = sgd_clf.decision_function([X_train[0]])
y_pred = (y_scores > threshold)
```

- 임곗값을 높이면 양성 판정이 까다로워져 일반적으로 정밀도는 올라가고 재현율은 내려갑니다.
- 임곗값을 낮추면 더 많은 양성을 찾으므로 재현율은 올라가지만 거짓 양성이 늘 수 있습니다.

모든 훈련 샘플의 결정 점수는 다음과 같이 구합니다.

```python
y_scores = cross_val_predict(
    sgd_clf,
    X_train,
    y_train_5,
    cv=3,
    method="decision_function"
)
```

`precision_recall_curve()`를 사용하면 가능한 여러 임곗값에서 정밀도와 재현율을 계산할 수 있습니다.

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(
    y_train_5,
    y_scores
)
```

노트북에서는 정밀도 90%를 달성하는 임곗값이 약 `3370.02`입니다.

```python
idx = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx]
y_train_pred_90 = (y_scores >= threshold_for_90_precision)
```

- 정밀도: 약 `90.0%`
- 재현율: 약 `48.0%`

정밀도 목표만 달성했다고 좋은 모델은 아닙니다. 이 경우 실제 5의 절반 이상을 놓치므로 프로젝트가 요구하는 최소 재현율도 함께 확인해야 합니다.

---

## 8. ROC 곡선과 ROC AUC

ROC(Receiver Operating Characteristic) 곡선은 여러 임곗값에서 거짓 양성 비율(FPR)에 대한 진짜 양성 비율(TPR)을 나타냅니다.

$$
TPR = Recall = \frac{TP}{TP + FN}
$$

$$
FPR = \frac{FP}{FP + TN} = 1 - TNR
$$

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
roc_auc_score(y_train_5, y_scores)
```

- 완벽한 분류기의 ROC AUC: `1`
- 무작위 분류기의 ROC AUC: `0.5`
- SGD 분류기의 ROC AUC: 약 `0.9605`

ROC 곡선에서도 재현율을 높이면 일반적으로 거짓 양성 비율이 증가하는 트레이드오프가 나타납니다.

### PR 곡선과 ROC 곡선 선택

- 양성 클래스가 드물거나 FP보다 FN에 더 관심이 있다면 PR 곡선이 유용합니다.
- 두 클래스가 균형에 가깝고 전체적인 순위 판별력을 비교하려면 ROC 곡선과 ROC AUC가 유용합니다.
- 불균형 데이터에서는 많은 TN 때문에 ROC 성능이 지나치게 좋아 보일 수 있으므로 PR 곡선을 함께 확인하는 것이 안전합니다.

---

## 9. SGD와 랜덤 포레스트 비교

`RandomForestClassifier`는 `decision_function()` 대신 각 클래스의 추정 확률을 반환하는 `predict_proba()`를 제공합니다.

```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(
    forest_clf,
    X_train,
    y_train_5,
    cv=3,
    method="predict_proba"
)
y_scores_forest = y_probas_forest[:, 1]
```

| 평가 지표 | SGD | 랜덤 포레스트 |
|---|---:|---:|
| ROC AUC | 약 0.9605 | 약 0.9983 |
| F1 | 약 0.7325 | 약 0.9275 |
| 정밀도 | 약 0.8371 | 약 0.9897 |
| 재현율 | 약 0.6512 | 약 0.8725 |

랜덤 포레스트의 PR 곡선은 SGD보다 오른쪽 위에 가깝고 다른 지표도 더 우수했습니다.

`predict_proba()`가 반환하는 값은 모델의 **추정 확률**입니다. 이 값이 실제 확률과 잘 일치하는지는 별도의 확률 보정(calibration) 관점에서 확인해야 합니다.

---

## 10. 다중 분류

다중 분류(multiclass classification)는 둘보다 많은 클래스를 구별하는 문제입니다. MNIST에서는 0부터 9까지 총 10개의 클래스가 있습니다.

### 10.1 OvR과 OvO

| 전략 | 학습 방법 | 분류기 수 |
|---|---|---:|
| OvR(One-versus-the-Rest) | 각 클래스를 나머지 전체와 구별 | 클래스가 N개이면 N개 |
| OvO(One-versus-One) | 클래스의 모든 두 쌍을 구별 | N × (N - 1) / 2개 |

MNIST에서는 OvR은 10개, OvO는 45개의 이진 분류기를 학습합니다.

SVM은 큰 데이터셋에서 느릴 수 있어 노트북에서는 처음 2,000개 샘플로 실습합니다.

```python
from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])
svm_clf.predict([X_train[0]])
```

`SVC`는 내부적으로 OvO 전략을 사용합니다. `decision_function()`의 기본 출력은 클래스별 점수 10개이며, 가장 높은 점수의 클래스를 예측합니다.

```python
some_digit_scores = svm_clf.decision_function([X_train[0]])
class_id = some_digit_scores.argmax()
svm_clf.classes_[class_id]
```

`OneVsRestClassifier`나 `OneVsOneClassifier`를 사용하면 원하는 전략을 명시적으로 강제할 수 있습니다.

### 10.2 스케일링의 효과

SGD 다중 분류기의 3-폴드 정확도는 약 `85.8%~87.4%`였습니다. 입력 특성을 표준화한 뒤에는 약 `89.1%~90.2%`로 개선되었습니다.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
```

픽셀 값처럼 특성 범위가 큰 데이터에서 경사 하강법 기반 모델은 스케일링의 영향을 크게 받습니다. 테스트 세트에는 훈련 세트에서 학습한 스케일러로 `transform()`만 적용해야 합니다.

---

## 11. 에러 분석

모델을 개선하려면 전체 점수뿐 아니라 어떤 클래스를 서로 혼동하는지 조사해야 합니다.

```python
from sklearn.metrics import ConfusionMatrixDisplay

y_train_pred = cross_val_predict(
    sgd_clf,
    X_train_scaled,
    y_train,
    cv=3
)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
```

샘플 수가 많은 클래스가 그림을 지배하지 않도록 행별 정규화를 적용할 수 있습니다.

```python
ConfusionMatrixDisplay.from_predictions(
    y_train,
    y_train_pred,
    normalize="true",
    values_format=".0%"
)
```

올바른 예측을 제외하고 오류만 강조할 수도 있습니다.

```python
sample_weight = (y_train_pred != y_train)
ConfusionMatrixDisplay.from_predictions(
    y_train,
    y_train_pred,
    sample_weight=sample_weight,
    normalize="true",
    values_format=".0%"
)
```

노트북에서는 많은 숫자가 8로 잘못 분류되는 경향이 눈에 띕니다. 이런 분석을 바탕으로 다음 개선을 시도할 수 있습니다.

- 혼동되는 숫자의 샘플을 직접 살펴보기
- 이미지 이동·회전·두께 보정 등의 전처리 또는 데이터 증식 적용
- 혼동되는 클래스에 유리한 새 특성 만들기
- 다른 모델이나 하이퍼파라미터 실험

---

## 12. 다중 레이블 분류

다중 레이블 분류(multilabel classification)는 하나의 샘플에 여러 개의 이진 레이블을 동시에 부여합니다.

노트북에서는 숫자가 `7 이상인지`와 `홀수인지`를 동시에 예측합니다.

```python
y_train_large = (y_train >= "7")
y_train_odd = (y_train.astype("int8") % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([X_train[0]])
```

첫 번째 이미지인 숫자 5는 7 이상은 아니고 홀수이므로 `[False, True]`가 됩니다.

다중 레이블 모델은 각 레이블의 F1 점수를 평균해 평가할 수 있습니다.

```python
f1_score(y_multilabel, y_train_knn_pred, average="macro")
```

- `macro`: 모든 레이블의 점수를 동일한 비중으로 평균
- `weighted`: 각 레이블의 지지도에 비례해 가중 평균

레이블들이 서로 연관되어 있다면 `ClassifierChain`을 사용할 수 있습니다. 앞선 분류기의 예측을 다음 분류기의 입력 특성으로 전달해 레이블 간 의존성을 학습합니다.

---

## 13. 다중 출력 분류

다중 출력 다중 클래스 분류(multioutput-multiclass classification)는 여러 레이블 각각이 둘 이상의 값을 가질 수 있는 문제입니다.

노트북에서는 MNIST 이미지에 무작위 잡음을 더하고, 원래의 깨끗한 이미지를 타깃으로 사용합니다.

```python
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
y_train_mod = X_train

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[0]])
```

출력은 784개 픽셀 각각의 강도를 예측합니다. 즉, 출력 레이블이 784개이고 각 레이블이 0~255 중 하나의 값을 가질 수 있습니다.

---

## 14. 연습 문제 1: 97% 정확도의 MNIST 분류기

기본 `KNeighborsClassifier`의 테스트 정확도는 약 `96.88%`였습니다. `n_neighbors`와 `weights`를 그리드 탐색합니다.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param_grid = [{
    "n_neighbors": [3, 4, 5, 6],
    "weights": ["uniform", "distance"]
}]

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5
)
grid_search.fit(X_train[:10000], y_train[:10000])
```

최적 하이퍼파라미터는 다음과 같습니다.

```python
{"n_neighbors": 4, "weights": "distance"}
```

최적 모델을 전체 훈련 세트로 다시 학습한 테스트 정확도는 약 `97.14%`입니다.

> 탐색을 일부 데이터로 빠르게 수행할 수 있지만, 최종 모델은 전체 훈련 세트로 다시 학습해야 합니다.

---

## 15. 연습 문제 2: 데이터 증식

이미지를 위·아래·왼쪽·오른쪽으로 한 픽셀씩 이동한 샘플을 훈련 세트에 추가합니다.

```python
from scipy.ndimage import shift

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted.reshape([-1])
```

원본 이미지 60,000개에 이동한 이미지 네 종류를 더하면 훈련 세트가 300,000개로 증가합니다. 증식한 데이터는 같은 변형끼리 몰리지 않도록 섞은 뒤 학습합니다.

- 증식 전 정확도: 약 `97.14%`
- 증식 후 정확도: 약 `97.63%`
- 오류율: 약 `17%` 감소

정확도의 절대 증가는 약 0.5%p이지만 이미 정확도가 높은 상황에서는 남은 오류 중 상당 부분을 줄인 결과입니다.

---

## 16. 연습 문제 3: 타이타닉 생존자 분류

목표는 승객의 객실 등급, 성별, 나이, 가족 수, 요금, 탑승 항구 등을 이용해 생존 여부를 예측하는 것입니다.

### 16.1 데이터 확인

- 훈련 샘플: 891개
- 생존자: 342명
- 사망자: 549명
- 결측값이 있는 주요 열: `Age`, `Cabin`, `Embarked`
- `Cabin`은 결측값이 너무 많아 제외
- `Name`, `Ticket`은 그대로 사용하기 어려워 기본 실습에서는 제외

### 16.2 전처리 파이프라인

```python
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("cat_encoder", OneHotEncoder(sparse_output=False))
])
```

- 수치형 특성: 중앙값으로 결측값 대체 후 표준화
- 범주형 특성: 최빈값으로 결측값 대체 후 원-핫 인코딩
- `ColumnTransformer`로 두 파이프라인 결합

### 16.3 모델 비교

10-폴드 교차 검증 평균 정확도는 다음과 같습니다.

| 모델 | 기본 특성 | 특성 공학 후 |
|---|---:|---:|
| 랜덤 포레스트 | 약 0.8138 | 약 0.8014 |
| SVM | 약 0.8249 | 약 0.8305 |

특성 공학에서는 다음 특성을 추가합니다.

```python
train_data["AgeBucket"] = train_data["Age"] // 15 * 15
train_data["RelativesOnboard"] = (
    train_data["SibSp"] + train_data["Parch"]
)
```

- `AgeBucket`: 연령을 15년 단위 구간으로 변환
- `RelativesOnboard`: 함께 탑승한 형제자매·배우자·부모·자녀 수

특성을 추가했다고 모든 모델의 점수가 반드시 좋아지는 것은 아닙니다. 교차 검증으로 실제 개선 여부를 확인해야 하며, 이 실습에서는 SVM만 소폭 개선되었습니다.

---

## 17. 연습 문제 4: 스팸 필터

SpamAssassin 공개 이메일 데이터에서 정상 메일(ham)과 스팸 메일을 구별합니다.

### 17.1 이메일 구조 분석

이메일은 단순 텍스트, HTML, 첨부 파일을 포함한 멀티파트 등 여러 구조를 가질 수 있습니다.

- 정상 메일은 `text/plain` 비율이 높습니다.
- 스팸 메일은 HTML 형식의 비율이 상대적으로 높습니다.
- 이메일 구조 자체도 유용한 특성이 될 수 있습니다.

### 17.2 텍스트 전처리

`EmailToWordCounterTransformer`는 다음 과정을 수행합니다.

1. 이메일에서 일반 텍스트 또는 HTML 본문 추출
2. HTML 태그 제거 및 링크를 `HYPERLINK`로 치환
3. 소문자 변환
4. URL을 `URL` 토큰으로 치환
5. 숫자를 `NUMBER` 토큰으로 치환
6. 구두점 제거
7. 단어별 출현 횟수 계산
8. Porter Stemmer로 어간 추출

`WordCounterToVectorTransformer`는 훈련 데이터에서 자주 등장한 단어로 어휘 목록을 만들고 이메일별 단어 횟수를 희소 행렬로 변환합니다.

```python
preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector", WordCounterToVectorTransformer())
])
```

`fit()`에서는 훈련 데이터로 어휘를 만들고, `transform()`에서는 이미 만들어진 어휘를 이용해야 합니다. 테스트 데이터까지 포함해 어휘를 만들면 데이터 누수가 발생합니다.

### 17.3 로지스틱 회귀 결과

```python
log_clf = LogisticRegression(max_iter=1000, random_state=42)
score = cross_val_score(
    log_clf,
    X_train_transformed,
    y_train,
    cv=3
)
```

- 3-폴드 교차 검증 평균 정확도: 약 `98.5%`
- 테스트 정밀도: 약 `96.88%`
- 테스트 재현율: 약 `97.89%`

스팸 필터에서는 정상 메일을 스팸으로 분류하는 FP와 스팸을 놓치는 FN의 비용이 다르므로 정확도와 함께 정밀도·재현율을 반드시 확인해야 합니다.

> 실습의 HTML 제거 함수는 학습 목적의 단순한 정규식 구현입니다. 실제 서비스에서는 중첩 태그, 잘못된 HTML, 인코딩 문제를 처리할 수 있는 HTML 파서를 사용하는 편이 안전합니다.

---

## 18. 주요 분류 문제 비교

| 문제 종류 | 샘플당 출력 | 예시 |
|---|---|---|
| 이진 분류 | 두 클래스 중 하나 | 5 / 5 아님, 스팸 / 정상 |
| 다중 분류 | 여러 클래스 중 하나 | 숫자 0~9 중 하나 |
| 다중 레이블 분류 | 여러 이진 레이블 | 7 이상 여부와 홀수 여부 |
| 다중 출력 분류 | 여러 다중 클래스 레이블 | 이미지의 각 픽셀 강도 예측 |

---

## 19. 핵심 클래스와 함수 정리

| 도구 | 역할 |
|---|---|
| `SGDClassifier` | 대규모 데이터에 효율적인 선형 분류기 |
| `DummyClassifier` | 단순 기준 모델 생성 |
| `cross_val_score` | 교차 검증 점수 계산 |
| `cross_val_predict` | 각 샘플의 교차 검증 예측 생성 |
| `confusion_matrix` | 실제 클래스와 예측 클래스의 조합 집계 |
| `precision_score` | 양성 예측의 신뢰도 측정 |
| `recall_score` | 실제 양성의 탐지 비율 측정 |
| `f1_score` | 정밀도와 재현율의 조화 평균 |
| `precision_recall_curve` | 임곗값별 정밀도와 재현율 계산 |
| `roc_curve` | 임곗값별 FPR과 TPR 계산 |
| `roc_auc_score` | ROC 곡선 아래 면적 계산 |
| `SVC` | 서포트 벡터 머신 분류기 |
| `RandomForestClassifier` | 여러 결정 트리를 결합한 분류기 |
| `KNeighborsClassifier` | 가까운 훈련 샘플을 이용한 분류기 |
| `OneVsRestClassifier` | OvR 다중 분류 전략 강제 |
| `OneVsOneClassifier` | OvO 다중 분류 전략 강제 |
| `ClassifierChain` | 레이블 간 의존성을 이용한 다중 레이블 분류 |
| `ConfusionMatrixDisplay` | 오차 행렬 시각화 |

---

## 20. 실습에서 꼭 기억할 규칙

1. 테스트 세트는 모델 선택과 튜닝이 끝난 뒤 한 번만 사용합니다.
2. 불균형 데이터에서는 정확도만 믿지 않고 정밀도, 재현율, F1 점수를 함께 봅니다.
3. 오차 행렬의 행은 실제 클래스, 열은 예측 클래스입니다.
4. 정밀도와 재현율 사이에는 일반적으로 트레이드오프가 있습니다.
5. 임곗값은 모델이 자동으로 정한 기본값만 쓰지 말고 서비스 목표에 맞춰 선택할 수 있습니다.
6. 양성 클래스가 드물 때는 ROC 곡선뿐 아니라 PR 곡선을 확인합니다.
7. 다중 분류 전략에는 OvR과 OvO가 있으며 알고리즘에 따라 기본 전략이 다릅니다.
8. 특성 스케일링은 SGD와 SVM 같은 모델의 성능에 큰 영향을 줄 수 있습니다.
9. 전체 점수뿐 아니라 클래스별 오류를 분석해야 구체적인 개선 방향을 찾을 수 있습니다.
10. 특성 공학이나 데이터 증식의 효과는 반드시 교차 검증 또는 분리된 검증 데이터로 확인합니다.
11. 텍스트의 어휘 목록과 모든 전처리 기준은 훈련 데이터에서만 학습합니다.
12. 분류 지표는 FP와 FN 중 어떤 오류가 더 위험한지에 따라 선택합니다.

---

## 21. 복습 질문

1. MNIST 이미지 하나가 784개의 특성을 가지는 이유는 무엇인가?
2. 숫자 5 분류에서 무조건 `5 아님`으로 예측해도 정확도가 90%를 넘는 이유는 무엇인가?
3. 오차 행렬에서 TN, FP, FN, TP는 각각 무엇을 의미하는가?
4. 정확도보다 정밀도가 중요한 사례와 재현율이 중요한 사례는 각각 무엇인가?
5. F1 점수에서 산술 평균 대신 조화 평균을 사용하는 효과는 무엇인가?
6. 결정 임곗값을 높이면 정밀도와 재현율은 일반적으로 어떻게 변하는가?
7. `cross_val_predict()`로 만든 예측을 평가에 사용하는 이유는 무엇인가?
8. PR 곡선과 ROC 곡선은 각각 어떤 상황에서 더 유용한가?
9. ROC AUC가 0.5와 1이라는 것은 각각 무엇을 뜻하는가?
10. `decision_function()`과 `predict_proba()`는 어떤 값을 반환하는가?
11. OvR과 OvO는 필요한 분류기 수와 학습 방식이 어떻게 다른가?
12. 표준화가 SGD 다중 분류기의 정확도를 높인 이유는 무엇인가?
13. 정규화된 오차 행렬과 오류만 표시한 오차 행렬은 각각 무엇을 보여 주는가?
14. 다중 레이블 분류와 다중 출력 분류의 차이는 무엇인가?
15. MNIST 데이터 증식으로 정확도는 조금만 올랐지만 오류율은 크게 줄어든 이유는 무엇인가?
16. 타이타닉 데이터에 새 특성을 추가했는데 모델에 따라 성능이 달라진 이유는 무엇인가?
17. 스팸 필터의 어휘 목록을 테스트 데이터까지 포함해 만들면 왜 데이터 누수인가?

---

## 22. 한 문장 요약

> 좋은 분류 시스템은 정확도 하나를 높이는 데서 끝나지 않고, 문제의 클래스 구조와 오류 비용에 맞는 지표·임곗값을 선택하고 오차를 분석해 일반화 성능을 개선하는 시스템입니다.
