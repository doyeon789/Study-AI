# 2장. 머신러닝 프로젝트 처음부터 끝까지

> 캘리포니아 주택 가격 예측 프로젝트를 통해 머신러닝 프로젝트의 전체 흐름을 정리한 학습 노트입니다.
>
> 실습 노트북: [02_end_to_machine_learning_project.ipynb](./02_end_to_machine_learning_project.ipynb)

## 1. 학습 목표

이 프로젝트의 목표는 캘리포니아 각 구역의 인구, 소득, 위치 등 여러 특성을 이용해 `median_house_value`, 즉 중간 주택 가격을 예측하는 것입니다.

이 문제는 다음과 같이 정의할 수 있습니다.

- 지도 학습: 정답인 주택 가격이 포함되어 있습니다.
- 회귀 문제: 연속적인 숫자 값을 예측합니다.
- 배치 학습: 준비된 전체 데이터로 한 번에 학습합니다.
- 주요 평가 지표: RMSE(Root Mean Squared Error)

전체 작업 흐름은 다음과 같습니다.

```text
문제 정의
  → 데이터 수집
  → 훈련·테스트 세트 분리
  → 데이터 탐색과 시각화
  → 데이터 전처리
  → 모델 선택과 교차 검증
  → 하이퍼파라미터 튜닝
  → 테스트 세트 최종 평가
  → 모델 저장 및 사용
```

---

## 2. 데이터 가져오기와 구조 확인

### 2.1 데이터 다운로드

`load_housing_data()`는 압축 파일이 로컬에 없을 때만 데이터를 내려받고, 압축을 해제한 뒤 CSV 파일을 데이터프레임으로 반환합니다.

```python
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))
```

### 2.2 데이터 구조 확인

처음 데이터를 받으면 바로 모델을 만드는 것이 아니라 구조와 품질부터 확인해야 합니다.

| 코드 | 확인하는 내용 |
|---|---|
| `housing.head()` | 앞부분의 실제 데이터와 열 이름 |
| `housing.info()` | 행 개수, 자료형, 결측값 여부 |
| `housing.describe()` | 수치형 특성의 분포와 요약 통계 |
| `value_counts()` | 범주형 특성의 카테고리별 개수 |
| `housing.hist()` | 각 수치형 특성의 전체적인 분포 |

데이터에는 다음과 같은 특성이 있습니다.

- 위치: `longitude`, `latitude`
- 주택: `housing_median_age`, `total_rooms`, `total_bedrooms`
- 인구: `population`, `households`
- 소득: `median_income`
- 해안 접근성: `ocean_proximity`
- 예측 대상: `median_house_value`

### 2.3 히스토그램에서 발견한 특징

- `median_income`은 실제 달러 금액이 아니라 스케일이 조정된 값입니다.
- `housing_median_age`와 `median_house_value`에는 상한이 존재합니다.
- 특성마다 값의 범위가 크게 다릅니다.
- 여러 특성이 오른쪽으로 긴 꼬리를 가진 분포입니다.

특히 타깃인 `median_house_value`에 상한이 있다는 점이 중요합니다. 모델은 상한보다 비싼 주택도 모두 상한 가격에 가깝다고 학습할 수 있기 때문입니다.

---

## 3. 훈련 세트와 테스트 세트 만들기

### 3.1 테스트 세트를 미리 분리하는 이유

테스트 데이터를 미리 들여다보면 그 데이터에서 발견한 패턴을 모델 설계에 반영하게 됩니다. 그러면 테스트 세트가 더 이상 처음 보는 데이터가 아니게 되며, 평가 결과가 실제보다 좋게 나타날 수 있습니다. 이를 **데이터 스누핑 편향(data snooping bias)** 이라고 합니다.

따라서 테스트 세트는 프로젝트 초기에 분리하고 최종 평가 전까지 사용하지 않습니다.

### 3.2 무작위 분할과 재현성

```python
train_set, test_set = train_test_split(
    housing,
    test_size=0.2,
    random_state=42
)
```

- `test_size=0.2`: 전체 데이터의 20%를 테스트 세트로 사용합니다.
- `random_state=42`: 실행할 때마다 동일하게 분할되도록 난수를 고정합니다.

`random_state`는 현재 데이터에 대한 분할을 재현해 주지만, 새로운 행이 추가되면 기존 행의 소속이 달라질 수 있습니다.

### 3.3 해시 기반 분할

```python
def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(
        lambda id_: is_id_in_test_set(id_, test_ratio)
    )
    return data.loc[~in_test_set], data.loc[in_test_set]
```

고유 ID의 해시값을 기준으로 분할하면 데이터가 추가되어도 기존 샘플의 훈련·테스트 소속이 안정적으로 유지됩니다. 단, 신뢰할 수 있고 변하지 않는 고유 ID가 필요합니다.

### 3.4 계층적 샘플링

중간 소득은 주택 가격을 예측하는 중요한 특성입니다. 테스트 세트에도 전체 데이터와 비슷한 소득 분포가 유지되도록 `income_cat`을 만들고 계층적 샘플링을 사용합니다.

```python
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

strat_train_set, strat_test_set = train_test_split(
    housing,
    test_size=0.2,
    stratify=housing["income_cat"],
    random_state=42
)
```

여기서 `stratify=housing["income_cat"]`은 각 소득 구간의 비율을 훈련 세트와 테스트 세트에 비슷하게 유지하는 핵심 설정입니다.

분할을 마치면 임시로 만든 `income_cat`은 제거합니다.

```python
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

---

## 4. 데이터 탐색과 시각화

탐색은 테스트 세트가 아닌 **훈련 세트의 복사본**으로 진행합니다.

```python
housing = strat_train_set.copy()
```

### 4.1 지리적 산점도

```python
housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.2
)
```

이 그래프는 경도와 위도를 이용한 **지리적 산점도(scatter plot)** 입니다. `alpha=0.2`로 투명도를 적용하면 점이 많이 겹치는 밀집 지역이 더 선명하게 보입니다.

인구와 주택 가격을 함께 표현할 수도 있습니다.

```python
housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    s=housing["population"] / 100,
    c="median_house_value",
    cmap="jet",
    colorbar=True
)
```

- 점의 크기 `s`: 인구 규모
- 점의 색상 `c`: 중간 주택 가격
- 관찰 결과: 해안과 대도시 주변의 주택 가격이 높은 경향이 있습니다.

### 4.2 상관관계 조사

```python
corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
```

상관계수는 두 변수의 **선형 관계**를 나타냅니다.

- `1`에 가까움: 강한 양의 선형 관계
- `-1`에 가까움: 강한 음의 선형 관계
- `0`에 가까움: 선형 관계가 약함

이 데이터에서는 `median_income`이 `median_house_value`와 가장 강한 양의 상관관계를 보입니다.

주의할 점은 상관관계가 인과관계를 뜻하지 않으며, 상관계수만으로 비선형 관계를 충분히 설명할 수도 없다는 것입니다.

### 4.3 새로운 특성 조합

기존 특성의 단순한 총량보다 비율이 더 유용할 수 있습니다.

```python
housing["rooms_per_house"] = (
    housing["total_rooms"] / housing["households"]
)
housing["bedrooms_ratio"] = (
    housing["total_bedrooms"] / housing["total_rooms"]
)
housing["people_per_house"] = (
    housing["population"] / housing["households"]
)
```

- `rooms_per_house`: 가구당 방 개수
- `bedrooms_ratio`: 전체 방 중 침실 비율
- `people_per_house`: 가구당 인원

이처럼 원본 특성에 도메인 지식을 적용해 더 의미 있는 특성을 만드는 작업을 **특성 공학(feature engineering)** 이라고 합니다.

---

## 5. 머신러닝 알고리즘을 위한 데이터 준비

먼저 입력 특성과 타깃을 분리합니다.

```python
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```

### 5.1 결측값 처리

`total_bedrooms`에는 결측값이 있습니다. 처리 방법은 다음과 같습니다.

1. 결측값이 있는 행 제거
2. 해당 특성 전체 제거
3. 평균이나 중앙값 등으로 결측값 대체

노트북에서는 데이터를 최대한 유지하기 위해 중앙값 대체를 사용합니다.

```python
imputer = SimpleImputer(strategy="median")
imputer.fit(housing_num)
X = imputer.transform(housing_num)
```

핵심은 `fit()`과 `transform()`의 역할을 구분하는 것입니다.

- `fit()`: 훈련 데이터에서 각 열의 중앙값을 학습합니다.
- `transform()`: 학습한 중앙값으로 실제 결측값을 채웁니다.
- 테스트 데이터에는 `fit()`하지 않고 훈련 데이터에서 학습한 값으로 `transform()`만 해야 합니다.

이 원칙을 지키지 않으면 테스트 데이터의 정보가 훈련 과정에 섞이는 **데이터 누수(data leakage)** 가 발생합니다.

### 5.2 범주형 특성 인코딩

`ocean_proximity`는 문자열로 이루어진 범주형 특성입니다.

`OrdinalEncoder`는 각 범주를 하나의 정수로 변환하지만, 순서가 없는 범주 사이에 잘못된 거리 관계를 만들 수 있습니다.

```text
INLAND = 0, NEAR OCEAN = 1, NEAR BAY = 2
```

위와 같이 변환하면 모델은 `INLAND`와 `NEAR OCEAN`이 `NEAR BAY`보다 더 가깝다고 오해할 수 있습니다. 따라서 순서가 없는 범주에는 원-핫 인코딩이 적합합니다.

```python
cat_encoder = OneHotEncoder(handle_unknown="ignore")
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
```

`handle_unknown="ignore"`는 학습할 때 없던 새로운 범주가 들어와도 오류를 내지 않고 해당 원-핫 값을 모두 0으로 처리합니다.

### 5.3 특성 스케일링

특성들의 범위가 크게 다르면 많은 머신러닝 알고리즘이 제대로 학습하지 못합니다.

| 방법 | 계산 | 특징 |
|---|---|---|
| Min-Max 스케일링 | `(값 - 최솟값) / (최댓값 - 최솟값)` | 일반적으로 0~1 범위, 이상치에 민감 |
| 표준화 | `(값 - 평균) / 표준편차` | 평균 0, 표준편차 1, 이상치에 상대적으로 덜 민감 |

```python
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
```

### 5.4 치우친 분포와 로그 변환

오른쪽 꼬리가 긴 양수 특성은 로그 변환을 적용하면 분포가 덜 치우치게 됩니다.

```python
log_transformer = FunctionTransformer(
    np.log,
    inverse_func=np.exp
)
```

- `np.log`: 현재 데이터를 로그 값으로 변환합니다.
- `inverse_func=np.exp`: 필요할 때 원래 값으로 되돌리는 역변환 함수입니다.

### 5.5 RBF 유사도 특성

RBF(Radial Basis Function)는 한 지점과 기준점 사이의 거리를 유사도로 바꿉니다. 기준점에 가까우면 1에 가깝고, 멀수록 0에 가까워집니다.

```python
rbf_transformer = FunctionTransformer(
    rbf_kernel,
    kw_args=dict(Y=[[35.]], gamma=0.1)
)
```

이 변환기는 `housing_median_age`가 기준 나이 35와 얼마나 유사한지 계산합니다.

- `Y=[[35.]]`: 비교할 기준점
- `gamma`: 거리가 멀어질 때 유사도가 감소하는 속도

`Y`는 그래프의 y축이 아니라 입력 샘플과 비교할 **기준 벡터**입니다.

### 5.6 사용자 정의 변환기

`ClusterSimilarity`는 다음 순서로 동작합니다.

1. `fit()`에서 `KMeans`로 지역 중심을 학습합니다.
2. `transform()`에서 각 구역과 각 클러스터 중심 사이의 RBF 유사도를 계산합니다.
3. 위도·경도 자체보다 모델이 활용하기 쉬운 지역 유사도 특성을 만듭니다.

사용자 정의 변환기를 사이킷런 파이프라인에서 사용하려면 일반적으로 다음 규칙을 지켜야 합니다.

- 생성자에는 하이퍼파라미터만 저장합니다.
- `fit(X, y=None)`은 학습 후 `self`를 반환합니다.
- 학습된 속성 이름은 `_`로 끝냅니다. 예: `kmeans_`
- `transform(X)`은 변환된 데이터를 반환합니다.
- 출력 특성 이름이 필요하면 `get_feature_names_out()`을 구현합니다.

> 노트북의 `StandardScalerClone`에는 `__int__`라고 작성된 부분이 있습니다. 생성자는 `__init__`이 올바른 이름입니다. 해당 클래스는 이후 최종 파이프라인에서는 사용되지 않습니다.

---

## 6. 전처리 파이프라인 구성

### 6.1 Pipeline을 사용하는 이유

전처리를 수동으로 실행하면 훈련 데이터와 테스트 데이터에 서로 다른 처리를 적용하거나, 단계 순서를 실수하기 쉽습니다. `Pipeline`은 여러 변환을 정해진 순서대로 묶어 줍니다.

```python
num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)
```

### 6.2 ColumnTransformer

`ColumnTransformer`는 열 그룹마다 서로 다른 전처리를 적용합니다.

```python
preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object))
)
```

- 수치형 열: 결측값 대체 후 표준화
- 범주형 열: 결측값 대체 후 원-핫 인코딩

### 6.3 최종 전처리 파이프라인

노트북의 최종 `preprocessing`은 다음 작업을 한 번에 수행합니다.

| 변환기 이름 | 입력 열 | 처리 내용 |
|---|---|---|
| `bedrooms` | `total_bedrooms`, `total_rooms` | 침실 비율 생성 후 표준화 |
| `rooms_per_house` | `total_rooms`, `households` | 가구당 방 개수 생성 후 표준화 |
| `people_per_house` | `population`, `households` | 가구당 인원 생성 후 표준화 |
| `log` | 방, 인구, 가구, 소득 관련 열 | 결측값 대체, 로그 변환, 표준화 |
| `geo` | `latitude`, `longitude` | 클러스터 중심과의 RBF 유사도 생성 |
| `cat` | 범주형 열 | 최빈값 대체, 원-핫 인코딩 |
| `remainder` | 나머지 수치형 열 | 중앙값 대체, 표준화 |

전처리 결과는 처음에 24개의 특성을 가진 배열이 됩니다. 이후 탐색 과정에서 `n_clusters`가 바뀌면 최종 특성 수도 함께 바뀝니다.

### 6.4 파이프라인의 가장 큰 장점

모델까지 파이프라인으로 묶으면 원본 데이터프레임을 그대로 `fit()`과 `predict()`에 전달할 수 있습니다.

```python
model = make_pipeline(preprocessing, RandomForestRegressor())
model.fit(housing, housing_labels)
predictions = model.predict(new_data)
```

예측 시에도 학습 때와 정확히 같은 전처리가 자동으로 적용됩니다.

---

## 7. 모델 선택과 훈련

평가 지표로 사용한 RMSE는 다음과 같습니다.

$$
RMSE = \sqrt{\frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^2}
$$

오차를 제곱하기 때문에 큰 오차에 더 큰 페널티를 부여하며, 단위가 타깃과 동일하여 달러 단위로 해석할 수 있습니다.

### 7.1 선형 회귀

```python
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
```

- 훈련 세트 RMSE: 약 `$68,648`
- 해석: 훈련 데이터에서도 오차가 크므로 모델이 데이터의 복잡한 패턴을 충분히 학습하지 못했습니다.
- 진단: **과소적합(underfitting)**

> 노트북 설명 중 이 결과를 과대적합으로 표현한 부분이 있지만, 훈련 오차 자체가 큰 경우는 과소적합으로 보는 것이 맞습니다.

### 7.2 결정 트리

```python
tree_reg = make_pipeline(
    preprocessing,
    DecisionTreeRegressor(random_state=42)
)
```

- 훈련 세트 RMSE: `0`
- 10-폴드 교차 검증 평균 RMSE: 약 `$67,153`
- 표준편차: 약 `$1,964`
- 진단: 훈련 데이터를 거의 외워 버린 **과대적합(overfitting)**

훈련 오차가 0이라고 해서 좋은 모델은 아닙니다. 새로운 데이터에서도 성능이 좋은지 반드시 검증해야 합니다.

### 7.3 랜덤 포레스트

```python
forest_reg = make_pipeline(
    preprocessing,
    RandomForestRegressor(random_state=42)
)
```

- 10-폴드 교차 검증 평균 RMSE: 약 `$47,003`
- 표준편차: 약 `$1,048`
- 선형 회귀와 결정 트리보다 훨씬 좋은 결과를 보였습니다.

랜덤 포레스트는 서로 다른 여러 결정 트리의 예측을 결합하는 **앙상블 모델**입니다. 하나의 결정 트리보다 분산을 줄이고 일반화 성능을 높일 수 있습니다.

### 7.4 교차 검증

```python
rmses = -cross_val_score(
    model,
    housing,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)
```

10-폴드 교차 검증은 훈련 세트를 10개로 나누고, 매번 9개 폴드로 학습하고 나머지 1개 폴드로 평가합니다.

사이킷런의 `cross_val_score()`는 점수가 클수록 좋다는 규칙을 따르기 때문에 RMSE에 음수를 붙인 값을 반환합니다. 실제 RMSE를 얻기 위해 앞에 `-`를 붙입니다.

교차 검증으로 알 수 있는 것은 다음 두 가지입니다.

- 평균: 모델의 예상 일반화 성능
- 표준편차: 평가 결과가 폴드마다 얼마나 흔들리는지

---

## 8. 모델 미세 튜닝

### 8.1 파이프라인 하이퍼파라미터 이름

파이프라인 내부의 하이퍼파라미터는 `단계이름__매개변수이름` 형식으로 접근합니다.

```text
preprocessing__geo__n_clusters
random_forest__max_features
```

### 8.2 GridSearchCV

`GridSearchCV`는 지정한 모든 하이퍼파라미터 조합을 교차 검증으로 평가합니다.

```python
grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=3,
    scoring="neg_root_mean_squared_error"
)
```

노트북의 최적 조합은 다음과 같습니다.

```python
{
    "preprocessing__geo__n_clusters": 15,
    "random_forest__max_features": 6
}
```

최상위 조합의 평균 검증 RMSE는 약 `$43,953`입니다.

### 8.3 RandomizedSearchCV

`RandomizedSearchCV`는 모든 조합을 검사하지 않고 지정한 분포에서 값을 무작위로 뽑아 정해진 횟수만 평가합니다.

```python
param_distribs = {
    "preprocessing__geo__n_clusters": randint(low=3, high=50),
    "random_forest__max_features": randint(low=2, high=20)
}
```

노트북의 최상위 랜덤 탐색 결과는 다음과 같습니다.

- `n_clusters=45`
- `max_features=9`
- 평균 검증 RMSE: 약 `$41,987`

| 탐색 방법 | 적합한 상황 |
|---|---|
| Grid Search | 후보 값과 조합 수가 적을 때 |
| Random Search | 탐색 범위가 넓거나 하이퍼파라미터가 많을 때 |

연속적인 값이나 탐색 범위가 매우 넓은 값은 적절한 확률 분포를 사용하는 것이 좋습니다.

- `randint`: 일정 범위의 정수
- `uniform`: 일정 범위의 연속값
- `expon`: 특정 스케일 주변을 주로 탐색
- `loguniform`: 적절한 값의 자릿수 자체를 모를 때

---

## 9. 최종 모델 분석과 테스트 평가

### 9.1 특성 중요도

```python
feature_importances = (
    final_model["random_forest"].feature_importances_
)
```

노트북에서 중요도가 높게 나타난 주요 특성은 다음과 같습니다.

1. 로그 변환된 `median_income`: 약 `0.190`
2. `ocean_proximity_INLAND`: 약 `0.077`
3. `bedrooms_ratio`: 약 `0.065`
4. `rooms_per_house`: 약 `0.057`
5. `people_per_house`: 약 `0.049`
6. 일부 지역 클러스터 유사도 특성

특성 중요도를 통해 덜 유용한 특성을 제거하거나, 중요한 특성과 관련된 새로운 특성을 추가하는 후속 실험을 설계할 수 있습니다.

단, 랜덤 포레스트의 기본 특성 중요도는 값의 종류가 많거나 분할 기회가 많은 특성을 더 중요하게 평가할 수 있으므로 절대적인 인과관계로 해석하면 안 됩니다.

### 9.2 테스트 세트 최종 평가

```python
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)
final_rmse = mean_squared_error(
    y_test,
    final_predictions,
    squared=False
)
```

- 최종 테스트 RMSE: 약 `$41,549`
- RMSE의 95% 신뢰 구간: 약 `$39,395 ~ $43,597`

신뢰 구간은 동일한 데이터 생성 과정에서 테스트 세트를 반복해서 뽑아 평가한다면, 모델의 실제 RMSE가 있을 것으로 추정되는 범위를 나타냅니다.

테스트 세트는 마지막에 한 번만 사용해야 합니다. 테스트 결과를 보고 다시 하이퍼파라미터를 반복해서 조정하면 테스트 세트에도 과대적합될 수 있습니다.

---

## 10. 모델 저장과 다시 불러오기

### 10.1 모델 저장

```python
import joblib

joblib.dump(
    final_model,
    "my_california_housing_model.pkl"
)
```

저장되는 `final_model`에는 다음 요소가 함께 포함됩니다.

- 결측값 처리
- 비율 및 로그 특성 생성
- 범주형 인코딩
- 클러스터 유사도 계산
- 특성 스케일링
- 학습된 랜덤 포레스트 모델

즉, 모델만 저장하는 것이 아니라 **학습이 완료된 전체 전처리·예측 파이프라인**을 저장합니다.

### 10.2 모델 불러오기와 예측

```python
final_model_reloaded = joblib.load(
    "my_california_housing_model.pkl"
)

predictions = final_model_reloaded.predict(new_data)
```

원본 형식의 새로운 데이터를 전달하면 저장된 전처리가 자동으로 적용된 후 예측이 수행됩니다.

주의할 점은 다음과 같습니다.

- `.pkl`은 저장할 때 사용한 Python과 라이브러리 버전의 영향을 받습니다.
- `ClusterSimilarity`, `column_ratio` 같은 사용자 정의 코드도 불러오는 환경에 존재해야 합니다.
- 출처를 신뢰할 수 없는 `.pkl`은 악성 코드가 실행될 수 있으므로 불러오지 않습니다.
- 이 파일은 100MB를 넘으므로 일반 GitHub 저장소에서는 제외합니다.

프로젝트의 `.gitignore`에는 다음을 추가할 수 있습니다.

```gitignore
*.pkl
```

---

## 11. 과소적합과 과대적합 구분

| 훈련 성능 | 검증 성능 | 진단 |
|---|---|---|
| 나쁨 | 나쁨 | 과소적합 가능성이 큼 |
| 매우 좋음 | 나쁨 | 과대적합 가능성이 큼 |
| 좋음 | 좋음 | 일반화가 잘 된 상태 |

이 노트북에서는 다음과 같이 구분할 수 있습니다.

- 선형 회귀: 훈련 RMSE부터 크므로 과소적합
- 결정 트리: 훈련 RMSE는 0이지만 검증 RMSE가 크므로 과대적합
- 랜덤 포레스트: 결정 트리보다 검증 RMSE가 크게 개선됨

과소적합을 개선하는 방법:

- 더 강력한 모델 사용
- 더 유용한 특성 추가
- 불필요한 규제 완화

과대적합을 개선하는 방법:

- 모델 단순화 또는 규제 강화
- 훈련 데이터 추가
- 불필요한 특성 제거
- 교차 검증을 이용한 하이퍼파라미터 선택

---

## 12. 핵심 클래스와 메서드 정리

| 도구 | 역할 |
|---|---|
| `train_test_split` | 훈련·테스트 데이터 분리 |
| `SimpleImputer` | 결측값 대체 |
| `OneHotEncoder` | 범주형 데이터를 원-핫 벡터로 변환 |
| `StandardScaler` | 평균 0, 표준편차 1로 표준화 |
| `FunctionTransformer` | 일반 함수를 사이킷런 변환기로 사용 |
| `Pipeline` | 전처리와 모델의 순차 실행 |
| `ColumnTransformer` | 열마다 다른 전처리 적용 |
| `cross_val_score` | 교차 검증으로 일반화 성능 추정 |
| `GridSearchCV` | 지정한 모든 조합 탐색 |
| `RandomizedSearchCV` | 분포에서 무작위 조합 탐색 |
| `joblib.dump` | 학습된 모델 저장 |
| `joblib.load` | 저장된 모델 복원 |

사이킷런 객체의 공통 사용 흐름도 기억해 두면 좋습니다.

```text
fit()           학습
transform()     입력 데이터 변환
fit_transform() 학습 후 바로 변환
predict()       예측
```

---

## 13. 프로젝트에서 꼭 기억할 규칙

1. 테스트 세트는 프로젝트 초기에 분리하고 마지막 평가 전까지 보지 않습니다.
2. 데이터 탐색은 훈련 세트만 사용합니다.
3. 결측값 대체와 스케일링 기준은 훈련 데이터에서만 학습합니다.
4. 범주형 특성에는 데이터의 의미에 맞는 인코딩을 사용합니다.
5. 특성 조합은 원본 특성보다 유용한 정보를 만들 수 있습니다.
6. 전처리와 모델을 하나의 파이프라인으로 묶어 데이터 누수와 처리 불일치를 방지합니다.
7. 훈련 오차만 보고 모델을 평가하지 않고 교차 검증을 사용합니다.
8. 하이퍼파라미터는 검증 결과로 선택하고 테스트 세트는 최종 확인에만 사용합니다.
9. 최종 모델을 저장할 때는 전처리 파이프라인까지 함께 저장합니다.
10. 저장된 모델을 재사용하려면 라이브러리 버전과 사용자 정의 클래스도 관리해야 합니다.

---

## 14. 복습 질문

1. 테스트 세트를 프로젝트 초기에 분리해야 하는 이유는 무엇인가?
2. `random_state`를 고정하는 것과 해시 기반 분할은 어떤 차이가 있는가?
3. `stratify=housing["income_cat"]`은 무엇을 보존하는가?
4. `SimpleImputer.fit()`을 테스트 데이터에 다시 실행하면 왜 안 되는가?
5. `OrdinalEncoder`보다 `OneHotEncoder`가 적합한 범주형 데이터는 무엇인가?
6. 표준화와 Min-Max 스케일링은 어떻게 다른가?
7. `FunctionTransformer(np.log, inverse_func=np.exp)`에서 두 함수의 역할은 무엇인가?
8. RBF의 `Y`와 `gamma`는 각각 무엇을 의미하는가?
9. `Pipeline`과 `ColumnTransformer`는 각각 어떤 문제를 해결하는가?
10. 결정 트리의 훈련 RMSE가 0인데도 좋은 모델이라고 할 수 없는 이유는 무엇인가?
11. `cross_val_score()`에서 RMSE 점수 앞에 음수를 붙이는 이유는 무엇인가?
12. Grid Search와 Random Search는 언제 각각 사용하는 것이 좋은가?
13. 테스트 세트 결과를 본 뒤 계속 모델을 수정하면 어떤 문제가 생기는가?
14. `joblib`으로 저장한 최종 모델에 전처리 과정까지 포함되는 이유는 무엇인가?

---

## 15. 한 문장 요약

> 좋은 머신러닝 시스템은 모델 하나만 잘 고르는 것이 아니라, 올바른 데이터 분할부터 탐색·전처리·검증·튜닝·최종 평가·저장까지의 전체 과정을 재현 가능한 파이프라인으로 만드는 것입니다.
