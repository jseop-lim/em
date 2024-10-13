# COSE474 Machine Learning - GMM Classification with EM Algorithm

2020170812 임정섭

## 파일 구조

## 실행 방법

## EM 알고리즘 상세

### 초기화 방법

어떤 class k에 대한 group z의 mean vector μ_kz, covariance matrix Σ_kz, prior probability ϕ_kz를 초기화하는 방법은 다음과 같다.

아래 우선순위에 따라 초기화 방법을 선택했다.

1. 계산이 빠를 것
2. 전체 데이터셋의 특성을 잘 반영할 것
3. 수렴 속도가 빠를 것

#### μ_kz

class k의 모든 sample들 중에서 무작위로 n개의 sample을 추출하여 평균을 μ_kz로 설정한다.

기각된 방법:

- 랜덤 벡터 생성: 랜덤 벡터를 생성하면 데이터셋의 특성을 반영하지 못한다. 또한, 랜덤 벡터가 너무 멀리 떨어져 있으면 수렴 속도가 느려질 것이라고 판단했다.
- class k의 모든 sample들의 평균을 μ_kz로 설정: 서로 다른 group 간 초기 평균이 완전히 같으므로 최적화 과정에서 데이터를 group으로 정밀하게 분류하지 못할 것이라고 판단했다. 또한, 실제 실행시켜보니 그로 인해 공분산 행렬이 singular matrix가 되는 경우가 발생했다.
- class k의 모든 sample들 중에서 무작위로 1개의 sample을 선택: 계산이 빠르나, 전체 데이터셋의 특성을 잘 반영하지 못할 것이라고 판단했다.

선택하는 n의 값이 커질수록 초기 평균의 분산이 작아지나, 너무 작으면 group 간의 차이를 잘 반영하지 못할 것이라고 판단했다. 따라서 n=100으로 설정했다.

#### Σ_kz

class k의 모든 sample에 대한 covariance matrix를 계산하여 Σ_kz로 설정한다.

근거: target covariance도 group마다 같을 수 있으므로, 초기화 과정에서는 sample covariance를 사용하는 것이 적절하다고 판단했다. 또한, sample covariance를 사용하면 전체 데이터셋의 특성을 잘 반영할 수 있다.

#### ϕ_kz

uniform distribution을 따른다고 가정하고 1/Z로 설정한다.

근거: group의 prior probability는 초기화 시점에 큰 직관이 없었으므로 균등하게 설정했다.

### 종료 조건

## 모델 선택

### 데이터셋

- input x는 D=13차원의 연속형 변수이다.
- output y는 K=2개의 클래스를 갖는 범주형 변수이다.
- 데이터셋에는 N=60290개의 sample이 존재한다.

### Inductive Biases

- iid 가정: 데이터셋의 sample끼리는 서로 독립적이고 동일한 분포를 따른다.
  - P(x_1, x_2, ..., x_N) = Π_{n=1}^{N} P(x_n)
- 베이지안 결정: x의 예측 결과는 각 class에 대한 posterior가 최대가 되는 class로 결정된다.
  - posterior P(c_k|x) = P(x|c_k) *P(c_k) / Σ_{j=1}^{K} P(x|c_j)* P(c_j)
- likelihood 분포: class k에 대한 likelihood 분포 P(x|c_k)를 Mixtured Gaussian Distribution으로 가정한다.
  - P(x|c_k) = Σ_{z=1}^{Z} P(x|g_z, c_k, θ_kz) * P(g_z|c_k, θ_z)
  - P(x|g_z, c_k, θ_kz) ~ N(x|μ_kz, Σ_kz)
  - P(g_z|c_k, θ_z) ~ Multinomial(g_z|ϕ_k) => P(g_z|c_k, θ_z) = ϕ_z
- prior 분포: class k에 대한 prior 분포 P(c_k)를 Multinomial Distribution으로 가정한다.
  - P(c_k) ~ Multinomial(c_k|π) => P(c_k|θ_k) = π_k
- 추정할 parameters: θ = {μ, Σ, ϕ, π}
  - μ_kz: class k일 때 group z의 D x 1 mean vector (μ: K x Z x D tensor)
  - Σ_kz: class k일 때 group z의 D x D covariance matrix (Σ: K x Z x D x D tensor)
  - ϕ_kz: class k일 때 group z의 prior probability (ϕ: K x Z matrix)
  - π_k: class k의 prior probability (π: K x 1 vector)
- Searching Strategy: EM 알고리즘
  - Loss Function: Minus expectation of log likelihood (-1 * Baum's Q function) = - Q(θ' | θ)

## 성능 평가
