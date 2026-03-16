# EIRM × LLM: 한국어·한국어교육·학문목적한국어 적용 방안

**Wilson, De Boeck & Carstensen (2008) 설명적 문항반응모형의 한국어 특화 확장**

---

**서용덕**
yndk@sogang.ac.kr
서강대학교 인공지능/글로벌한국학부

---

## 목차

1. [한국어교육 맥락의 특수성과 EIRM 적용 가능성](#sec0)
2. [연구 1: TOPIK 문항의 언어적 속성 자동 추출 (LLM-LLTM)](#sec1)
3. [연구 2: 학습자 작문·발화 임베딩을 공변량으로 (LLM-LR-Rasch)](#sec2)
4. [연구 3: LLM 기반 사전분포로 소표본 TOPIK 문항 보정](#sec3)
5. [연구 4: 학문목적한국어 문항 자동 생성 루프 (Active Learning)](#sec4)
6. [연구 5: 정오답 + 작문 텍스트 결합 모델 (Joint Model)](#sec5)
7. [연구 6: 언어권별 DIF 가설 생성 및 검증](#sec6)
8. [공개 데이터 및 데이터 획득 방법 종합](#sec7)
9. [연구 로드맵 및 우선순위](#sec8)

---

<a name="sec0"></a>
## 1. 한국어교육 맥락의 특수성과 EIRM 적용 가능성

### 1.1 왜 한국어교육에 EIRM이 필요한가

한국어능력시험(TOPIK)은 연간 약 40만 명이 응시하는 세계 최대 규모의 한국어 숙달도 평가 도구이다. 그러나 기존 연구는 주로 고전검사이론(CTT)에 의존하며, EIRM과 같은 설명적 측정 모형의 적용은 극히 제한적이다. 이는 크게 두 가지 구조적 제약에 기인한다.

첫째, NIIED(국립국제교육원)는 「한국어능력시험 성적 처리 규정 제12조」에 따라 문항별 응답 데이터를 비공개로 유지한다. TOPIK의 관리 기관인 국립국제교육원은 관련 규정에 따라 정보를 공개하지 않아, 비교적 최근의 시험 시행 데이터를 활용하기 어렵다.

둘째, 한국어는 교착어로서 형태소 복잡성, 경어법 체계, 한자어·고유어·외래어의 삼층 어휘 구조 등 언어 특수적 속성이 문항 난이도를 결정하는 핵심 요인임에도, 이를 체계적으로 수량화하는 방법론이 부재하다.

LLM과 EIRM의 결합은 이 두 제약을 동시에 돌파할 가능성을 제공한다. LLM은 기존에 전문가가 수작업으로 해야 했던 언어적 속성 코딩을 자동화하고, Bayesian EIRM은 소규모 데이터에서도 안정적인 추정을 가능하게 한다.

### 1.2 한국어교육 특수 설계 행렬의 예시

TOPIK II 읽기·쓰기 문항의 설계 행렬 $\mathbf{X} \in \mathbb{R}^{I \times K}$는 아래와 같이 구성될 수 있다.

| 속성 범주 | 속성 $k$ | 측정 방식 | 한국어 특수성 |
|---------|---------|---------|------------|
| **어휘** | 어휘 등급 (1~6급) | KSL 어휘 목록 참조 | 한자어/고유어/외래어 층위 |
| **어휘** | 어휘 밀도 (TTR) | 형태소 분석기 | Mecab-ko, Kiwi |
| **문법** | 문법 항목 등급 | 국제통용 한국어 문법 목록 | 조사·어미 복잡성 |
| **문법** | 문장 길이 (어절 수) | 자동 계산 | — |
| **문법** | 내포절 깊이 | 파싱 결과 | 관형절 중첩 구조 |
| **담화** | 텍스트 유형 | 분류 (설명/논증/서사) | 학문 텍스트 vs. 실용 텍스트 |
| **문화** | 문화 배경 지식 필요도 | LLM 평가 (0~1) | 한국 사회·역사 맥락 |
| **과제** | 인지적 처리 깊이 | LLM 평가 (1~5) | 이해/적용/분석/평가 |

LLTM 난이도 예측 공식:

$$\beta_i' = \beta_{\text{어휘등급}} \cdot X_{i,\text{어휘}} + \beta_{\text{문법복잡도}} \cdot X_{i,\text{문법}} + \beta_{\text{담화유형}} \cdot X_{i,\text{담화}} + \beta_{\text{문화지식}} \cdot X_{i,\text{문화}}$$

---

<a name="sec1"></a>
## 2. 연구 1: TOPIK 문항의 언어적 속성 자동 추출 (LLM-LLTM)

### 2.1 연구 문제

TOPIK 문항의 난이도는 어휘 등급, 문법 복잡도, 텍스트 유형, 문화 배경 지식 요구도의 어떤 조합으로 설명되는가? LLM이 추출한 속성이 전문가 코딩보다 문항 난이도를 더 잘 예측하는가?

### 2.2 한국어 특화 속성 추출 파이프라인

**형태소 분석 기반 속성 (자동 계산):**

```python
# KoNLPy + Kiwi를 이용한 형태소 수준 속성 추출
from kiwipiepy import Kiwi
kiwi = Kiwi()

def extract_morphological_features(text):
    result = kiwi.analyze(text)[0][0]
    tokens = [t for t in result]
    
    features = {
        'mean_word_length': np.mean([len(t.form) for t in tokens]),
        'pos_diversity': len(set(t.tag for t in tokens)) / len(tokens),
        'eomi_complexity': sum(1 for t in tokens if t.tag.startswith('E')) / len(tokens),
        'josa_ratio': sum(1 for t in tokens if t.tag == 'JX') / len(tokens),
        'hanja_ratio': sum(1 for c in text if '\u4e00' <= c <= '\u9fff') / len(text),
        'clause_depth': estimate_clause_depth(result)
    }
    return features
```

**LLM 기반 속성 추출 (고차원 인지적 속성):**

TOPIK 문항에 특화된 프롬프트 구조:

```
다음 한국어능력시험(TOPIK) 문항을 분석하십시오.

[문항 텍스트]
{item_text}

아래 기준에 따라 각 속성의 값을 1–5 척도로 평가하십시오:

1. 어휘 난이도: TOPIK 1급=1, 6급=5
2. 문법 복잡도: 단문=1, 다중 내포절·복합 어미=5
3. 문화 배경 지식: 불필요=1, 한국 고유 문화 필수=5
4. 추론 깊이: 표면적 이해=1, 심층 추론·비판적 사고=5
5. 담화 결속성: 독립 문장=1, 긴밀한 담화 구조 파악 필수=5

각 속성의 점수와 간략한 이유를 JSON 형식으로 출력하십시오.
```

### 2.3 모델 수식 및 비교

**M₀ (Rasch 기준선):**
$$\eta_{pi} = \theta_p - \beta_i, \quad \theta_p \sim \mathcal{N}(0, \sigma_\theta^2)$$

**M₁ (전문가 코딩 LLTM):**
$$\beta_i' = \sum_{k \in K_{\text{human}}} \beta_k X_{ik}^{\text{expert}}$$

**M₂ (LLM 추출 LLTM):**
$$\beta_i' = \sum_{k \in K_{\text{morph}}} \beta_k X_{ik}^{\text{auto}} + \sum_{k \in K_{\text{LLM}}} \beta_k f_{\text{LLM}}(\text{item}_i, k)$$

**M₃ (형태소 + LLM 혼합):**
$$\beta_i' = \mathbf{x}_i^{\top}\boldsymbol{\beta}, \quad \mathbf{x}_i = [\underbrace{X_{i,\text{어휘등급}},\ X_{i,\text{TTR}},\ X_{i,\text{어절수}}}_{\text{형태소 분석기}},\ \underbrace{X_{i,\text{문화}},\ X_{i,\text{추론}},\ X_{i,\text{담화}}}_{\text{LLM 추출}}]$$

**모델 비교 지표:**

$$r(\hat{\beta}^{\text{Rasch}}, \hat{\beta}') \geq 0.95 \implies \text{LLTM 구조 타당성 확인}$$

$$\Delta\text{WAIC} = \text{WAIC}_{M_1} - \text{WAIC}_{M_2} > 0 \implies \text{LLM 추출 우위}$$

### 2.4 공개 데이터 및 획득 방법

#### 즉시 사용 가능한 공개 데이터

**TOPIK 기출문항 (문항 텍스트):**

- **출처:** NIIED 공식 사이트 (www.topik.go.kr) — 과거 시험지 PDF 공개
- NIIED가 공개한 모든 시험지는 해당 페이지에서 이용 가능하며, 이 자료들은 TOPIK 시험을 철저히 준비하기에 충분하다.
- 공개 시험 회차: 약 60회차 이상, 회차당 읽기 40문항 + 듣기 30문항
- 활용 방법: PDF → 텍스트 추출 → 형태소 분석 + LLM 속성 추출

**CLIcK 데이터셋 (한국어 문화·언어 문항):**

- CLIcK은 1,995개의 QA 쌍으로 구성된 한국어 문화·언어 능력 벤치마크 데이터셋으로, 11개 범주의 공식 한국어 시험 및 교재에서 자료를 수집했으며, 각 문항마다 필요한 문화적·언어적 지식에 대한 세부 주석이 포함되어 있다.
- GitHub 공개: `https://github.com/naver-ai/CLIcK`
- LLTM 속성 행렬의 "문화 지식 필요도" 기준선 코딩에 활용

**국립국어원 한국어 학습자 말뭉치:**

- 국립국어원은 총 1,588만 어절의 한국어 학습자 말뭉치를 공개하고 있으며, 이는 108개 언어권 한국어 학습자들의 표본을 수집해 구축한 방대한 자료이다.
- 주소: https://kcorpus.korean.go.kr
- 활용: 학습자 수준별 어휘·문법 사용 패턴 → 속성 기준선 도출

#### 응답 데이터 획득 방법 (핵심 병목)

NIIED가 문항별 응답 데이터를 비공개하므로, 다음 대안적 방법으로 응답 행렬 $\mathbf{Y}_{P \times I}$를 구성한다.

**방법 A: 온라인 학습 플랫폼 협력**
- Quizlet, Anki 덱 기반 TOPIK 모의 플랫폼 운영자와 MOU 체결
- TOPIK GUIDE 모의고사 플랫폼 (www.topikguide.com) — 월간 수만 명 응시
- 기대 규모: $P \geq 2000$명, $I = 40$문항

**방법 B: 대학 한국어 교육 기관 네트워크**
- 국내 대학 언어교육원 TOPIK 준비반 수강생 응시 데이터 수집
- 기관당 100~300명 × 10개 기관 협력 → $P \approx 1500$명
- 윤리 승인: 각 기관 IRB + NIIED 연구 협력 신청

**방법 C: 크라우드소싱 실험**
- Prolific.ac / 크몽 등에서 TOPIK 학습 경험자 모집
- 선별된 30~50 문항에 대한 응답 수집 ($P \geq 500$, 약 5만 원 인센티브)

---

<a name="sec2"></a>
## 3. 연구 2: 학습자 작문·발화 임베딩을 공변량으로 (LLM-LR-Rasch)

### 3.1 연구 문제

학습자의 한국어 작문 텍스트나 구어 발화를 LLM으로 임베딩하면, 성별·모국어·학습 기간 등 기존 배경 변수 외에 추가적인 한국어 숙달도 분산을 설명할 수 있는가? 학문목적 텍스트(학술 에세이)와 일상 텍스트 중 어느 것이 숙달도 예측력이 더 높은가?

### 3.2 모델 수식

**표준 LR-Rasch (기준선):**

$$\eta_{pi} = \sum_{j=1}^{J} \vartheta_j Z_{pj} + \theta_p - \beta_i$$

여기서 $Z_{pj}$는 모국어 더미($j$ = 중국어/일본어/영어/기타), 학습 기간, 체류 기간, TOPIK 급수 등.

**LLM-임베딩 LR-Rasch (제안):**

$$\eta_{pi} = \underbrace{\mathbf{z}_p^{\top}\boldsymbol{\vartheta}}_{\text{배경 변수}} + \underbrace{\tilde{\mathbf{e}}_p^{\top}\boldsymbol{\gamma}}_{\text{작문 임베딩}} + \theta_p - \beta_i$$

**임베딩 생성 방법:**

학습자 $p$의 텍스트 $\mathbf{t}_p$를 한국어 특화 모델로 인코딩:

$$\mathbf{e}_p = \text{KoSBERT}(\mathbf{t}_p) \in \mathbb{R}^{768} \quad \text{또는} \quad \mathbf{e}_p = \text{GPT-4o-embed}(\mathbf{t}_p) \in \mathbb{R}^{3072}$$

$$\tilde{\mathbf{e}}_p = \text{PCA}(\mathbf{e}_p, d=20), \quad d \ll 768$$

**분산 분해:**

$$\sigma_{\text{total}}^2 = \underbrace{\boldsymbol{\vartheta}^{\top}\text{Var}(\mathbf{Z})\boldsymbol{\vartheta}}_{\text{배경 변수 설명}} + \underbrace{\boldsymbol{\gamma}^{\top}\text{Var}(\tilde{\mathbf{E}})\boldsymbol{\gamma}}_{\text{텍스트 임베딩 추가 설명}} + \underbrace{\sigma_\theta^2}_{\text{잔차}}$$

**임베딩의 한계 기여도:**

$$\Delta R^2_{\text{텍스트}} = \frac{\boldsymbol{\gamma}^{\top}\text{Var}(\tilde{\mathbf{E}})\boldsymbol{\gamma}}{\sigma_{\text{total}}^2}$$

### 3.3 학문목적한국어(KAP) 특화 설계

학문목적 한국어 맥락에서는 아래 3가지 텍스트 유형의 예측력을 비교한다.

| 텍스트 유형 | 과제 | 예상 정보량 | 획득 용이성 |
|-----------|------|-----------|-----------|
| 학술 에세이 | 논문 초록 / 수업 리포트 | 높음 | 대학 한국어 수업 |
| 수업 내 발화 전사 | 발표·토론 전사 | 중-높음 | 실험 설계 필요 |
| TOPIK 쓰기 답안 | TOPIK II 54번 (논술) | 높음 | TOPIK 쓰기 채점 DB |
| 일상 대화 | 자유 회화 | 낮음 | 언어 교환 앱 |

**텍스트 기여 지수 (해석 가능성):**

임베딩 차원 $d$와 한국어 언어 특성 간 상관:

$$\rho_d = \text{corr}(\tilde{e}_{p,d},\ \hat{\theta}_p^{\text{Rasch}})$$

$|\rho_d| > 0.3$인 차원 $d$에 대해 LLM으로 해당 차원의 언어적 의미를 해석.

### 3.4 공개 데이터 및 획득 방법

#### 즉시 사용 가능한 공개 데이터

**국립국어원 한국어 학습자 말뭉치 (오류 주석):**

- 이 말뭉치에는 학습자의 글쓰기와 말하기 자료를 수집한 원시 말뭉치, 단어의 구성 및 품사 정보가 주석된 형태 주석 말뭉치, 그리고 학습자의 오류 정보가 주석된 오류 주석 말뭉치가 포함되어 있다.
- 수준별 작문 텍스트 + TOPIK 급수 레이블 → 임베딩 공변량 + 기준 숙달도 동시 확보
- 언어권별 분류 가능: 일본어권, 중국어권, 영어권 등

**AI Hub 한국어 음성 데이터 (외국인):**

- AI Hub에서는 영어, 중·일어, 아시아어, 유럽어 모국어 사용자의 한국어 음성 데이터를 제공하며, 자동 평가, 말하기, 발음 오류 원인 레이블링, 철자 전사, 발음 전사 등의 태그가 포함되어 있다.
- 주소: https://aihub.or.kr (회원가입 후 승인 신청)
- 활용: 구어 발화 텍스트 + TOPIK 수준 메타데이터 → 발화 임베딩

**Elicited Imitation (EI) 연구 데이터:**

- NIA 지원 프로젝트를 통해 한국어 학습자 3,000시간의 음성 데이터와 국적, 모국어, TOPIK 수준 등 메타데이터가 구축되었다. 이 데이터는 독일어, 러시아어, 말레이어, 몽골어 등 다양한 모국어 화자를 포함한다.
- NIA(한국지능정보사회진흥원) 협력 연구 신청

#### 연구자 직접 수집 방법

**방법 A: 대학 한국어 학당 협력**
- 연세대 한국어학당, 서울대 언어교육원 등과 MOU
- TOPIK 준비 과정 수강생의 작문 과제 + TOPIK 성적 연계
- 예상 규모: 기관당 $P = 100 \sim 300$명

**방법 B: 학술 한국어 수업 내 데이터 수집**
- 한국 대학의 외국인 유학생 대상 학술 한국어(KAP) 수업 참여 관찰
- 수업 발표·토론 음성 녹음 → Whisper 전사 → KoSBERT 임베딩
- 동의서: 대학 IRB 승인 필수

---

<a name="sec3"></a>
## 4. 연구 3: LLM 기반 사전분포로 소표본 TOPIK 문항 보정

### 4.1 연구 문제

TOPIK은 전 세계 소규모 응시 지역(일부 국가의 응시자 수 $P < 100$)에서 문항 보정에 어려움을 겪는다. LLM이 제공하는 문항별 정보적 사전분포는 소표본 조건에서 난이도 추정의 정확성을 개선할 수 있는가?

### 4.2 모델 수식

**LLM 사전 예측 획득:**

TOPIK 문항을 LLM에 제시하여 정답률 $\pi_i^{\text{LLM}}$을 예측:

```
다음은 TOPIK II(중급~고급) 읽기 문항입니다.
TOPIK 3급 학습자 100명이 이 문항을 풀 때 예상 정답률은 몇 %입니까?
95% 신뢰 구간도 제시하십시오.
[문항 텍스트]
```

로짓 척도 변환:

$$\mu_i^{\text{LLM}} = \log\frac{\pi_i^{\text{LLM}}}{1 - \pi_i^{\text{LLM}}}$$

$$(\sigma_i^{\text{LLM}})^2 = \left(\frac{\text{logit}(\pi_{i,97.5\%}) - \text{logit}(\pi_{i,2.5\%})}{2 \times 1.96}\right)^2$$

**문항 유형별 차등 사전분포:**

TOPIK 문항 유형(어휘·문법, 읽기 이해, 추론, 쓰기)에 따라 LLM 신뢰도 가중치 부여:

$$\sigma_i^{\text{adj}} = \frac{\sigma_i^{\text{LLM}}}{\omega_{\text{type}(i)}}, \quad \omega_{\text{type}} \in \{0.5, 0.8, 1.0, 1.5\}$$

여기서 $\omega$가 클수록 LLM 예측 불확실성이 크다고 판단.

**완전 Bayesian 모델:**

$$p(\boldsymbol{\beta}, \boldsymbol{\theta} \mid \mathbf{Y}) \propto \underbrace{\prod_{p,i} \sigma(\theta_p - \beta_i)^{y_{pi}}[1-\sigma(\theta_p-\beta_i)]^{1-y_{pi}}}_{\text{우도}} \cdot \underbrace{\prod_i \mathcal{N}(\beta_i; \mu_i^{\text{LLM}}, (\sigma_i^{\text{adj}})^2)}_{\text{LLM 사전분포}} \cdot \underbrace{\prod_p \mathcal{N}(\theta_p; 0, \sigma_\theta^2)}_{\text{능력 사전분포}}$$

**소표본 조건별 RMSE 비교:**

$$\text{RMSE}_P = \sqrt{\frac{1}{I}\sum_{i=1}^{I}(\hat{\beta}_i^{(P)} - \beta_i^{\text{ref}})^2}, \quad P \in \{30, 50, 100, 200, 500\}$$

여기서 $\beta_i^{\text{ref}}$는 대표본($P \geq 2000$) 기준 추정값.

### 4.3 한국어 특화 응용: 해외 소규모 시험 지역 지원

TOPIK은 전 세계 70개국 이상에서 시행되지만, 아프리카·중동·중앙아시아 일부 국가는 응시자가 매우 적다. LLM 사전분포를 활용하면 $P = 30$명의 소규모 지역에서도 신뢰할 수 있는 문항 보정이 가능해진다.

**교정 문항 세트 설계:**

$$\mathcal{D}_{\text{calib}} = \{(\text{item}_i, \hat{\beta}_i^{\text{ref}})\}_{i=1}^{I_c}, \quad I_c \geq 50$$

TOPIK 공개 회차에서 CTT 기반 난이도($p_i = \bar{Y}_i$)를 참조값으로 사용하여 LLM 예측 편향 보정:

$$\tilde{\mu}_i^{\text{LLM}} = a \cdot \mu_i^{\text{LLM}} + b, \quad \hat{a}, \hat{b} = \arg\min\sum_{i \in \mathcal{D}_{\text{calib}}}(\tilde{\mu}_i - \hat{\beta}_i^{\text{ref}})^2$$

### 4.4 공개 데이터 및 획득 방법

**즉시 사용 가능:**

| 데이터 | 규모 | 용도 | 접근 |
|-------|------|------|------|
| TOPIK 공개 기출 (35~100회차) | $I \approx 2400$문항 | 문항 텍스트 + LLM 예측 | www.topik.go.kr |
| 28회차 TOPIK IRT 연구 데이터 | 듣기 30 + 읽기 30문항 | 참조 난이도 $\hat{\beta}_i^{\text{ref}}$ | 28회 TOPIK 시행의 문항별 응답 자료가 연구에 활용된 바 있다. 해당 논문 저자 협력 요청 |
| CLIcK 벤치마크 | 1,995문항 | 문화·언어 문항 추가 | GitHub 공개 |

---

<a name="sec4"></a>
## 5. 연구 4: 학문목적한국어 문항 자동 생성 루프 (Active Learning)

### 5.1 연구 문제

학문목적한국어(KAP) 교육에서 필요한 특정 수준(예: TOPIK 4급 수준)의 학술 어휘·문법 문항을 LLM이 자동 생성하고, LLTM 피드백으로 품질을 검증하는 순환 시스템이 가능한가?

### 5.2 KAP 문항 설계 공간

학문목적한국어 문항의 속성 공간 $\mathcal{X}$:

$$\mathbf{x} = [x_{\text{담화유형}},\ x_{\text{학술어휘}},\ x_{\text{문법복잡도}},\ x_{\text{전공분야}},\ x_{\text{추론깊이}}]^{\top}$$

- $x_{\text{담화유형}} \in \{\text{정의}, \text{비교}, \text{인과}, \text{논증}, \text{요약}\}$
- $x_{\text{학술어휘}} \in [0, 1]$: 학술 기본어휘 목록(한국어 학술 기본어휘 5965어) 비율
- $x_{\text{전공분야}} \in \{\text{인문}, \text{사회}, \text{이공}, \text{예체능}\}$

**목표 난이도 달성을 위한 역방향 최적화:**

$$\mathbf{x}^* = \arg\min_{\mathbf{x} \in \mathcal{X}} \left(\mathbf{x}^{\top}\hat{\boldsymbol{\beta}}_k - \beta^*\right)^2 + \lambda_1 \|\mathbf{x}\|_1 + \lambda_2 \text{Diversity}(\mathbf{x}, \mathcal{I}_{\text{existing}})$$

여기서 $\text{Diversity}$ 항은 기존 문항 집합과의 내용 다양성 보장.

### 5.3 능동 학습 루프 설계

**초기화 ($t=0$):**

공개 TOPIK 기출문항 $I_0 = 80$개로 KAP-LLTM 초기 추정:

$$\hat{\boldsymbol{\beta}}_k^{(0)}, \hat{\sigma}_{k}^{(0)} \leftarrow \text{Stan MCMC}(\mathbf{Y}^{(0)}, \mathbf{X}^{(0)})$$

**루프 ($t \geq 1$):**

1. **문항 속성 선택:** 불확실성이 가장 큰 속성 조합 선택
   $$\mathbf{x}_{t}^* = \arg\max_{\mathbf{x}} \mathbb{V}\!\left[\mathbf{x}^{\top}\boldsymbol{\beta}_k \mid \mathbf{Y}^{(0:t-1)}\right]$$

2. **LLM 문항 생성:** KAP 특화 프롬프트
   ```
   다음 조건의 학문목적한국어 읽기 문항을 작성하십시오:
   - 담화 유형: [인과 관계]
   - 목표 학술 어휘 비율: [40%]
   - 추론 깊이: [3단계]
   - 전공: [사회과학]
   - 목표 TOPIK 수준: [4~5급]
   - 문항 형식: [4지선다 세부 내용 파악]
   ```

3. **품질 검증:** 한국어교육 전문가 + LLM judge
   $$\text{quality}(i^*) = \frac{1}{3}\left[\text{정확성} + \text{교육적 적합성} + \text{TOPIK 형식 준수}\right]$$

4. **현장 투입 및 응답 수집:** 온라인 KAP 교육 플랫폼에 문항 투입

5. **LLTM 업데이트:**
   $$p(\boldsymbol{\beta}_k \mid \mathbf{Y}^{(0:t)}) \propto p(\mathbf{Y}^{(t)} \mid \boldsymbol{\beta}_k) \cdot p(\boldsymbol{\beta}_k \mid \mathbf{Y}^{(0:t-1)})$$

**수렴 판단:**

$$\Delta_t = \left\|\hat{\boldsymbol{\beta}}_k^{(t)} - \hat{\boldsymbol{\beta}}_k^{(t-1)}\right\|_2 < 0.05 \implies \text{루프 종료}$$

### 5.4 공개 데이터 및 획득 방법

**즉시 사용 가능:**

- **한국어 학술 기본어휘 목록:** 국립국어원 발간 (5965어, 전공별 분류)
- **한국어 교육과정 문법 목록:** 국제통용 한국어 표준 교육과정 (2023 개정)
- **KoEdu 학습 플랫폼 데이터:** 세종학당재단 온라인 한국어 강좌 응답 데이터 (협력 신청)

**데이터 수집 전략:**

- 루프당 $n = 50$명 응답: 한국어 교육 기관 재학생 참여 보상 지급
- 루프 횟수 상한: $T_{\max} = 15$회 (총 $n \cdot T_{\max} = 750$명 참여)

---

<a name="sec5"></a>
## 6. 연구 5: 정오답 + 작문 텍스트 결합 모델 (Joint Model)

### 6.1 연구 문제

TOPIK II 쓰기 영역(51~54번)의 답안 텍스트는 해당 학습자의 한국어 숙달도에 대한 풍부한 정보를 담고 있다. 이 텍스트 우도를 TOPIK 읽기·듣기의 이진 응답 우도와 결합하면 $\theta_p$ 추정이 개선되는가?

### 6.2 모델 수식

**이진 응답 우도 (읽기·듣기):**

$$\log p(\mathbf{Y}_p^{\text{읽기}} \mid \theta_p) = \sum_{i=1}^{I_R} y_{pi}\log\sigma(\theta_p - \beta_i) + (1-y_{pi})\log[1-\sigma(\theta_p-\beta_i)]$$

**쓰기 답안 텍스트 우도:**

쓰기 문항 $i_w \in \{51, 52, 53, 54\}$에 대한 답안 $T_{p,i_w}$:

$$\log p(T_{p,i_w} \mid \theta_p, \beta_{i_w}) = \mathbf{e}_{p,i_w}^{\top}\mathbf{W}\boldsymbol{\xi}(\theta_p, \beta_{i_w}) - \log Z(\theta_p, \beta_{i_w})$$

여기서 $\mathbf{e}_{p,i_w} = \text{KoSBERT}(T_{p,i_w})$, $\boldsymbol{\xi}$는 $(\theta_p, \beta_{i_w})$의 비선형 변환.

**ELBO (변분 추론):**

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(\theta_p)}\!\left[\underbrace{\log p(\mathbf{Y}_p^{\text{읽기}} \mid \theta_p)}_{\text{객관식 우도}} + \underbrace{\sum_{i_w}\log p(T_{p,i_w} \mid \theta_p)}_{\text{쓰기 우도}}\right] - \text{KL}\!\left[q(\theta_p) \| p(\theta_p)\right]$$

변분 분포: $q(\theta_p) = \mathcal{N}(\mu_p^q, (\sigma_p^q)^2)$

**TOPIK 채점 기준 통합:**

TOPIK 쓰기 채점 기준(내용·조직·언어 사용)을 명시적으로 모델에 반영:

$$s_{p,i_w}^{(r)} = f_r(\mathbf{e}_{p,i_w}), \quad r \in \{\text{내용}, \text{조직}, \text{언어}\}$$

$$\log p(T_{p,i_w} \mid \theta_p) \approx \sum_r w_r \cdot g_r(s_{p,i_w}^{(r)}, \theta_p)$$

**텍스트 기여 정보량:**

$$\Delta I_p = \text{KL}\!\left[q(\theta_p \mid \mathbf{Y}_p, \mathbf{T}_p) \| q(\theta_p \mid \mathbf{Y}_p)\right] > 0 \implies \text{쓰기 답안이 숙달도 추정에 기여}$$

### 6.3 공개 데이터 및 획득 방법

#### 즉시 사용 가능한 공개 데이터

**TOPIK II 쓰기 모범 답안:**

- NIIED 공개 모범 답안 (일부 회차 공개)
- 수준별(3~6급) 예시 답안: 난이도 수준의 기준선

**국립국어원 학습자 말뭉치 (쓰기):**

- 작문 유형 말뭉치: 읽고 쓰기, 주제 쓰기 등 과제 유형별 분류
- 학습자 수준 레이블(TOPIK 급수) 포함 → $(\mathbf{Y}_p^{\text{읽기}}, T_{p,\text{작문}})$ 쌍 구성 가능

#### 연구자 직접 수집 방법

**방법: 통합 TOPIK 모의 시험 + 쓰기 답안 수집**

1. 참가자 $P \geq 300$명 모집 (TOPIK 준비생, 외국인 유학생)
2. TOPIK I/II 모의 시험 (객관식) + 쓰기 2문항 (53, 54번 유형)
3. 쓰기 답안을 전문 채점자 2인이 채점 (급간 신뢰도 $\kappa \geq 0.75$)
4. 개인 동의서 + IRB 승인

---

<a name="sec6"></a>
## 7. 연구 6: 언어권별 DIF 가설 생성 및 검증

### 7.1 연구 문제

TOPIK 문항 중 특정 언어권 학습자에게 불공정하게 작용하는 문항이 존재하는가? LLM은 문항 내용 분석만으로 어떤 문항이, 어떤 언어권에, 왜 불리한지를 사전에 예측할 수 있는가?

이 문제는 한국어교육에서 특히 중요하다. TOPIK은 단일 시험으로 중국어권, 일본어권, 영어권, 동남아어권 등 매우 이질적인 학습자 집단을 평가하기 때문이다.

### 7.2 한국어 특수 DIF 가설

언어권별 DIF가 예상되는 대표적 문항 속성:

| DIF 원인 | 불리 언어권 | 이유 | 문항 예시 |
|---------|-----------|------|---------|
| **한자어 비율 高** | 영어권·동남아 | 한자 기반 어휘 유추 불가 | 고급 학술 어휘 문항 |
| **일본어 유사 오답** | 일본어권 | 유사 형태 유인 오답 | 유사 어휘 변별 문항 |
| **한국 문화 배경** | 서양권 전반 | 문화 스키마 부재 | 한국 명절·예절 주제 |
| **경어법 판단** | SOV 비사용 언어권 | 격식성 판단 어려움 | 격식/비격식 변별 |
| **받침 발음·표기** | 성조어권(중·베트남) | L1 음운 전이 | 받침 맞춤법 문항 |

### 7.3 모델 수식

**언어권별 DIF Rasch 모델:**

$$\eta_{pi} = \theta_p - \beta_i - \delta_i^{(g)} \cdot \mathbb{1}[g_p = g], \quad g \in \{\text{중국어}, \text{일본어}, \text{영어}, \text{기타}\}$$

**다집단 동시 DIF 모델:**

$$\eta_{pi} = \theta_p - \beta_i - \sum_{g=1}^{G} \delta_{ig} \cdot \mathbb{1}[g_p = g], \quad \sum_{g} \delta_{ig} = 0 \text{ (식별 조건)}$$

**LLM DIF 가설 생성 프롬프트:**

```
다음 TOPIK 문항을 읽고, 아래 학습자 집단 중 어느 집단에게
동일한 한국어 능력에도 불구하고 이 문항이 더 어렵게 작용할
가능성이 있는지, 그 이유와 함께 평가하십시오.

집단: 중국어권 / 일본어권 / 영어권 / 베트남어권 / 기타

[문항 텍스트 및 선택지]

출력 형식:
{
  "집단": "중국어권",
  "DIF_방향": "유리" | "불리" | "없음",
  "확률": 0.0 ~ 1.0,
  "언어학적 이유": "..."
}
```

**LLM 가설의 Bayes Factor 검증:**

$$\text{BF}_h = \frac{p(\mathbf{Y} \mid M_{\text{DIF},h})}{p(\mathbf{Y} \mid M_{\text{no-DIF}})}$$

**LLM 예측 타당성 ROC 분석:**

$$\text{AUC} = P\!\left(\text{BF}_h > 3 \mid p_{\text{LLM},h} > 0.6\right)$$

### 7.4 공개 데이터 및 획득 방법

#### 즉시 사용 가능한 공개 데이터

**TOPIK 언어권별 통계:**

- NIIED 연차보고서: 국가·언어권별 응시자 수, 급수별 합격률 공개
- 합격률 차이 $\Delta_{g,\text{level}} = P(\text{합격} \mid g, \text{level})$를 DIF의 생태학적 지표로 활용

**PISA 한국 학생 데이터 (비교군):**

- PISA 국제 학업성취도 평가에 한국어 구사 외국어 학습자 포함
- 언어권별 학업 성취 패턴 → DIF 연구의 배경 지식

#### 연구자 직접 수집 방법

**방법 A: TOPIK 응시자 설문 연계**

- 온라인 TOPIK 모의 시험 플랫폼 이용자 모집
- 응시 후 언어권·모국어·학습 기간 설문 → 응답 행렬과 연계
- 목표: 언어권당 $P_g \geq 200$명, 주요 4개 언어권

**방법 B: 세종학당 글로벌 네트워크 활용**

- 세종학당재단 온라인 한국어 강좌 수강생 대상 모의 TOPIK 시행
- 84개국 세종학당 네트워크 → 다양한 언어권 자연 표본
- 협력 기관: 세종학당재단 연구 사업 공모 지원

---

<a name="sec7"></a>
## 8. 공개 데이터 및 데이터 획득 방법 종합

### 8.1 즉시 활용 가능한 공개 데이터셋

| # | 데이터셋 | 기관 | 규모 | 접근 방법 | 주요 활용 연구 |
|---|---------|------|------|---------|------------|
| 1 | **TOPIK 기출문항 PDF** | NIIED | 60회차+ (~2400문항) | www.topik.go.kr 무료 | 연구 1,3,4,6 |
| 2 | **한국어 학습자 말뭉치** | 국립국어원 | 1,588만 어절, 108개 언어권 | kcorpus.korean.go.kr (연구자 신청) | 연구 2,5 |
| 3 | **모두의 말뭉치** | 국립국어원 | 다종 (일상·문어·전문) | kli.korean.go.kr (신청) | 연구 1,4 |
| 4 | **CLIcK 벤치마크** | NAVER AI | 1,995 QA쌍, 11범주 | GitHub 공개 | 연구 1,3 |
| 5 | **AI Hub 외국인 한국어 음성** | NIA | 언어권별 수백 시간 | aihub.or.kr (승인 신청) | 연구 2 |
| 6 | **AI Hub 한국어 대화 데이터** | NIA | 다종 | aihub.or.kr (승인 신청) | 연구 2,5 |
| 7 | **28회 TOPIK IRT 연구 데이터** | 학술 논문 | 60문항 × P명 | 논문 저자 직접 요청 | 연구 1,3 |
| 8 | **KIIP 교재 QA** | 법무부/KIIP | 기본급 문항 | CLIcK에 일부 포함 | 연구 6 |
| 9 | **한국어 학술 기본어휘** | 국립국어원 | 5,965어 (전공별) | 국립국어원 발간 자료 | 연구 1,4 |
| 10 | **국제통용 한국어 문법 목록** | 국립국어원 | 수준별 문법 항목 | 표준 교육과정 문서 | 연구 1,4 |

### 8.2 연구자 직접 구축이 필요한 데이터

| 데이터 | 구축 방법 | 예상 비용 | 소요 기간 | 해당 연구 |
|-------|---------|--------|--------|---------|
| **TOPIK 모의 응답 행렬** ($P \geq 1000$) | 온라인 플랫폼 협력 or 크라우드소싱 | 500~1000만 원 | 3~6개월 | 1,3,6 |
| **KAP 학습자 작문** ($P \geq 300$) | 대학 언어교육원 협력 MOU | 100~200만 원 | 3개월 | 2,5 |
| **사고 발화 전사** ($P \geq 100$) | 실험실 프로토콜 (음성+전사) | 300~500만 원 | 6개월 | 2 |
| **언어권별 응답 행렬** ($P_g \geq 200$) | 세종학당 네트워크 협력 | 200~400만 원 | 6~12개월 | 6 |
| **KAP 문항 전문가 코딩** ($I \geq 100$) | 한국어교육 전문가 패널 (3인) | 100~200만 원 | 2개월 | 1,4 |

### 8.3 데이터 획득 신청 절차 상세

**국립국어원 한국어 학습자 말뭉치 신청:**

```
1. kcorpus.korean.go.kr 접속 → 회원 가입
2. 연구 계획서 제출 (연구 목적, 활용 방법, 보안 서약)
3. 심의 위원회 검토 (약 2~4주)
4. 승인 후 FTP 다운로드 또는 나눔터 직접 다운로드
5. 저작권: 국립국어원 출처 명시 필수, 원문 재배포 금지
```

**AI Hub 데이터 신청:**

```
1. aihub.or.kr 회원 가입 (개인/기관)
2. 원하는 데이터셋 페이지 → [다운로드 신청] 클릭
3. 활용 계획서 + 개인정보 보호 서약서 제출
4. 승인 후 API 또는 분할 압축 파일 다운로드
5. 비공개 의료 데이터는 K-ICT 빅데이터센터 오프라인 안심구역 이용
```

---

<a name="sec8"></a>
## 9. 연구 로드맵 및 우선순위

### 9.1 단계별 실행 계획

```
Phase 1 (0~6개월): 기반 구축
├── TOPIK 기출문항 PDF 수집 및 파싱
├── 형태소 분석 파이프라인 구축 (Kiwi + KoNLPy)
├── LLM 속성 추출 프롬프트 개발 및 평가
├── 국립국어원 말뭉치 신청 및 수령
└── 연구 1 파일럿 (I=60문항, TOPIK 기출)

Phase 2 (6~12개월): 핵심 연구 수행
├── 연구 1: TOPIK LLM-LLTM 본 연구 (I=200, P=500 목표)
├── 연구 3: 소표본 사전분포 시뮬레이션 + TOPIK 파일럿
├── 연구 6: 언어권 DIF 가설 생성 (LLM) + NIIED 협력 협상
└── 응답 데이터 수집 프로토콜 확정 (IRB 승인)

Phase 3 (12~24개월): 확장 연구
├── 연구 2: KAP 학습자 작문 수집 + 임베딩 LR-Rasch
├── 연구 4: KAP 문항 자동 생성 루프 (Active Learning)
├── 연구 6: 세종학당 네트워크 DIF 데이터 수집
└── 연구 5: 결합 모델 파일럿 (ELBO 근사)

Phase 4 (24~36개월): 통합 및 검증
├── 6개 연구 결과 통합 모형 개발
├── TOPIK 문항 개발 지원 시스템 프로토타입
└── 성과 발표 및 NIIED 정책 제언
```

### 9.2 연구별 우선순위 종합 매트릭스

| 연구 | 데이터 가용성 | 방법론 성숙도 | 한국어교육 기여도 | 우선순위 |
|-----|------------|------------|----------------|--------|
| **연구 1** (LLM-LLTM) | ★★★★★ | ★★★★☆ | ★★★★★ | **1순위** |
| **연구 3** (생성 사전분포) | ★★★★☆ | ★★★★★ | ★★★★☆ | **2순위** |
| **연구 6** (DIF) | ★★★☆☆ | ★★★★☆ | ★★★★★ | **3순위** |
| **연구 2** (임베딩 공변량) | ★★★☆☆ | ★★★★☆ | ★★★★☆ | **4순위** |
| **연구 4** (생성 루프) | ★★★☆☆ | ★★★☆☆ | ★★★★★ | **5순위** |
| **연구 5** (결합 모델) | ★★☆☆☆ | ★★☆☆☆ | ★★★★☆ | **6순위** |

### 9.3 핵심 인프라 요구사항

**소프트웨어 스택:**

```python
# 한국어 형태소 분석
kiwipiepy >= 0.17    # Kiwi 형태소 분석기
konlpy >= 0.6        # 다중 형태소 분석기 래퍼

# 한국어 임베딩
sentence-transformers  # KoSBERT 모델 포함
# 모델: jhgan/ko-sroberta-multitask (HuggingFace)

# Bayesian 추정
cmdstanpy >= 1.2     # Stan MCMC
pymc >= 5.0          # 대안적 Bayesian 프레임워크

# LLM API
openai               # GPT-4o
anthropic            # Claude 3.5 Sonnet
```

**필요 컴퓨팅:**
- CPU: 16코어 이상 (Stan MCMC 병렬 체인)
- GPU: NVIDIA A100 또는 동급 (KoSBERT 임베딩 대량 처리)
- 스토리지: 1TB 이상 (말뭉치 + 임베딩 캐시)

---

## 참고문헌

- Wilson, M., De Boeck, P., & Carstensen, C. H. (2008). Explanatory item response models: A brief introduction. *Assessment of Competencies in Educational Contexts*, 83–110.
- Cho, S.-J., & Cohen, A. S. (2010). A multilevel mixture IRT model with an application to DIF. *Journal of Educational and Behavioral Statistics*, 35(3), 336–370.
- Kim, H. S., & Lee, S. (2019). Item analysis of Test of Proficiency in Korean: Classical test theory and item response theory. *The Korean Language in America*, 23(1), 1–27.
- 국립국어원 (2023). *국제 통용 한국어 표준 교육과정*. 문화체육관광부.
- 국립국어원 (2025). 한국어 학습자 말뭉치 나눔터. https://kcorpus.korean.go.kr
- Lee, J., et al. (2024). CLIcK: A benchmark dataset of cultural and linguistic intelligence in Korean. *arXiv:2403.06412*.
- Kim, J., et al. (2024). Validation of an elicited imitation test as a measure of Korean language proficiency. *Language Testing in Asia*, 14(1).
- NIIED (2024). *TOPIK 시행 결과 연차 보고서*. 국립국제교육원.
