# KOS-5101

- Jupyter Notebooks

## Progress in 2026
#### Week 1
1. Installation: python, vscode, github, copilot
2. python in a day: `https://wikidocs.net/book/1`
#### Week 2
- Unicode
- CLT (Central Limit Theorem)
- Concepts of z-Test, t-Test and its real meaning.
- Random Sampling, Histogram/KDE
- PDF, PMF: Gaussian, Beta, Bernoulli, Gamma, Uniform

#### Week 3
- 논문 발제 : Usage of statistical analysis in research papers
- CLT, simulation t-Test
- Bayesian Inference by posterior simulation

#### Week 4
- 


### CoinToss
이 주피터노트북은 베이즈통계해석과 빈도수통계해석의 차이를 보여주기 위해 제작한다.

1. 동전을 11번 던져서 3번 앞면이 나온 경우의 샘플 데이터를 만들어라.
2. 빈도수통계해석 방법으로 이 동전이 정상동전인지 테스트하라. 즉 Prob[X=앞면] = 0.5 를 Null Hypothesis로 설정하고 통계적 테스트를 적용하라. 테스트 결과의 의미를 해설하라.
3. 베이즈통계해석 방법으로 이 동전의 앞면 나오는 확률 \theta 에 대한 해석을 하라.  prior(\theta) = Beta(2,2) 로 설정하라.
4. conjugate pair 에 대한 closed form 의 posterior distribution 을 찾아라. posterior distribution 의 pdf 를 그림으로 보여라.
5. Stan/cmdstanpy 를 이용하여 데이터를 입력하고 posterior sample을 구하는 방법을 적용하라. 
- 베이즈 해석 이전에 windows OS를 기준으로 stan/cmdstanpy 를 설치하는 루틴을 포함하는 셀을 작성하라.
- posterior sample 들의 empirical distribution (histogram/KDE)와 공식을 적용하여 구한 posterior pdf를 하나의 figure에 그려서 두 개의 결과가 실질적으로 동일함을 보여라.


### Z-Test vs Bayesian
주피터노트북 파일 'Bayesian_vs_Frequentist_ZTest.ipynb'를 만들어라.
이 주피터노트북은 베이즈통계해석과 빈도수통계해석의 차이를 보여주기 위해 제작한다.

- figure에 한글이 잘 표시되도록 설정하라.
- 수식은 latex 을 사용하라.


문제: 대한민국 중학교 1학년 학생의 평균키는 159cm로 알려져 있다. 서울지역 중학생 1학년 학생 50명을 대상으로 조사한 결과 표본평균은 160, 표본분산은 6으로 나타났다. 서울지역 중학교 1학년 학생의 평균키와 대한민국 중학교 1학년 학생의 평균키 차이가 있는지 통계분석을 시행하라.

1. 빈도수통계해석 방법으로 가설검정을 진행하라.

2. 베이즈통계해석 방법으로 가설검정을 진행하라. conjugate pair 공식을 적용하라.
   
- windows OS를 기준으로 stan/cmdstanpy 를 설치하는 루틴을 포함하는 셀을 작성하라.

3. Stan/cmdstanpy 를 이용하여 데이터를 입력하고 posterior sample을 구하는 방법을 적용하라. 
- posterior sample 들의 empirical distribution (histogram/KDE)와 공식을 적용하여 구한 posterior pdf를 하나의 figure에 그려서 두 개의 결과가 실질적으로 동일함을 보여라.
- posterior sample을 사용한 가설검정을 진행하라. 
- Bayesian inference 를 시행하라.


### T-Test vs Bayesian

주피터 노트북 파일 'Bayesian_vs_Frequentist_t_Test.ipynb' 파일을 생성하라.
이 주피터노트북은 베이즈통계해석과 빈도수통계해석의 차이를 보여주기 위해 제작한다.

- figure에 한글이 잘 표시되도록 설정하라.
- 수식은 latex 을 사용하라. latex 심볼을 정의할 때는 '\' 를 사용하라. '\\'를 사용할 필요없다.
  
기본 설정: 다음은 컴퓨터 관련 기업 24개의 자기자본이익률 (ROE: Return On Equity) 데이터이다. 단위 %.
22.4 32.4 21.2 36.8 42.2 16.4 15.5 38.8 24.9 26.6 28.5 25.0 21.8 26.5 10.1 14.1 12.5 14.6 30.5 13.0 41.3 40.4 25.5 35.8

다음은 건설 관련 기업 17개의 ROE 데이터이다: 15.7 14.1 12.3 12.1 14.7 10.3 14.3 21.4 15.0 40.8 11.5 13.7 16.9 11.0 8.7 11.8 8.2

문제:
0. t-Test  에 대한 이론적 배경을 설명하시오.
1. 두 산업의 ROW 평균을 비교하고 싶어하는 이유를 설명하시오.
2. 두 산업에 대한 ROE 평균에 대한 차이가 있는지를 분산이 같다는 가정하에 유의수준 1%에서 검정하시오.
3. 두 산업의 ROE 데이터에 대한 등분산성(equal vairance) 가정을 검정하여라.
4. 두 산업에 대한 ROE 평균에 대한 차이가 있는지를 분산이 같다는 가정 없이 유의수준 1%에서 검정하시오.
5. 위에 수행한 빈도수통계분석 문제에 대해서, Stan/cmdstanpy 를 사용하여 Bayesian Inference 를 수행하시오.
