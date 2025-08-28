# 데이터로 증명하는 우리 지역다움
  

## 디렉토리 구조
```
analyzing/            # 분석 코드/노트
data_collecting/      # 수집 스크립트/원문 백업
reporting/
  ├─ indicator_calculator.py  # EPI/U/C/G 산출
  └─ report_generator.py      # GPT로 슬로건/코멘트 생성 + MD/CSV 리포트
data/                 # 입력 CSV들
results/              # 모든 산출물 저장
main.py               # 전체 파이프라인 실행 진입점
```
  
## Quickstart
### 1. 가상환경 + 패키지 설치

Windows (PowerShell)
```
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux (bash/zsh)
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
  
### 2. 실행
```
python main.py
```

  
## 파이프라인 설명
### 1. 데이터 수집 (data_collecting/)

- 한국관광공사/지자체 오픈API 및 CSV 다운로드 기반 수집

- 호출 제한 등으로 일부 데이터는 CSV로 저장하여 활용

- 주요 수집 항목:

  - 관광지/콘텐츠 메타 (콘텐츠 분류체계 포함)

  - 전국문화축제 표준데이터 (기간 필터/일수 집계 등)

  - 블로그 스니펫 크롤링 결과
  

### 2. 분석 (analyzing/)

- TF-IDF 상위 키워드: 지역별 블로그 스니펫 → 정규식(한글 2자↑), 불용어 제거 → TfidfVectorizer → 상위 키워드/점수

- 빈도 기반 키워드: 단순 카운팅(보조)

- LDA 토픽 모델링: 지역별 문서 묶음 → CountVectorizer → LDA → 토픽별 상위 단어들(term, beta)

- 고유성 분석: regional_uniqueness_analysis.csv에 uniqueness_composite 등 포함
  

### 3. 지표 산출 (reporting/indicator_calculator.py)

- 입력: data/의 세 CSV
- 출력: results/의 4개 CSV (타임스탬프 없이 고정 저장)
  - EPI (외부 인식 지수)
    - region별 score_tfidf 평균 → MinMax(0~1) → EPI_raw = 0.8 * blog + 0.2 * 0.5
    - 다시 MinMax(0~100) → EPI
  - U (차별화 지수)
    - regional_uniqueness_analysis.csv의 uniqueness_composite를 그대로 사용 → U
  - C (정합성 지수)
    - 블로그 TF-IDF 상위 TOPK_FOR_C (기본 50) 키워드의 확률분포 vs
    - LDA 상위 TOPK_FOR_C 토픽단어 확률분포
    - 두 분포의 코사인 유사도 (0~1)
  - G (괴리도)
    - gov_slogans.csv의 현재 행정 슬로건 vs 사용자 주제(블로그+LDA)
    - 단어 수준 Jaccard와 문자 n-gram TF-IDF 코사인 중 최댓값을 유사도로 보고, G = 1 - 유사도 (낮을수록 좋음)
  

### 4. 리포트 생성 (reporting/report_generator.py)

- 입력: results/regional_branding_scores.csv (+선택 gov_slogans.csv)
- 출력: 마크다운과 시각화용 CSV (타임스탬프 포함)
  - 새 슬로건 자동 제안(GPT)
    - 리포트에서는 **각 지역의 새로운 ‘제안 슬로건’**을 GPT로 1개 생성(한국어, 10자 이내)
    - G는 현재 슬로건과의 괴리를 측정하는 지표로서 계속 유지됨
  - 지역별 코멘트(GPT)
    - 사분위 기반 규칙 요약(EPI/U/C/G의 상대 위치)을 힌트로, 정책 보고서 톤의 2~3문장 코멘트 생성
  - 산출물
    - results/regional_branding_report_YYYYMMDD_HHMMSS.md
    - results/regional_branding_insights_YYYYMMDD_HHMMSS.csv

  
## 데이터 스키마 요약

`TF-IDF_top_keywords.csv`
| column       | type  | 설명        |
| ------------ | ----- | --------- |
| region       | str   | 지역명       |
| rank         | int   | 키워드 랭크    |
| keyword      | str   | 키워드(표면형)  |
| score\_tfidf | float | TF-IDF 점수 |

  
`region_lda_topics_summary.csv`
| column | type  | 설명             |
| ------ | ----- | -------------- |
| region | str   | 지역명            |
| topic  | int   | 토픽 번호          |
| rank   | int   | 토픽 내 단어 랭크     |
| term   | str   | 토픽의 상위 단어      |
| beta   | float | 단어 가중치(확률/비례값) |

  
`regional_uniqueness_analysis.csv`
| column                | type  | 설명                      |
| --------------------- | ----- | ----------------------- |
| region                | str   | 지역명                     |
| uniqueness\_composite | float | 최종 고유성 점수(LOF/희귀도/네트워크) |

  
`gov_slogans.csv`
| column | type | 설명            |
| ------ | ---- | ------------- |
| region | str  | 지역명           |
| slogan | str  | **현재** 행정 슬로건 |
  

## 팀원(Contributors)
- 오수민: 데이터 분석 및 시각화
- 윤서영: 데이터 수집 및 분석
- 김동희: 기획 및 데이터 분석
- 설민하: 기획