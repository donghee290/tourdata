import os
import re
import json
import math
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

INPUT_CSV = r"C:/Users/ohsum/OneDrive/바탕 화면/취업준비 및 포트폴리오/공모전/2025 한국관광 데이터랩 활용 경진대회/분석 파트 데이터/네이버 블로그 크롤링.csv"  # <- 사용자 경로
OUTPUT_DIR = r"./lda_region_outputs"        # 결과 저장 폴더
RESULTS_SUMMARY = "region_lda_topics_summary.csv"  # 모든 지역 결과를 합친 요약 파일명

# 벡터라이저/토픽 모델 파라미터
USE_BIGRAMS = True       # 바이그램 포함(토픽 해석력 ↑)
MAX_FEATURES = 30000     # 지역별 단어 수 상한
MIN_DF = 2               # 지역 내 너무 희귀한 단어 무시(문서빈도)
MAX_DF = 0.95            # 지역 내 너무 흔한 단어 무시(문서비율)
MAX_ITER = 20            # LDA 학습 반복 수
RANDOM_STATE = 42

TOPIC_RULE = {
    # 문서 수(documents) 기준 구간 → 토픽 수
    # 예: 0~99개 3토픽, 100~299개 4토픽, 300~999개 5토픽, 1000개 이상 6토픽
    "bins": [0, 100, 300, 1000, math.inf],
    "topics": [3, 4, 5, 6]
}

# 지역별 최소 문서 수(이 미만이면 스킵)
MIN_DOCS_PER_REGION = 30

# 토픽별 상위 단어 개수
N_TOP_WORDS = 15


# =========================
# 2) 간단 한국어 토크나이저 & 불용어
# =========================
KOREAN_PATTERN = re.compile(r"[가-힣]{2,}")

STOPWORDS_KO = set("""
하다 되다 이다 있다 없다 아니다 그리고 그러나 그래서 또는 또한 매우 너무 더 좀 같이
하면 해서 한 후 전 후에 위해 위한 대한 등의 등 으로 으로서 으로써 로서 로써 에서 에
부터 까지 도 는 은 이 가 을 를 의 에 게 와 과 로 는데 다가 보다 보다도
저희 우리 제가 내가 당신 여러분 오늘 어제 내일 지금 현재 경우 정말 진짜 바로 그냥 모두 모든
사진 블로그 리뷰 방문 위치 제공 이용 확인 안내 소개 정보 메뉴 지도 링크 클릭 기사 뉴스
주차 예약 전화 번호 운영 시간 영업 운영시간 가격 비용 할인 이벤트 행사 추천 인기 최신 베스트
근처 주변 인근 근교 가까운 가까이에
""".split())

def tokenize_ko(text_in: str) -> List[str]:
    toks = KOREAN_PATTERN.findall(str(text_in))
    return [t for t in toks if len(t) >= 2 and t not in STOPWORDS_KO]


def build_vectorizer():
    ngram = (1, 2) if USE_BIGRAMS else (1, 1)
    return CountVectorizer(
        tokenizer=tokenize_ko,
        preprocessor=lambda x: x,
        token_pattern=None,
        lowercase=False,
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=ngram
    )


def safe_filename(name: str) -> str:
    # OS 호환 안전 파일명
    return "".join([c if c.isalnum() or c in (" ", ".", "-", "_") else "_" for c in str(name)]).strip("_ ")


def choose_topics_by_docs(n_docs: int, bins: List[float], topics: List[int]) -> int:
    for b, t in zip(bins, topics):
        if n_docs < b:
            return t
    return topics[-1]


# =========================
# 3) 데이터 로드/전처리
# =========================
def load_work(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # 컬럼명 확정: region, snippet 필요
    assert "region" in df.columns and "snippet" in df.columns, "CSV에 'region'과 'snippet' 컬럼이 필요합니다."

    work = df[["region", "snippet"]].astype(str)
    work["region"] = work["region"].str.strip()
    work["snippet"] = work["snippet"].str.replace(r"\s+", " ", regex=True).str.strip()
    work = work[(work["region"] != "") & (work["snippet"] != "")]
    return work


# =========================
# 4) 메인
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    work = load_work(INPUT_CSV)

    # 지역 목록 및 문서 수 체크
    region_groups = work.groupby("region")
    regions = list(region_groups.groups.keys())

    # 요약 결과 누적
    summary_rows = []

    print(f"[INFO] 총 지역 수: {len(regions)}")
    for region in tqdm(regions, desc="지역별 LDA 진행"):
        sub = region_groups.get_group(region)

        n_docs = len(sub)
        if n_docs < MIN_DOCS_PER_REGION:
            print(f"[SKIP] {region}: 문서 수 {n_docs} < {MIN_DOCS_PER_REGION}")
            continue

        # 벡터화
        vect = build_vectorizer()
        X = vect.fit_transform(sub["snippet"])
        vocab = np.array(vect.get_feature_names_out())

        # 토픽 수 자동 결정
        n_topics = None
        for b, t in zip(TOPIC_RULE["bins"], TOPIC_RULE["topics"]):
            if n_docs < b:
                n_topics = t
                break
        if n_topics is None:
            n_topics = TOPIC_RULE["topics"][-1]

        # LDA 학습
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=RANDOM_STATE,
            learning_method="batch",
            max_iter=MAX_ITER,
            n_jobs=-1
        ).fit(X)

        # 토픽별 상위 단어 추출
        rows = []
        for k, comp in enumerate(lda.components_):  # shape: (K, V)
            top_idx = comp.argsort()[::-1][:N_TOP_WORDS]
            for rank, j in enumerate(top_idx, start=1):
                rows.append({
                    "region": region,
                    "n_docs": n_docs,
                    "topic": k,
                    "rank": rank,
                    "term": vocab[j],
                    "beta": float(comp[j])
                })
                # 요약에도 같은 포맷으로 누적
                summary_rows.append({
                    "region": region,
                    "n_docs": n_docs,
                    "topic": k,
                    "rank": rank,
                    "term": vocab[j],
                    "beta": float(comp[j])
                })

        # 지역별 파일 저장
        out_df = pd.DataFrame(rows).sort_values(["topic", "rank"])
        fname = f"topic_top_words_{safe_filename(region)}.csv"
        out_path = os.path.join(OUTPUT_DIR, fname)
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        # 메타정보 JSON도 덤으로 저장(나중에 재현/추적에 도움)
        meta = {
            "region": region,
            "n_docs": n_docs,
            "n_topics": n_topics,
            "params": {
                "USE_BIGRAMS": USE_BIGRAMS,
                "MAX_FEATURES": MAX_FEATURES,
                "MIN_DF": MIN_DF,
                "MAX_DF": MAX_DF,
                "MAX_ITER": MAX_ITER,
                "RANDOM_STATE": RANDOM_STATE,
                "N_TOP_WORDS": N_TOP_WORDS
            }
        }
        with open(os.path.join(OUTPUT_DIR, f"meta_{safe_filename(region)}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # 전체 요약 저장
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(["region", "topic", "rank"])
        summary_df.to_csv(os.path.join(OUTPUT_DIR, RESULTS_SUMMARY), index=False, encoding="utf-8-sig")
        print(f"[DONE] 요약 저장: {os.path.join(OUTPUT_DIR, RESULTS_SUMMARY)}")
    else:
        print("[WARN] 저장할 요약 결과가 없습니다(모든 지역이 스킵되었을 수 있음).")


if __name__ == "__main__":
    main()