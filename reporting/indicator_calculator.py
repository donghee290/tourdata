import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BLOG_KEYWORDS_CSV = DATA_DIR / "TF-IDF_top_keywords.csv"        # region, rank, keyword, score_tfidf
LDA_TOPICS_CSV     = DATA_DIR / "region_lda_topics_summary.csv" # region, topic, rank, term, beta
UNIQUENESS_CSV     = DATA_DIR / "regional_uniqueness_analysis.csv"
GOV_SLOGANS_CSV    = DATA_DIR / "gov_slogans.csv"               # region, slogan

TOPK_FOR_C = 50
SENTIMENT_CONST = 0.5

KOREAN_PATTERN = re.compile(r"[가-힣]{2,}")
STOPWORDS_KO = set("""
하다 되다 이다 있다 없다 아니다 그리고 그러나 그래서 또는 또한 매우 너무 더 좀 같이
하면 해서 한 후 전 후에 위해 위한 대한 등의 등 으로 으로서 으로써 로서 로써 에서 에
부터 까지 도 는 은 이 가 을 를 의 에 게 와 과 로 는데 다가 보다 보다도
저희 우리 제가 내가 당신 여러분 오늘 어제 내일 지금 현재 경우 정말 진짜 바로 그냥 모두 모든
사진 블로그 리뷰 방문 위치 제공 이용 확인 안내 소개 정보 메뉴 지도 링크 기사 뉴스
주차 예약 전화 번호 운영 시간 영업 운영시간 가격 비용 할인 이벤트 행사 추천 인기 최신 베스트
근처 주변 인근 근교 가까운 가까이에
""".split())

def tokenize_ko(text_in: str):
    toks = KOREAN_PATTERN.findall(str(text_in))
    return [t for t in toks if len(t) >= 2 and t not in STOPWORDS_KO]

def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def jaccard(a, b):
    A, B = set(a), set(b)
    if not (A or B):
        return 1.0
    return len(A & B) / len(A | B) if (A | B) else 0.0

def cosine_safe(v1, v2):
    if not v1 and not v2:
        return 0.0
    if len(v1) != len(v2):
        return 0.0
    try:
        return float(cosine_similarity([v1], [v2])[0][0])
    except Exception:
        return 0.0

# -------------------------
# 0. 데이터 로드
# -------------------------

def load_inputs():
    blog = pd.read_csv(BLOG_KEYWORDS_CSV)
    lda  = pd.read_csv(LDA_TOPICS_CSV)
    uniq = pd.read_csv(UNIQUENESS_CSV)

    gov = None
    if GOV_SLOGANS_CSV.exists():
        gov = pd.read_csv(GOV_SLOGANS_CSV)
        if "slogan" not in gov.columns:
            raise ValueError("gov_slogans.csv는 반드시 'slogan' 컬럼이 있어야 합니다.")
    return blog, lda, uniq, gov

# -------------------------
# 1. EPI 산출
# -------------------------
def compute_epi(blog_keywords: pd.DataFrame) -> pd.DataFrame:
    s = blog_keywords.groupby("region")["score_tfidf"].mean().reset_index()
    s.rename(columns={"score_tfidf": "blog_topic_strength"}, inplace=True)

    mm = MinMaxScaler()
    s["blog_topic_strength_scaled"] = mm.fit_transform(s[["blog_topic_strength"]])

    s["EPI_raw"] = 0.8 * s["blog_topic_strength_scaled"] + 0.2 * SENTIMENT_CONST
    s["EPI"] = MinMaxScaler(feature_range=(0, 100)).fit_transform(s[["EPI_raw"]])
    return s[["region", "EPI"]]

# -------------------------
# 2. U 산출 (차별화지수)
# -------------------------
def compute_u(uniq: pd.DataFrame) -> pd.DataFrame:
    out = uniq[["region", "uniqueness_composite"]].copy()
    out.rename(columns={"uniqueness_composite": "U"}, inplace=True)
    return out

# -------------------------
# 3. C 산출 (정합성지수)
# -------------------------
def compute_c(blog_keywords: pd.DataFrame, lda_topics: pd.DataFrame) -> pd.DataFrame:
    blog_sorted = blog_keywords.sort_values(["region", "score_tfidf"], ascending=[True, False])
    blog_topk = blog_sorted.groupby("region").head(TOPK_FOR_C)
    blog_vecs = {}
    for region, sub in blog_topk.groupby("region"):
        kv = sub.set_index("keyword")["score_tfidf"].to_dict()
        tot = sum(kv.values()) or 1.0
        blog_vecs[region] = {k: v / tot for k, v in kv.items()}

    lda_agg = lda_topics.groupby(["region", "term"])["beta"].sum().reset_index()
    lda_sorted = lda_agg.sort_values(["region", "beta"], ascending=[True, False])
    lda_topk = lda_sorted.groupby("region").head(TOPK_FOR_C)
    lda_vecs = {}
    for region, sub in lda_topk.groupby("region"):
        kv = sub.set_index("term")["beta"].to_dict()
        tot = sum(kv.values()) or 1.0
        lda_vecs[region] = {k: v / tot for k, v in kv.items()}

    rows = []
    regions = sorted(set(blog_vecs.keys()) & set(lda_vecs.keys()))
    for r in regions:
        b, l = blog_vecs[r], lda_vecs[r]
        vocab = sorted(set(b.keys()) | set(l.keys()))
        bv = [b.get(t, 0.0) for t in vocab]
        lv = [l.get(t, 0.0) for t in vocab]
        c = cosine_safe(bv, lv)
        rows.append({"region": r, "C": c})
    return pd.DataFrame(rows)

# -------------------------
# 4. G 산출 (괴리도)
# -------------------------
def jaccard_word(a_tokens, b_tokens):
    A, B = set(a_tokens), set(b_tokens)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def compute_g(blog_keywords: pd.DataFrame, gov: pd.DataFrame | None) -> pd.DataFrame:
    """
    G = 1 - max( 단어 Jaccard, 문자 n-gram TF-IDF 코사인 )
    - 단어 Jaccard: 동일 표면형이 겹칠 때 유사도↑ (보수적)
    - char TF-IDF cos: 띄어쓰기/형태가 달라도(예: '행복영도' vs '행복 영도') 유사도 반영
    """
    if gov is None or gov.empty:
        return pd.DataFrame(columns=["region", "G"])

    # 1) 사용자 토픽 텍스트(지역별): 블로그 TF-IDF 상위 + LDA 상위 term
    topk = TOPK_FOR_C
    blog_sorted = blog_keywords.sort_values(["region", "score_tfidf"], ascending=[True, False])
    blog_topk = blog_sorted.groupby("region").head(topk)
    user_kw = blog_topk.groupby("region")["keyword"].apply(lambda s: list(dict.fromkeys(s))).to_dict()

    # LDA term 상위 합치기(있으면 보강)
    try:
        lda = pd.read_csv(LDA_TOPICS_CSV)
        lda_agg = lda.groupby(["region","term"])["beta"].sum().reset_index()
        lda_top = lda_agg.sort_values(["region","beta"], ascending=[True, False]).groupby("region").head(topk)
        lda_terms = lda_top.groupby("region")["term"].apply(lambda s: list(dict.fromkeys(s))).to_dict()
    except Exception:
        lda_terms = {}

    # 지역별 사용자 토큰/텍스트 만들기
    user_tokens_by_region = {}
    user_text_by_region = {}
    regions_all = set(user_kw.keys()) | set(lda_terms.keys())
    for r in regions_all:
        toks = []
        if r in user_kw:   toks.extend(map(str, user_kw[r]))
        if r in lda_terms: toks.extend(map(str, lda_terms[r]))
        # 한글 2자 이상만 + 불용어 제거 + 중복 제거
        toks = [t for t in toks for t in tokenize_ko(t)]
        toks = list(dict.fromkeys(toks))
        user_tokens_by_region[r] = toks
        user_text_by_region[r] = " ".join(toks)

    # 2) 슬로건 토큰/텍스트 준비
    gov_map = {}
    for _, row in gov.iterrows():
        r = str(row["region"]).strip()
        s = str(row["slogan"])
        tok = tokenize_ko(s)
        gov_map[r] = {
            "tokens": tok,
            "text": " ".join(tok)
        }

    # char TF-IDF 코사인
    def char_cosine(a_text: str, b_text: str) -> float:
        if not a_text and not b_text:
            return 1.0
        if not a_text or not b_text:
            return 0.0
        vect = TfidfVectorizer(analyzer="char", ngram_range=(2,3), min_df=1)
        X = vect.fit_transform([a_text, b_text])
        return float(cosine_similarity(X[0], X[1])[0, 0])

    # 3) 지역별 유사도 → G 산출
    rows = []
    regions = sorted(set(gov_map.keys()) & set(user_text_by_region.keys()))
    for r in regions:
        gov_tok = gov_map[r]["tokens"]
        gov_txt = gov_map[r]["text"]
        user_tok = user_tokens_by_region.get(r, [])
        user_txt = user_text_by_region.get(r, "")

        sim_word = jaccard_word(gov_tok, user_tok)   # 0~1
        sim_char = char_cosine(gov_txt, user_txt)    # 0~1
        sim = max(sim_word, sim_char)

        G = 1.0 - sim
        rows.append({"region": r, "G": G})

    # 슬로건만 있고 사용자 텍스트가 전혀 없는 지역은 비교불가 → G=1.0
    for r in set(gov_map.keys()) - set(regions):
        rows.append({"region": r, "G": 1.0})

    return pd.DataFrame(rows)

# -------------------------
# 메인
# -------------------------
def main():
    blog, lda, uniq, gov = load_inputs()
    ts = now_tag()

    epi_df = compute_epi(blog)
    u_df   = compute_u(uniq)
    c_df   = compute_c(blog, lda)
    g_df   = compute_g(blog, gov)

    final = epi_df.merge(u_df, on="region", how="left") \
                  .merge(c_df, on="region", how="left") \
                  .merge(g_df, on="region", how="left")

    # 일단 타임스탬프 제거함
    (RESULTS_DIR / "regional_branding_scores.csv").write_text(
        final.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8"
    )
    (RESULTS_DIR / "epi.csv").write_text(
        epi_df.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8"
    )
    (RESULTS_DIR / "consistency.csv").write_text(
        c_df.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8"
    )
    (RESULTS_DIR / "gap.csv").write_text(
        g_df.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8"
    )

    print(f"[완료] 지표 산출 → {RESULTS_DIR / 'regional_branding_scores.csv'}")
    print(final.head())
    return str(RESULTS_DIR / 'regional_branding_scores.csv')

if __name__ == "__main__":
    main()