import os
import re
import math
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SCORES_PATH = RESULTS_DIR / "regional_branding_scores.csv"
DEFAULT_GOV_SLOGANS = DATA_DIR / "gov_slogans.csv"

OPENAI_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.3
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

QUANTILES = [0.25, 0.5, 0.75]

# 유틸
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_env():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 가 .env 또는 환경변수에 설정되어 있지 않습니다.")
    return api_key

def find_scores(scores_csv: Optional[str]) -> Path:
    p = Path(scores_csv) if scores_csv else DEFAULT_SCORES_PATH
    if not p.exists():
        raise FileNotFoundError("results/regional_branding_scores.csv 가 없습니다. 먼저 지표 산출을 실행하세요.")
    return p

def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None

def minmax01(series: pd.Series) -> pd.Series:
    vals = series.astype(float).values.reshape(-1, 1)
    scaled = MinMaxScaler().fit_transform(vals).ravel()
    return pd.Series(scaled, index=series.index)

def tokens_ko(s: str) -> List[str]:
    pat = re.compile(r"[가-힣]{2,}")
    basic_stop = set(("하다 되다 이다 있다 없다 아니다 그리고 그러나 그래서 또는 또한 매우 너무 더 좀 같이 "
                     "하면 해서 한 후 전 후에 위해 위한 대한 등의 등 으로 으로서 으로써 로서 로써 에서 에 "
                     "부터 까지 도 는 은 이 가 을 를 의 에 게 와 과 로 는데 다가 보다 보다도 저희 우리 제가 내가 "
                     "당신 여러분 오늘 어제 내일 지금 현재 경우 정말 진짜 바로 그냥 모두 모든 사진 블로그 리뷰 방문 "
                     "위치 제공 이용 확인 안내 소개 정보 메뉴 지도 링크 기사 뉴스 주차 예약 전화 번호 운영 시간 영업 "
                     "운영시간 가격 비용 할인 이벤트 행사 추천 인기 최신 베스트 근처 주변 인근 근교 가까운 가까이에").split())
    toks = [t for t in pat.findall(str(s)) if t not in basic_stop]
    return toks

def get_openai_client():
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai 패키지가 설치되지 않았거나 버전이 낮습니다. `pip install openai`로 설치/업데이트 해주세요.") from e
    api_key = load_env()
    return OpenAI(api_key=api_key)

def gpt_chat(client, prompt: str, model: str = OPENAI_MODEL, temperature: float = TEMPERATURE) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            time.sleep(RETRY_BACKOFF * attempt)
    # GPT 실패 시 빈 문자열 리턴 (리포트는 계속 생성되도록)
    return ""

# 수치 해석
def compute_quantile_bands(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    bands = {}
    for col in ["EPI", "U", "C", "G"]:
        if col not in df.columns:
            continue
        qs = df[col].quantile(QUANTILES).to_dict()
        bands[col] = {
            "q25": qs.get(QUANTILES[0], float("nan")),
            "q50": qs.get(QUANTILES[1], float("nan")),
            "q75": qs.get(QUANTILES[2], float("nan")),
        }
    return bands

def interpret_value(col: str, val: float, bands: Dict[str, Dict[str, float]]) -> str:
    b = bands.get(col, {})
    q25, q50, q75 = b.get("q25", math.nan), b.get("q50", math.nan), b.get("q75", math.nan)
    direction = "high" if col in ("EPI", "U", "C") else "low"
    if math.isnan(val) or any(map(math.isnan, [q25, q50, q75])):
        return "데이터 기준 부족"

    if direction == "high":
        if val >= q75: return "상위권"
        if val >= q50: return "중상위"
        if val >= q25: return "중하위"
        return "하위권"
    else:
        if val <= q25: return "상위권(괴리 낮음)"
        if val <= q50: return "중상위(괴리 중간)"
        if val <= q75: return "중하위(괴리 다소 높음)"
        return "하위권(괴리 큼)"

def build_rule_based_summary(row: pd.Series, bands: Dict[str, Dict[str, float]], slogan: Optional[str]) -> str:
    epi, u, c, g = float(row["EPI"]), float(row["U"]), float(row["C"]), float(row["G"])
    epi_tier = interpret_value("EPI", epi, bands)
    u_tier   = interpret_value("U", u, bands)
    c_tier   = interpret_value("C", c, bands)
    g_tier   = interpret_value("G", g, bands)

    hints = []
    hints.append("외부 인식은 비교적 확보됨(EPI 양호)" if epi_tier in ("상위권", "중상위") else "외부 인식 보강 필요(EPI 낮음)")
    hints.append("차별화 가능성 높음(U 양호)" if u_tier in ("상위권", "중상위") else "차별화 포인트 보강 필요(U 낮음)")
    hints.append("채널 인식 일관됨(C 양호)" if c_tier in ("상위권", "중상위") else "채널 메시지 일치 강화 필요(C 낮음)")
    hints.append("행정-사용자 인식 일치(G 낮음)" if "상위권" in g_tier else "행정-사용자 괴리 완화 필요(G 높음)")

    base = f"[진단요약] EPI:{epi_tier}, U:{u_tier}, C:{c_tier}, G:{g_tier}. " + " / ".join(hints)
    if slogan:
        base += f" [행정 슬로건] {slogan}"
    return base

def format_number(x: float, decimals: int = 3) -> str:
    try:
        return f"{float(x):.{decimals}f}"
    except:
        return str(x)

def make_region_comment_gpt(client, region: str, row: pd.Series, slogan: Optional[str], summary_hint: str) -> str:
    epi, u, c, g = row["EPI"], row["U"], row["C"], row["G"]
    prompt = f"""
다음 지역의 브랜딩 지표와 행정 슬로건, 규칙 기반 요약을 참고하여,
정책 보고서 톤의 간결한 코멘트를 한국어로 작성해 주세요.

- 지역명: {region}
- 지표: EPI={epi:.2f}, U={u:.3f}, C={c:.3f}, G={g:.3f}
- 행정 슬로건: {slogan if slogan else "슬로건 정보 없음"}
- 규칙 기반 요약: {summary_hint}

요청사항:
1) 2~3문장 내로 간결하게.
2) 첫 문장에 현재 포지션(외부 인식/차별화/정합성/괴리)에 대한 총평.
3) 두 번째 문장에 우선순위 1~2개 수준의 액션 제안(홍보/콘텐츠/메시지 정렬 중 택일).

출력은 마크다운 본문 문장만, 불릿 없이.
"""
    return gpt_chat(client, prompt)

def make_overall_intro(df: pd.DataFrame) -> str:
    def topn(col, n=3, asc=False):
        part = df.sort_values(col, ascending=asc).head(n)[["region", col]]
        lines = [f"- {r} ({col}: {format_number(v, 3)})" for r, v in part.values]
        return "\n".join(lines)

    lines = []
    lines.append("### 하이라이트")
    lines.append("**외부 인식(EPI) 상위 3**\n" + topn("EPI", 3, asc=False))
    lines.append("\n**차별화지수(U) 상위 3**\n" + topn("U", 3, asc=False))
    lines.append("\n**정합성(C) 상위 3**\n" + topn("C", 3, asc=False))
    lines.append("\n**괴리도(G) 하위 3(= 괴리 낮음)**\n" + topn("G", 3, asc=True))
    return "\n\n".join(lines)

# 리포트 생성 (MD + CSV)
def generate_markdown_report(
    df: pd.DataFrame,
    gov: Optional[pd.DataFrame],
    output_md: Path,
    output_csv: Path,
    client
) -> None:

    slogan_map = {}
    if gov is not None and "region" in gov.columns and "slogan" in gov.columns:
        for _, r in gov.iterrows():
            slogan_map[str(r["region"]).strip()] = str(r["slogan"]).strip()

    bands = compute_quantile_bands(df)

    md = []
    md.append(f"# 지역 브랜딩 분석 리포트")
    md.append(f"- 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    md.append("")
    md.append("본 리포트는 EPI(외부 인식), U(차별화), C(채널 정합성), G(행정-사용자 괴리) 4대 지표를 기반으로 작성되었습니다. "
              "EPI/U/C는 높을수록 긍정적이며, G는 낮을수록 바람직합니다.")
    md.append("")
    md.append(make_overall_intro(df))
    md.append("\n---\n")

    csv_rows: List[Dict] = []

    for _, row in df.sort_values("region").iterrows():
        region = row["region"]
        epi, u, c, g = float(row["EPI"]), float(row["U"]), float(row["C"]), float(row["G"])
        slogan = slogan_map.get(region)

        summary_hint = build_rule_based_summary(row, bands, slogan)
        comment = make_region_comment_gpt(client, region, row, slogan, summary_hint)

        md.append(f"## {region}")
        if slogan:
            md.append(f"- **행정 슬로건**: {slogan}")
        md.append("")
        md.append("| 지표 | 값 | 해석 |")
        md.append("|---|---:|---|")
        md.append(f"| EPI | {format_number(epi, 2)} | {interpret_value('EPI', epi, bands)} |")
        md.append(f"| U   | {format_number(u, 3)} | {interpret_value('U', u, bands)} |")
        md.append(f"| C   | {format_number(c, 3)} | {interpret_value('C', c, bands)} |")
        md.append(f"| G   | {format_number(g, 3)} | {interpret_value('G', g, bands)} |")
        md.append("")
        md.append(comment if comment else "_(코멘트 생성 실패: GPT 응답 없음)_")
        md.append("\n---\n")

        csv_rows.append({
            "region": region,
            "EPI": epi,
            "U": u,
            "C": c,
            "G": g,
            "EPI_tier": interpret_value("EPI", epi, bands),
            "U_tier": interpret_value("U", u, bands),
            "C_tier": interpret_value("C", c, bands),
            "G_tier": interpret_value("G", g, bands),
            "slogan": slogan or "",
            "comment": comment
        })

    output_md.write_text("\n".join(md), encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[완료] 리포트 저장 → {output_md}")
    print(f"[완료] 시각화용 CSV 저장 → {output_csv}")

# 메인
def main(scores_csv: Optional[str] = None, gov_csv: Optional[str] = None):
    client = get_openai_client()

    scores_path = find_scores(scores_csv)
    df = pd.read_csv(scores_path)

    required = {"region", "EPI", "U", "C", "G"}
    if not required.issubset(df.columns):
        raise ValueError(f"지표 CSV에 다음 컬럼이 필요합니다: {required}")

    gov_path = Path(gov_csv) if gov_csv else DEFAULT_GOV_SLOGANS
    gov_df = safe_read_csv(gov_path)

    ts = now_tag()
    out_md  = RESULTS_DIR / f"regional_branding_report_{ts}.md"
    out_csv = RESULTS_DIR / f"regional_branding_insights_{ts}.csv"

    generate_markdown_report(df, gov_df, out_md, out_csv, client)

if __name__ == "__main__":
    main()