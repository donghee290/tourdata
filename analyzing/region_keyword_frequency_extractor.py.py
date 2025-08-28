import re
from collections import Counter
from typing import List
import pandas as pd

INPUT_CSV = "naver_blog_local_with_body.csv"
OUTPUT_CSV = "region_keywords_frequency.csv"
TOP_N = 20  # 지역별 상위 키워드 개수

# 한국어 토크나이저 & 불용어
KOREAN_PATTERN = re.compile(r"[가-힣]{2,}")
STOPWORDS_KO = set("""
하다 되다 이다 있다 없다 아니다 그리고 그러나 그래서 또는 또한 매우 너무 더 좀 같이
하면 해서 한 후 전 후에 위해 위한 대한 대한들 등의 등 으로 으로서 으로써 로서 로써 에서 에
부터 까지 도 는 은 이 가 을 를 의 에 게 와 과 로 는데 다가 보다 보다도 보다가는
저희 우리 제가 내가 너가 당신 여러분 오늘 어제 내일 지금 현재 경우 경우에 경우로
정말 진짜 바로 그냥 모두 모든 많은 많이 조금 가장 사실 솔직히 약간
사진 블로그 리뷰 방문 위치 제공 이용 확인 안내 소개 정보 메뉴 지도 링크 클릭 기사 뉴스
주차 예약 전화 번호 운영 시간 영업 운영시간 가격 비용 할인 이벤트 행사 추천 인기 최신 베스트
근처 주변 인근 근교 가까운 가까이에
""".split())

def tokenize_ko(text_in: str) -> List[str]:
    tokens = KOREAN_PATTERN.findall(str(text_in))
    return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS_KO]

def main():
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    assert "region" in df.columns and "snippet" in df.columns, "CSV에 'region'과 'snippet' 컬럼이 필요합니다."

    work = df[["region", "snippet"]].astype(str)
    work["region"] = work["region"].str.strip()
    work["snippet"] = work["snippet"].str.replace(r"\s+", " ", regex=True).str.strip()
    work = work[(work["region"] != "") & (work["snippet"] != "")]

    rows = []
    for region, sub in work.groupby("region"):
        bag = []
        for txt in sub["snippet"]:
            bag.extend(tokenize_ko(txt))
        counts = Counter(bag)
        for kw, cnt in counts.most_common(TOP_N):
            rows.append({"region": region, "keyword": kw, "count": cnt})

    out = pd.DataFrame(rows).sort_values(["region", "count"], ascending=[True, False])
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[완료] 저장: {OUTPUT_CSV} (행수: {len(out)})")

if __name__ == "__main__":
    main()