from typing import List, Dict, Any
from API_getter import get_blog_links
from naver.blog_crawler import crawl_blog_post, rank_by_relevance
from database_saver import save_posts_to_csv

FETCH_MULTIPLIER = 3

def _dedup_by_url(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for p in posts:
        u = (p.get("resolved_url") or p.get("source_url") or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(p)
    return out

if __name__ == "__main__":
    region = input("지역명을 입력하세요: ").strip()
    display_count_raw = input("몇 개의 게시글을 가져올까요? (기본 5개): ").strip()

    try:
        display_count = int(display_count_raw)
    except ValueError:
        display_count = 5

    # 1) 검색(리콜↑)
    keyword = f"{region} 여행 관광 맛집 볼거리"
    fetch_count = max(display_count * FETCH_MULTIPLIER, display_count)
    links: List[str] = get_blog_links(keyword, display=fetch_count)

    # 2) 전부 크롤링
    posts: List[Dict[str, Any]] = []
    for link in links:
        try:
            post = crawl_blog_post(link)
            if post:
                posts.append(post)
        except Exception:
            continue

    if not posts:
        print("검색/크롤 결과가 없습니다.")
        raise SystemExit(0)

    # 3) 중복 제거
    posts = _dedup_by_url(posts)

    # 4) 지역명으로 관련도 랭킹 → 최종 N개 출력
    ranked = rank_by_relevance(posts, query=region, top_k=display_count)

    if not ranked:
        print("랭킹 결과가 없습니다.")
        raise SystemExit(0)

    # 5) 결과 출력
    for post in ranked:
        print("=" * 60)
        print(f"원본 URL:   {post.get('source_url', post.get('resolved_url', ''))}")
        print(f"최종 URL:   {post.get('resolved_url', '')}")
        print(f"제목: {post.get('title','')}")
        print(f"날짜: {post.get('date','')}")
        print(f"본문 전체:\n{post.get('content','')}")
        print(f"태그: {post.get('tags', [])}")

    # 6) CSV 저장 (지역 컬럼 포함)
    saved_path = save_posts_to_csv(ranked, region=region)
    print(f"\nCSV 저장 완료: {saved_path}")