from __future__ import annotations
import os
import re
import csv
import datetime as dt
from typing import List, Dict, Any, Optional

def _default_csv_path(region: str) -> str:
    safe_region = re.sub(r"[^\w가-힣]+", "_", region).strip("_") or "region"
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"naver_blog_{safe_region}_{ts}.csv"
    return os.path.join(os.getcwd(), filename)

def save_posts_to_csv(
    posts: List[Dict[str, Any]],
    region: str,
    csv_path: Optional[str] = None,
) -> str:
    """
    posts: crawl_blog_post / rank_by_relevance 결과 리스트
    region: 입력 지역명(모든 행에 동일 값으로 추가)
    csv_path: 지정 시 append 모드로 저장, 없으면 새 파일 생성
    반환: 실제 저장된 CSV 경로
    """
    if not posts:
        raise ValueError("posts 가 비었습니다.")

    path = csv_path or _default_csv_path(region)
    write_header = not os.path.exists(path)

    fieldnames = [
        "region",
        "title",
        "date",
        "tags",
        "content",
        "resolved_url"
    ]

    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for p in posts:
            writer.writerow({
                "region": region,
                "title": (p.get("title") or "").strip(),
                "date": (p.get("date") or "").strip(),
                "tags": ", ".join(p.get("tags") or []),
                "content": (p.get("content") or "").strip(),
                "resolved_url": (p.get("resolved_url") or "").strip()
            })

    return path