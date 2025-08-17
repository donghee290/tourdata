import os
import requests
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

def get_blog_links(query: str, display: int = 10):
    """
    네이버 블로그 검색 API로 블로그 링크 목록 가져오기
    sort=sim → 관련도순, date → 최신순
    """
    url = f"https://openapi.naver.com/v1/search/blog.json?query={quote(query)}&display={display}&sort=sim"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    items = res.json().get("items", [])
    return [item["link"] for item in items]