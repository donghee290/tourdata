import re
import time
import json
import os
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

REQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.google.com/",
}

SEL_WAIT = 15


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _requests_get(url: str) -> BeautifulSoup:
    res = requests.get(url, headers=REQ_HEADERS, timeout=15)
    res.raise_for_status()
    return BeautifulSoup(res.text, "html.parser")


def _new_driver() -> webdriver.Chrome:
    os.environ.setdefault("WDM_LOG", "0")
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1280,900")
    opts.add_argument("--log-level=3")
    opts.add_argument("--blink-settings=imagesEnabled=false")
    opts.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.page_load_strategy = "eager"
    opts.add_argument(f"user-agent={REQ_HEADERS['User-Agent']}")
    service = Service(ChromeDriverManager().install(), log_output=os.devnull)
    return webdriver.Chrome(service=service, options=opts)


_HASHTAG_RE = re.compile(
    r"(?<![0-9A-Za-z가-힣_])#([0-9A-Za-z_]+|[가-힣]+|[A-Za-z]+[0-9A-Za-z_]*)",
    flags=re.UNICODE
)

def _clean_tags(raw: List[str]) -> List[str]:
    out, seen = [], set()
    for t in raw:
        if not t:
            continue
        s = _normalize_text(t)
        s = re.sub(r"^[#\u0023\uFE0F\s]+", "", s)
        if not s:
            continue
        if len(s) > 50:
            continue
        if re.fullmatch(r"[A-Fa-f0-9]{3,8}", s):
            continue
        if re.search(r"(https?://|www\.)", s, flags=re.I):
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _extract_hashtags_from_text(text: str) -> List[str]:
    return _clean_tags([m.group(0) for m in _HASHTAG_RE.finditer(text)])


def _extract_hashtags_from_html(html: str) -> List[str]:
    tags = []

    tags.extend(_extract_hashtags_from_text(html))

    try:
        soup = BeautifulSoup(html, "html.parser")
        for sel in [
            ('meta', {"property": "og:title"}),
            ('meta', {"property": "og:description"}),
            ('meta', {"name": "description"}),
        ]:
            m = soup.find(*sel)
            if m and m.get("content"):
                tags.extend(_extract_hashtags_from_text(m["content"]))
        for sc in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(sc.string or sc.text or "{}")
                if isinstance(data, dict):
                    cap = data.get("caption") or data.get("articleBody") or ""
                    tags.extend(_extract_hashtags_from_text(cap))
            except Exception:
                continue
    except Exception:
        pass

    return _clean_tags(tags)

def _expand_caption_if_possible(driver: webdriver.Chrome) -> None:
    candidates = [
        "//span[contains(text(),'더 보기')]",            
        "//div[contains(text(),'더 보기')]",
        "//span[contains(text(),'more') and @role='button']",
        "//button//*[contains(text(),'more')]/..",
    ]
    for xp in candidates:
        try:
            el = WebDriverWait(driver, 2).until(EC.element_to_be_clickable((By.XPATH, xp)))
            el.click()
            time.sleep(0.3)
            return
        except Exception:
            continue


def _extract_meta_with_bs(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    out: Dict[str, Any] = {}

    ogt = soup.find("meta", {"property": "og:title"})
    if ogt and ogt.get("content"):
        c = ogt["content"]
        m = re.match(r"([^:]+) on Instagram", c)
        if m:
            out["username"] = m.group(1).strip()

    tm = soup.find("time")
    if tm and (tm.get("datetime") or tm.get("title")):
        out["date"] = (tm.get("datetime") or tm.get("title")).strip()

    loc_a = soup.select_one("a[href*='/explore/locations/']")
    if loc_a:
        out["location"] = _normalize_text(loc_a.get_text(" ", strip=True))

    ogd = soup.find("meta", {"property": "og:description"})
    if ogd and ogd.get("content"):
        out["caption_hint"] = ogd["content"]

    return out


def crawl_instagram_post(url: str) -> Optional[Dict[str, Any]]:
    """
    인스타그램 공개 포스트 URL에서 메타정보/캡션/해시태그 추출
    반환: {
        "username": str, "date": str, "location": str,
        "caption": str, "tags": List[str], "resolved_url": str
    }
    """
    meta_html = ""
    try:
        soup = _requests_get(url)
        meta_html = str(soup)
    except Exception:
        meta_html = ""

    meta = _extract_meta_with_bs(meta_html) if meta_html else {}

    driver = None
    caption = ""
    tags: List[str] = []
    resolved = url

    try:
        driver = _new_driver()
        driver.get(url)

        try:
            WebDriverWait(driver, SEL_WAIT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article"))
            )
        except Exception:
            pass

        _expand_caption_if_possible(driver)

        caption_selectors = [
            "[data-testid='post-comment-root'] span",
            "h1[dir='auto']",
            "ul li div[dir='auto'] span",
            "article div[role='button'] div[dir='auto']",
        ]
        for sel in caption_selectors:
            try:
                el = driver.find_elements(By.CSS_SELECTOR, sel)
                if el:
                    text = " ".join([e.text for e in el if e.text]).strip()
                    if len(text) > len(caption):
                        caption = text
            except Exception:
                continue

        html = driver.page_source or ""
        resolved = driver.current_url or url

        tags = _extract_hashtags_from_html(html)
        tags.extend(_extract_hashtags_from_text(caption))
        tags = _clean_tags(tags)

        if not meta.get("username") or not meta.get("date") or not meta.get("location"):
            meta2 = _extract_meta_with_bs(html)
            meta.update({k: v for k, v in meta2.items() if v})
    finally:
        if driver:
            driver.quit()

    if not tags and meta_html:
        tags = _extract_hashtags_from_html(meta_html)

    out = {
        "username": meta.get("username", ""),
        "date": meta.get("date", ""),
        "location": meta.get("location", ""),
        "caption": _normalize_text(caption or meta.get("caption_hint", "")),
        "tags": tags,
        "resolved_url": resolved,
    }
    if not (out["caption"] or out["tags"]):
        return out
    return out

def crawl_instagram_posts(urls: List[str]) -> List[Dict[str, Any]]:
    results = []
    for u in urls:
        try:
            if not u or "instagram.com" not in u:
                continue
            p = crawl_instagram_post(u)
            if p:
                p["source_url"] = u
                results.append(p)
        except Exception:
            continue
    return results