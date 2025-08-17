import os
import re
import time
import json
import requests
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains


REQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.naver.com/",
}
SEL_WAIT = 15


def _requests_get(url: str) -> BeautifulSoup:
    res = requests.get(url, headers=REQ_HEADERS, timeout=15)
    res.raise_for_status()
    return BeautifulSoup(res.text, "html.parser")


def _selenium_get_dom(url: str, iframe_url: Optional[str] = None) -> webdriver.Chrome:
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
    driver = webdriver.Chrome(service=service, options=opts)
    driver.get(iframe_url or url)
    return driver


def _switch_into_iframe_if_present(driver: webdriver.Chrome) -> None:
    try:
        WebDriverWait(driver, SEL_WAIT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        if "m.blog.naver.com" in driver.current_url:
            return
        try:
            iframe = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.ID, "mainFrame"))
            )
            driver.switch_to.frame(iframe)
        except Exception:
            pass
        WebDriverWait(driver, SEL_WAIT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
        )
    except Exception:
        pass

def _expand_tag_panel(driver: webdriver.Chrome) -> None:
    """
    일부 스킨은 '태그' 섹션이 접혀 있음 → 더보기/토글 클릭으로 펼침
    """
    candidates = [
        ".post_tag .btn_more",
        ".tag_list_area .btn_more",
        "button[aria-controls*='tag']",
        "a#tag_toggle",
        ".se_tag_area .btn_more",
        ".post_tag .btn_open",
        ".tag_area .btn_more",
    ]
    try:
        driver.execute_script("window.scrollBy(0, 400);")
        for css in candidates:
            try:
                btns = driver.find_elements(By.CSS_SELECTOR, css)
                if not btns:
                    continue
                for b in btns:
                    if not b.is_displayed():
                        continue
                    try:
                        ActionChains(driver).move_to_element(b).pause(0.1).click(b).perform()
                    except Exception:
                        try:
                            driver.execute_script("arguments[0].click();", b)
                        except Exception:
                            continue
                    time.sleep(0.4)
            except Exception:
                continue
    except Exception:
        pass

def _collect_tags_via_regex(html_or_soup) -> List[str]:
    """
    마지막 보루: HTML 전체에서 태그 후보를 정규식으로 긁음
    - /TagSearch.naver?...tag=키워드
    - "tagName":"키워드"
    - meta keywords
    """
    text = html_or_soup if isinstance(html_or_soup, str) else str(html_or_soup)
    cand: List[str] = []

    for m in re.findall(r'[?&](?:tag|keyword|query)=([^&#"]+)', text, flags=re.I):
        cand.append(m)

    for m in re.findall(r'"tagName"\s*:\s*"([^"]+)"', text):
        cand.append(m)

    for m in re.findall(
        r'<meta[^>]+name=["\']keywords["\'][^>]+content=["\']([^"\']+)["\']',
        text, flags=re.I
    ):
        parts = re.split(r'[,\s/|]+', m)
        cand.extend([p for p in parts if p])

    try:
        from urllib.parse import unquote
        cand = [unquote(x) for x in cand]
    except Exception:
        pass

    return _clean_tags(cand)

def _first_text(driver: webdriver.Chrome, selectors: List[str]) -> str:
    for css in selectors:
        try:
            el = driver.find_elements(By.CSS_SELECTOR, css)
            if el and el[0].text.strip():
                return el[0].text.strip()
        except Exception:
            continue
    return ""


def _collect_texts(driver: webdriver.Chrome, selector: str) -> List[str]:
    try:
        WebDriverWait(driver, SEL_WAIT).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
        )
    except Exception:
        return []
    items = driver.find_elements(By.CSS_SELECTOR, selector)
    texts = [i.text.strip() for i in items if i.text and i.text.strip()]
    return list(dict.fromkeys(texts))


def _get_content_text(driver: webdriver.Chrome) -> str:
    selectors = [
        ".se-main-container",       # 스마트에디터 최신
        "#postViewArea",            # 구버전
        ".se_component_wrap",       # 일부 최신 스킨
        ".se_textView",             # 구/혼합
        "#viewTypeSelector",        # 모바일 일부
        "#ct",                      # 모바일 컨테이너
    ]
    try:
        WebDriverWait(driver, SEL_WAIT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ", ".join(selectors)))
        )
    except Exception:
        pass

    longest = ""
    for sel in selectors:
        try:
            elems = driver.find_elements(By.CSS_SELECTOR, sel)
            for e in elems:
                t = (e.text or "").strip()
                if len(t) > len(longest):
                    longest = t
        except Exception:
            continue

    if not longest:
        try:
            longest = driver.find_element(By.TAG_NAME, "body").text.strip()
        except Exception:
            pass
    return longest


def _clean_tags(raw: List[str]) -> List[str]:
    drop_words = {"태그", "tag", "해시태그"}
    cleaned: List[str] = []
    for t in raw:
        if not t:
            continue
        s = t.strip()
        s = re.sub(r"^[#\u0023\uFE0F\s]+", "", s)
        s = s.strip("·•,|/#")
        if not s or s in drop_words or len(s) < 2:
            continue
        cleaned.append(s)
    return list(dict.fromkeys(cleaned))


def _collect_tags_from_dom(driver: webdriver.Chrome) -> List[str]:
    """
    태그 텍스트가 요소 text가 아닌 title/aria-label/data-* 또는 href 쿼리에만 있는 스킨 대응
    """
    selectors = [
        ".se-hashtag",
        ".tag_list li a",
        ".post_tag a",
        ".se_tag_area a",
        "a.link_tag",
        "a[class*='tag']",
        "span[class*='tag'] a",
        "dl.tag_area dd a",
    ]

    tags: List[str] = []

    for css in selectors:
        try:
            for el in driver.find_elements(By.CSS_SELECTOR, css):
                txt = (el.text or "").strip()
                if txt:
                    tags.append(txt)
        except Exception:
            pass

    attr_candidates = ["title", "aria-label", "data-tag", "data-log-tag", "data-click-tag", "data-value", "data_keyword"]
    for css in selectors:
        try:
            for el in driver.find_elements(By.CSS_SELECTOR, css):
                for attr in attr_candidates:
                    val = el.get_attribute(attr)
                    if val and val.strip():
                        tags.append(val.strip())
        except Exception:
            pass

    for css in selectors:
        try:
            for el in driver.find_elements(By.CSS_SELECTOR, css):
                href = el.get_attribute("href") or ""
                if not href:
                    continue
                try:
                    q = parse_qs(urlparse(href).query)
                    for key in ("tag", "keyword", "query"):
                        if key in q and q[key]:
                            tags.extend([v for v in q[key] if v])
                except Exception:
                    continue
        except Exception:
            pass

    expanded: List[str] = []
    for t in tags:
        parts = re.split(r"[,\s/|]+", t)
        for p in parts:
            if p:
                expanded.append(p)

    return _clean_tags(expanded)

def _extract_ids_from_urls(url: str, iframe_url: Optional[str]) -> Optional[Dict[str, str]]:
    """
    주어진 URL/iframe_url 쿼리에서 blogId, logNo를 추출
    """
    def pick(qs):
        blog_id = qs.get("blogId", [None])[0]
        log_no  = qs.get("logNo", [None])[0]
        if blog_id and log_no:
            return {"blogId": blog_id, "logNo": log_no}
        return None

    # 1) 원본 URL에서 시도
    try:
        qs = parse_qs(urlparse(url).query)
        r = pick(qs)
        if r:
            return r
    except Exception:
        pass

    # 2) iframe_url에서 시도
    if iframe_url:
        try:
            qs = parse_qs(urlparse(iframe_url).query)
            r = pick(qs)
            if r:
                return r
        except Exception:
            pass

    return None

def _get_tags_via_mobile_json(blog_id: str, log_no: str) -> List[str]:
    """
    m.blog.naver.com/PostView.json 호출로 태그 추출
    스키마가 종종 바뀌므로 여러 경로를 방어적으로 조회
    """
    api = f"https://m.blog.naver.com/PostView.json?blogId={blog_id}&logNo={log_no}"
    try:
        res = requests.get(
            api,
            headers={
                **REQ_HEADERS,
                "Referer": f"https://m.blog.naver.com/{blog_id}/{log_no}",
                "Accept": "application/json, text/plain, */*",
            },
            timeout=15,
        )
        res.raise_for_status()
        data = res.json()
    except Exception:
        return []

    # 가능한 경로 후보들
    candidates = [
        ["result", "post", "tagList"],   # 가장 흔함: [{tagName: "..."}]
        ["post", "tagList"],
        ["result", "tagList"],
    ]

    tag_objs: List[Dict[str, Any]] = []
    for path in candidates:
        cur: Any = data
        ok = True
        for k in path:
            if isinstance(cur, dict) and (k in cur):
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, list):
            tag_objs = cur
            break

    tags: List[str] = []
    for obj in tag_objs:
        if not isinstance(obj, dict):
            continue
        for key in ("tagName", "name", "keyword", "tag"):
            v = obj.get(key)
            if v and isinstance(v, str) and v.strip():
                tags.append(v.strip())

    return _clean_tags(tags)

def _extract_ids_from_urls(url: str, iframe_url: Optional[str]) -> Optional[Dict[str, str]]:
    """주어진 URL/iframe_url 쿼리에서 blogId, logNo 추출"""
    def pick(qs):
        blog_id = qs.get("blogId", [None])[0]
        log_no  = qs.get("logNo", [None])[0]
        if blog_id and log_no:
            return {"blogId": blog_id, "logNo": log_no}
        return None

    try:
        qs = parse_qs(urlparse(url).query)
        r = pick(qs)
        if r:
            return r
    except Exception:
        pass

    if iframe_url:
        try:
            qs = parse_qs(urlparse(iframe_url).query)
            r = pick(qs)
            if r:
                return r
        except Exception:
            pass

    return None

def _get_tags_from_mobile_html(blog_id: str, log_no: str) -> List[str]:
    url = f"https://m.blog.naver.com/{blog_id}/{log_no}"
    try:
        soup = _requests_get(url)
    except Exception:
        return []

    sels = [
        ".post_tag a",
        ".post_tag_inner a",
        ".tag_list_area a",
        ".tag_area a",
        "a.link_tag",
        "a[href*='TagSearch.naver']",
    ]
    raw: List[str] = []
    for sel in sels:
        for a in soup.select(sel):
            txt = (a.get_text(strip=True) or "").strip()
            if txt:
                raw.append(txt)
            title = a.get("title")
            if title and title.strip():
                raw.append(title.strip())
            href = a.get("href")
            if href:
                try:
                    q = parse_qs(urlparse(href).query)
                    for key in ("tag", "keyword", "query"):
                        if key in q and q[key]:
                            raw.extend([v for v in q[key] if v])
                except Exception:
                    pass

    parsed = _clean_tags(raw)
    if not parsed:
        parsed = _collect_tags_via_regex(soup)
    return parsed

def crawl_blog_post(url: str) -> Optional[Dict[str, Any]]:
    """
    네이버 블로그 본문 크롤링 (requests 1차 시도 + Selenium 렌더링 fallback)
    반환: {"title": str, "content": str, "date": str, "tags": List[str]}
    """
    try:
        soup = _requests_get(url)
    except Exception:
        soup = None

    iframe_url = None
    if soup:
        iframe = soup.find("iframe", id="mainFrame")
        if iframe and iframe.get("src"):
            iframe_url = "https://blog.naver.com" + iframe["src"]

    title, content, date_text, tags = "", "", "", []
    resolved_url = None

    if iframe_url:
        try:
            soup2 = _requests_get(iframe_url)

            # 제목
            title_el = soup2.select_one(".se-title-text, .pcol1, .se_title, h3.se_textarea")
            title = title_el.get_text(strip=True) if title_el else ""

            # 본문
            content_el = soup2.select_one(".se-main-container, #postViewArea")
            content = content_el.get_text(" ", strip=True) if content_el else ""

            # 날짜
            date_el = soup2.select_one(
                "span.se_publishDate, .date, .se_publish_date, .se_date, span.post_date, time"
            )
            if not date_el and soup:
                date_el = soup.select_one(
                    "span.se_publishDate, .date, .se_publish_date, .se_date, span.post_date, time"
                )
            date_text = date_el.get_text(strip=True) if date_el else ""

            # 태그
            tags = [
                t.get_text(strip=True)
                for t in soup2.select(".se-hashtag, .tag_list li a, .post_tag a, .se_tag_area a")
                if t.get_text(strip=True)
            ]
            if not tags and soup:
                tags = [
                    t.get_text(strip=True)
                    for t in soup.select(".se-hashtag, .tag_list li a, .post_tag a, .se_tag_area a")
                    if t.get_text(strip=True)
                ]
            tags = _clean_tags(tags)
            if not tags:
                tags = _collect_tags_via_regex(soup2)
        except Exception:
            pass

    if (not date_text) or (not tags) or (not content):
        driver = None
        try:
            driver = _selenium_get_dom(url, iframe_url)
            _switch_into_iframe_if_present(driver)

            try:
                resolved_url = driver.current_url
            except Exception:
                pass

            # 제목 보강
            if not title:
                title = _first_text(
                    driver,
                    [
                        ".se-title-text",
                        ".pcol1",
                        ".se_title",
                        "h3.se_textarea",
                        "h3.se-module-title",
                    ],
                )

            # 본문 보강
            if not content:
                content = _get_content_text(driver)

            # 날짜 보강
            if not date_text:
                date_text = _first_text(
                    driver,
                    [
                        ".se_publishDate",
                        ".date",
                        ".post_date",
                        ".se_date",
                        "[class*='date']",
                        "time",
                    ],
                )
                if not date_text:
                    try:
                        t = driver.find_elements(By.CSS_SELECTOR, "time")
                        if t:
                            dt = t[0].get_attribute("datetime") or t[0].get_attribute("title")
                            if dt:
                                date_text = dt.strip()
                    except Exception:
                        pass

            # 태그 보강
            if not tags:
                _expand_tag_panel(driver)
                dom_tags = _collect_tags_from_dom(driver)
                if dom_tags:
                    tags = dom_tags

            if not tags:
                try:
                    tags = _collect_tags_via_regex(driver.page_source)
                except Exception:
                    pass

        finally:
            if driver:
                driver.quit()

    if not (title or content):
        return None
    
    if not resolved_url:
        resolved_url = iframe_url or url

    return {
        "title": title,
        "content": content,
        "date": date_text,
        "tags": tags,
        "resolved_url": resolved_url
    }