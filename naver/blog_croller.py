import os
import re
import time
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

    devnull = open(os.devnull, "w")
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
        except Exception:
            pass

    # 3) 날짜/태그/본문 중 하나라도 비면 Selenium으로 보강
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
                tags = _collect_tags_from_dom(driver)

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