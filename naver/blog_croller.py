import os
import re
import time
import requests
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, parse_qs, unquote
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


def _new_driver(iframe_url: Optional[str], url: str) -> webdriver.Chrome:
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
        WebDriverWait(driver, SEL_WAIT).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
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
    candidates = [
        ".post_tag .btn_more",
        ".tag_list_area .btn_more",
        "button[aria-controls*='tag']",
        "#tag_toggle",
        ".se_tag_area .btn_more",
        ".post_tag .btn_open",
        ".tag_area .btn_more",
    ]
    try:
        driver.execute_script("window.scrollBy(0, 400);")
        for css in candidates:
            btns = driver.find_elements(By.CSS_SELECTOR, css)
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
                time.sleep(0.3)
                return
    except Exception:
        pass


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = s.replace("\xa0", " ")
    return s.strip()


def _clean_tags(raw: List[str]) -> List[str]:
    drop_words = {"태그", "tag", "해시태그"}
    out: List[str] = []
    seen = set()
    for t in raw:
        if not t:
            continue
        s = _normalize_text(t)
        s = re.sub(r"^[#\u0023\uFE0F\s]+", "", s)
        s = s.strip("·•,|/#[]()")
        if not s or s in drop_words:
            continue
        s = re.sub(r"\s{2,}", " ", s)
        if len(s) < 2:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _extract_ids_from_urls(url: str, iframe_url: Optional[str]) -> Optional[Dict[str, str]]:
    def pick(qs):
        blog_id = qs.get("blogId", [None])[0]
        log_no = qs.get("logNo", [None])[0]
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

    try:
        soup = _requests_get(url)
        og = soup.find("meta", property="og:url")
        if og and og.get("content"):
            qs = parse_qs(urlparse(og["content"]).query)
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
    
    def _from_meta_tags(soup: BeautifulSoup) -> List[str]:
        raw = []
        for m in soup.find_all("meta", attrs={"property": ["og:article:tag", "article:tag"]}):
            c = (m.get("content") or "").strip()
            if c:
                raw.append(c)
        # name="keywords" 백업 (콤마/슬래시/파이프 구분)
        for m in soup.find_all("meta", attrs={"name": "keywords"}):
            c = (m.get("content") or "").strip()
            if c:
                raw.extend([x.strip() for x in re.split(r"[,\|/]", c) if x.strip()])
        return _clean_tags(raw)


    def _from_anchor(soup: BeautifulSoup) -> List[str]:
        sels = [
            ".post_tag a", ".post_tag_inner a", ".tag_list_area a", ".tag_area a",
            "a.link_tag", "a[href*='TagSearch.naver']", "dl.tag_area dd a",
            'a[aria-label*="태그"]', ".se_hashtag a", ".se-hashtag a"
        ]
        raw: List[str] = []
        for sel in sels:
            for a in soup.select(sel):
                txt = _normalize_text(a.get_text(" ", strip=True))
                if txt:
                    raw.append(txt)
                href = a.get("href") or ""
                if href:
                    try:
                        q = parse_qs(urlparse(href).query)
                        for key in ("tag", "keyword", "query"):
                            for v in q.get(key, []):
                                if v:
                                    raw.append(unquote(v))
                    except Exception:
                        pass
                for k in ("title", "aria-label", "data-tag", "data_keyword", "data-value"):
                    v = a.get(k)
                    if v:
                        raw.append(_normalize_text(v))
        return _clean_tags(raw)

    def _from_ldjson(soup: BeautifulSoup) -> List[str]:
        # schema.org Article에 keywords가 문자열(콤마) 또는 배열로 올 수 있음
        try:
            for s in soup.find_all("script", type="application/ld+json"):
                txt = s.string or s.text
                if not txt:
                    continue
                data = json.loads(txt)
                if isinstance(data, list):
                    cands = data
                else:
                    cands = [data]
                raw = []
                for d in cands:
                    kw = d.get("keywords")
                    if not kw:
                        continue
                    if isinstance(kw, str):
                        raw.extend([x.strip() for x in re.split(r"[,\|/]", kw) if x.strip()])
                    elif isinstance(kw, list):
                        raw.extend([str(x) for x in kw if x])
                cleaned = _clean_tags(raw)
                if cleaned:
                    return cleaned
        except Exception:
            pass
        return []

    def _from_inline_json(soup: BeautifulSoup) -> List[str]:
        # Next.js(__NEXT_DATA__) / Apollo / 임베디드 JSON에서 hashtags / tagList 등 긁기
        raw: List[str] = []
        try:
            # 1) __NEXT_DATA__ 전체 JSON에서 배열형 키 추출
            for sc in soup.find_all("script"):
                txt = sc.string or sc.text
                if not txt or ("tag" not in txt and "hash" not in txt and "keyword" not in txt):
                    continue
                # 배열 형태 ["태그1","태그2"] 를 통째로 꺼내기
                for arr_txt in re.findall(r'(?:"(?:tags?|hash(?:Tags?|tags?)|keywords?)"\s*:\s*\[(.*?)\])', txt, flags=re.I|re.S):
                    # 배열 내부 문자열 전부 추출
                    for m in re.findall(r'"([^"]+)"', arr_txt):
                        raw.append(m)
                # 객체 배열 내부의 name 필드("tagName","hashtagName","name") 뽑기
                for m in re.findall(r'"(?:tagName|hashTagName|hashtagName|name)"\s*:\s*"([^"]+)"', txt):
                    raw.append(m)
                # 쿼리스트링에 태그 담긴 경우도 잡기
                for m in re.findall(r'[?&](?:tag|keyword|query)=([^&#"]+)', txt, flags=re.I):
                    raw.append(unquote(m))
        except Exception:
            pass
        return _clean_tags(raw)

    # 1) 앵커로 먼저 시도
    tags = _from_anchor(soup)
    if tags:
        return tags
    # 2) ld+json의 keywords
    tags = _from_ldjson(soup)
    if tags:
        return tags
    # 3) __NEXT_DATA__/임베디드 JSON
    tags = _from_inline_json(soup)
    if tags:
        return tags
    # 4) 마지막 보루: 전체 HTML 정규식
    return _collect_tags_via_regex(soup)


def _collect_tags_via_regex(html_or_soup) -> List[str]:
    text = html_or_soup if isinstance(html_or_soup, str) else str(html_or_soup)
    cand: List[str] = []

    # /TagSearch.naver?...tag=키워드
    for m in re.findall(r'[?&](?:tag|keyword|query)=([^&#"]+)', text, flags=re.I):
        cand.append(unquote(m))

    # "tagName":"키워드" / "hashtagName":"키워드" / "name":"키워드"
    for m in re.findall(r'"(?:tagName|hashTagName|hashtagName|name)"\s*:\s*"([^"]+)"', text):
        cand.append(m)

    # "tags":["A","B"] / "hashtags":["A","B"] / "keywords":["A","B"]
    for arr_txt in re.findall(r'(?:"(?:tags?|hash(?:Tags?|tags?)|keywords?)"\s*:\s*\[(.*?)\])', text, flags=re.I|re.S):
        for m in re.findall(r'"([^"]+)"', arr_txt):
            cand.append(m)

    # meta keywords
    for m in re.findall(r'<meta[^>]+name=["\']keywords["\'][^>]+content=["\']([^"\']+)["\']', text, flags=re.I):
        parts = re.split(r'[,\s/|]+', m)
        cand.extend([p for p in parts if p])

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


def _get_content_text(driver: webdriver.Chrome) -> str:
    selectors = [
        ".se-main-container",
        "#postViewArea",
        ".se_component_wrap",
        ".se_textView",
        "#viewTypeSelector",
        "#ct",
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


def _collect_tags_with_js(driver: webdriver.Chrome) -> List[str]:
    js = r"""
    (function(){
      const qs = (s)=>Array.from(document.querySelectorAll(s));
      const links = [
        ...qs('.se_tag_area a'),
        ...qs('.post_tag a'),
        ...qs('.post_tag_inner a'),
        ...qs('.tag_list_area a'),
        ...qs('a[href*="TagSearch.naver"]'),
        ...qs('a[class*="tag"]'),
        ...qs('span[class*="tag"] a'),
        ...qs('dl.tag_area dd a')
      ];
      const out = [];
      for (const a of links){
        let t = (a.textContent || '').trim();
        if (!t) {
          try {
            const u = new URL(a.href, location.href);
            t = u.searchParams.get('tag') || u.searchParams.get('keyword') || u.searchParams.get('query') || '';
          } catch(e) {}
        }
        if (!t) {
          t = (a.getAttribute('title') || a.getAttribute('aria-label') ||
               a.getAttribute('data-tag') || a.getAttribute('data_keyword') ||
               a.getAttribute('data-value') || '').trim();
        }
        if (t) {
          t = t.replace(/^#/, '').trim();
          out.push(t);
        }
      }
      const seen = new Set(); const dedup = [];
      for (const v of out) { if (v && !seen.has(v)) { seen.add(v); dedup.push(v); } }
      return dedup;
    })();
    """
    try:
        return driver.execute_script(js) or []
    except Exception:
        return []


def crawl_blog_post(url: str) -> Optional[Dict[str, Any]]:
    """
    네이버 블로그 본문 크롤링
    반환: {"title": str, "content": str, "date": str, "tags": List[str], "resolved_url": str}
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
    resolved_url = iframe_url or url

    ids = _extract_ids_from_urls(url, iframe_url)
    if ids:
        t1 = _get_tags_from_mobile_html(ids["blogId"], ids["logNo"])
        if t1:
            tags = t1

    if iframe_url:
        try:
            soup2 = _requests_get(iframe_url)
            title_el = soup2.select_one(".se-title-text, .pcol1, .se_title, h3.se_textarea")
            if title_el:
                title = title_el.get_text(strip=True)
            content_el = soup2.select_one(".se-main-container, #postViewArea")
            if content_el:
                content = content_el.get_text(" ", strip=True)
            date_el = soup2.select_one(
                "span.se_publishDate, .date, .se_publish_date, .se_date, span.post_date, time"
            )
            if not date_el and soup:
                date_el = soup.select_one(
                    "span.se_publishDate, .date, .se_publish_date, .se_date, span.post_date, time"
                )
            if date_el:
                date_text = date_el.get_text(strip=True)
        except Exception:
            pass

    if (not content) or (not date_text) or (not title) or (not tags):
        driver = None
        try:
            driver = _new_driver(iframe_url, url)
            _switch_into_iframe_if_present(driver)
            try:
                resolved_url = driver.current_url or resolved_url
            except Exception:
                pass

            if not title:
                title = _first_text(
                    driver,
                    [".se-title-text", ".pcol1", ".se_title", "h3.se_textarea", "h3.se-module-title"],
                )
            if not content:
                content = _get_content_text(driver)
            if not date_text:
                date_text = _first_text(
                    driver,
                    [".se_publishDate", ".date", ".post_date", ".se_date", "[class*='date']", "time"],
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

            if not tags:
                _expand_tag_panel(driver)
                js_tags = _collect_tags_with_js(driver)
                if js_tags:
                    tags = _clean_tags(js_tags)

            if not tags:
                tags = _collect_tags_via_regex(driver.page_source)

            if not tags and ids:
                m_tags = _get_tags_from_mobile_html(ids["blogId"], ids["logNo"])
                if m_tags:
                    tags = m_tags
        finally:
            if driver:
                driver.quit()

    if not (title or content):
        return None

    return {
        "title": title,
        "content": content,
        "date": date_text,
        "tags": tags,
        "resolved_url": resolved_url,
    }