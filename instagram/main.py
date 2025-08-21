from post_crawler import crawl_instagram_posts

if __name__ == "__main__":
    urls = [
        "https://www.instagram.com/p/XXXXXXXXXXX/",
        "https://www.instagram.com/p/YYYYYYYYYYY/",
    ]
    posts = crawl_instagram_posts(urls)
    for p in posts:
        print("="*60)
        print("원본 URL:", p.get("source_url"))
        print("최종 URL:", p.get("resolved_url"))
        print("작성자:", p.get("username"))
        print("날짜:", p.get("date"))
        print("위치:", p.get("location"))
        print("캡션:", p.get("caption"))
        print("태그:", p.get("tags"))