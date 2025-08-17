from API_getter import get_blog_links
from blog_croller import crawl_blog_post

if __name__ == "__main__":
    region = input("지역명을 입력하세요: ").strip()
    display_count = input("몇 개의 게시글을 가져올까요? (기본 5개): ").strip()

    try:
        display_count = int(display_count)
    except ValueError:
        display_count = 5

    keyword = f"{region} 여행 관광 맛집 볼거리"
    links = get_blog_links(keyword, display=display_count)

    for link in links:
        post = crawl_blog_post(link)
        if post:
            print("="*60)
            print(f"원본 URL:   {post.get('source_url', link)}")
            print(f"최종 URL:   {post.get('resolved_url', link)}")
            print(f"제목: {post['title']}")
            print(f"날짜: {post['date']}")
            print(f"본문 전체:\n{post['content']}")
            print(f"태그: {post['tags']}")