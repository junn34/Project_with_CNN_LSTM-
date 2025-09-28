import pandas as pd
from newspaper import Article, Config
import requests
from bs4 import BeautifulSoup

excel_path = r"naver_news_기준금리.xlsx"
df = pd.read_excel(excel_path)

url_column = 'url'

config = Config()
config.browser_user_agent = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/115.0.0.0 Safari/537.36"
)

article_texts = []

for url in df[url_column]:
    try:
        if "news.einfomax.co.kr" in url:
            headers = {"User-Agent": config.browser_user_agent}
            resp = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            article_tag = soup.select_one("article#article-view-content-div")
            if article_tag:
                text = article_tag.get_text(separator="\n", strip=True)
            else:
                print(f"본문을 찾을 수 없습니다: {url}")
                text = ""

        elif any(domain in url for domain in ["www.hankyung.com", "www.dt.co.kr", "www.bizwatch.co.kr"]):
            article = Article(url, language='ko', config=config)
            article.download()
            article.parse()
            text = article.text

        else:
            article = Article(url, language='ko')
            article.download()
            article.parse()
            text = article.text

        article_texts.append(text)

    except Exception as e:
        print(f"URL 크롤링 실패: {url}, 오류: {e}")
        article_texts.append("")  # 실패 시 빈 문자열

df['content'] = article_texts

output_path = '기준금리_content.xlsx'
df.to_excel(output_path, index=False)