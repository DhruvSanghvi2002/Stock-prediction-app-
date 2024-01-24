from urllib.request import urlopen,Request
from bs4 import BeautifulSoup
import app
from app import user_input
news_tables={}
finviz_url="https://finviz.com/quote.ashx?t="
ticker=user_input
url=finviz_url+ticker
req=Request(url=url,headers={'user-agent':'Stock Prediction App'})
response=urlopen(req)
html=BeautifulSoup(response,'html',features='lxml')
news_table=html.find(id='news-table')
news_tables[ticker]=news_table
print(news_tables)