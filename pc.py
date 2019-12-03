import urllib
import re
import xlwt
from bs4 import BeautifulSoup
from distutils.filelist import findall
    
page = urllib.request.urlopen('https://home.firefoxchina.cn/')
contents = page.read()
#print(contents)
soup = BeautifulSoup(contents,"html.parser")
soup_f=soup.find('div', class_='hot-right')

newTable = 'test.xls'
wb=xlwt.Workbook(encoding='utf-8')
ws=wb.add_sheet('test1')
headData=['新闻内容','新闻链接']
for i in range(0, 2):
    ws.write(0,i,headData[i])

index = 1
for tag in soup_f.find_all('li'):
    item_href=tag.find('a').get('href')
    item_name=tag.find('a').get('title')
    ws.write(index,0,item_name)
    ws.write(index,1,item_href)
    index+=1
wb.save(newTable)
    