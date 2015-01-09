import sys
import csv
from bs4 import BeautifulSoup

with open('leaders.html') as f:
    data = f.read()

with open('twitter_leaders.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['username', 'userid', 'name', 'bio'])

    for item in BeautifulSoup(data).select('.stream-item'):
        name = item.select('.fullname')[0].text.strip().encode('utf8')
        username = item.select('.username')[0].text.strip().encode('utf8')
        bio = item.select('.bio')[0].text.strip().encode('utf8')
        userid = item.select('.avatar')[0].get('data-user-id').strip().encode('utf8')

        writer.writerow([username, userid, name, bio])
