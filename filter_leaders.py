import datetime as dt
import pandas as pd
import glob
import os

last_week = dt.datetime.today() - dt.timedelta(days = 7)

for f in glob.glob('csv/*.csv'):
    df = pd.read_csv(f, index_col='SQLDATE', parse_dates=True)

    if len(df) > 100:
        continue

    d = df.tail(1).index[0]
    last_entry = dt.datetime(d.year, d.month, d.day)
    if last_entry > last_week:
        continue

    print f

leaders = pd.read_csv('everyone.csv')
invalid = leaders[pd.isnull(leaders['username'])]
for row in invalid.iterrows():
    leader = row[1]
    print 'csv/' + leader['display_name'].replace(' ', '_') + '.csv'

for f in glob.glob('csv/*.csv'):
    display_name = os.path.basename(f).replace('.csv', '').replace('_', ' ')
    leader = leaders[leaders['display_name'] == display_name]
    if len(leader) == 0:
        print f
