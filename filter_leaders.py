import datetime as dt
import pandas as pd
import glob

last_week = dt.datetime.today() - dt.timedelta(days = 7)

for f in glob.glob('csv/*.csv'):
    df = pd.read_csv(f, index_col='SQLDATE', parse_dates=True)
    #df = pd.read_csv(f, parse_dates=True)
    #print df.tail(1)
    if len(df) > 100:
        continue

    #print df.tail(1).index[0]
    #d = df.tail(1)['SQLDATE'].values[0]
    d = df.tail(1).index[0]
    #last_entry = dt.datetime.strptime(str(d), '%Y%m%d')
    last_entry = dt.datetime(d.year, d.month, d.day)
    if last_entry > last_week:
        continue

    print f


