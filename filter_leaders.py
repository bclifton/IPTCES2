import datetime as dt
import pandas as pd
import glob

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


