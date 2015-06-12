#!/usr/bin/env python
import sys
import os
import glob
import json
import csv
import random
import pickle
import datetime as dt
from collections import deque

import numpy as np
import pandas as pd
from pandas import *
from pandas.io.json import json_normalize

from bigquery import get_client

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
#from matplotlib import lines

import statsmodels.api as sm
from statsmodels.tsa import *
tsa = sm.tsa

import compose


##############################################

with open('config/settings.json') as json_data:
    settings = json.load(json_data)

#big query
client = get_client(settings['PROJECT_ID'], service_account=settings['SERVICE_ACCOUNT'], private_key=settings['KEY'], readonly=True)

# Font variables
FONT_LOCATION = 'assets/AkzidenzGrotesk/AkzidenzGroteskBE-Regular.otf'
font = fm.FontProperties(fname=FONT_LOCATION)

# Path variables
IMG_PATH = 'graphs/'
CSV_PATH = 'csv/'

# File variables
GS_PATH = 'assets/goldstein_suggestions.tsv'

# Time variables
tomorrow = str(dt.date.today())
twoweeksago = str(dt.date.today() - dt.timedelta(days = 14))
lastweek = dt.datetime.today() - dt.timedelta(days = 7)
nextweek = dt.date.today() + dt.timedelta(days = 8)
day_index = dt.datetime.today().weekday()

leaders = pd.read_csv('everyone.csv')
leaders = leaders.dropna(subset=['username', 'display_name', 'gdelt_search'])

total = len(leaders)
leaders_per_day = total/7
start = day_index * leaders_per_day
end = start + leaders_per_day

leaders = leaders[start:end]


##############################################################

def scrape():
    query_template = 'SELECT GLOBALEVENTID, SQLDATE, Actor1Name, Actor1CountryCode, GoldsteinScale, NumMentions FROM [gdelt-bq:full.events] {} AND SQLDATE>{} AND SQLDATE<{} IGNORE CASE;'

    date_end = ''.join(str(dt.date.today()).split('-'))

    for row in leaders.iterrows():
        leader = row[1]

        filename = leader['display_name'].replace(' ', '_') + '.csv'
        date_start = ''
        file_exists = os.path.isfile(CSV_PATH + filename)

        if file_exists:
            date_start = get_last_row(CSV_PATH + filename)[-1]
        else:
            date_start = '19790101'

        query = query_template.format(leader['gdelt_search'], date_start, date_end)

        print ''
        print 'Search: ' + leader['display_name']
        print 'Query: ' + query

        try:
            job_id, results = client.query(query, timeout=2000)
            print str(len(results)) + ' results'

            if results == []:
                continue

            df = json_normalize(results)
            df = df.sort(columns='SQLDATE')

            if file_exists:
                with open(CSV_PATH + filename, 'a') as f:
                    df.to_csv(f, header=False)
            else:
                df.to_csv(CSV_PATH + filename, encoding='utf-8')

            print 'Saved: ' + filename

        except Exception as e:
            print 'Timed out'
            print e

##############################################################

def get_last_row(csv_filename):
    with open(csv_filename, 'rb') as f:
        return deque(csv.reader(f), 1)[0]

##############################################################

def create_schedule():

    tweets = []

    gsCodes = pd.read_csv(GS_PATH, index_col='code', sep='\t')

    for row in leaders.iterrows():
        leader = row[1]

        f = CSV_PATH + leader['display_name'].replace(' ', '_') + '.csv'

        print 'Analyzing ' + f

        try:
            leader = leaders[(leaders.display_name == leader['display_name'])]
            tweet = perform_analysis(f, gsCodes, leader)
            if tweet:
                tweets.append(tweet)
        except:
            print 'Could not analyze ' + f
            continue

    total_seconds = 12*60*60
    interval = total_seconds / len(tweets)
    for i, tweet in enumerate(tweets):
        tweet['seconds'] = interval * i

    with open('tweet_schedule.p', 'wb') as f:
        pickle.dump(tweets, f)


##############################################################

def perform_analysis(data, gsCodes, leader):
    df = pd.read_csv(data, index_col='SQLDATE', parse_dates=True)
    if len(df) < 100:
        print 'skipped', data
        return False

    d = df.tail(1).index[0]
    last_entry = dt.datetime(d.year, d.month, d.day)
    if last_entry <= lastweek:
        print 'skipped', data
        return False

    name = leader['display_name'].values[0]
    country = leader['display_country'].values[0]
    username = leader['username'].values[0]

    if type(username) != type('') or type(name) != type('') or type(country) != type(''):
        return False


    # These steps incorporate the number of mentions a Goldstein score is associated with, reducing the impact of error in the event encoding, making the average better reflect the event's presence in the GDELT.
    df['GoldMentions'] = df['GoldsteinScale'] * df['NumMentions']
    goldstein = df.groupby([df.index.date]).agg({'GoldMentions': np.sum, 'NumMentions': np.sum})
    goldstein['GoldAverage'] = goldstein['GoldMentions'] / goldstein['NumMentions']

    full_daterange = pd.date_range(start = min(df.index), end = max(df.index))

    # ffill() takes care of days that do not have a entry / Goldstein score in GDELT:
    goldstein = goldstein.reindex(full_daterange).ffill()


    if len(goldstein['GoldAverage']) < 230:
        day_difference = 230 - len(goldstein['GoldAverage']) + 1

        new_dates = [goldstein.index[0] - dt.timedelta(days=x) for x in range(1, day_difference)]
        new_dates = reversed(new_dates)

        dummy = [None for x in range(1, day_difference)]
        
        temp_data = {}
        temp_data['NumMentions'] = dummy
        temp_data['GoldMentions'] = dummy
        temp_data['GoldAverage'] = dummy

        missing = pd.DataFrame(temp_data, index=new_dates, columns=['NumMentions', 'GoldMentions', 'GoldAverage'])

        temp_goldstein = missing.append(goldstein)
        goldstein = temp_goldstein.fillna(method='bfill')

    #------------- REMOVED THE SMOOTHING FROM THE GRAPH -----------------#
    # Creates a rolling_mean using a 30-day window:
    # goldstein['sma-30'] = pd.rolling_mean(goldstein['GoldAverage'], 30)
    # # The first 30 entries in the rolling_mean become NaN, so...
    # grm = goldstein['sma-30'].dropna()
    #--------------------------------------------------------------------#


    test_sample = pd.DataFrame(goldstein['GoldAverage'])
    test_sample.index = pd.to_datetime(test_sample.index)
    test_sample.columns = ['Goldstein daily average']

    plot_sample = pd.DataFrame(goldstein['GoldAverage'][-230:])
    plot_sample.index = pd.to_datetime(plot_sample.index)
    plot_sample.columns = ['Goldstein daily average']

    # Creates the forcasting model using Autoregressive Moving Average (ARMA):
    #model = sm.tsa.ARMA(test_sample,(12, 0)).fit() # 12 Lags seems to be enough to get an accurate prediction.
    tries = 0
    success = False
    while tries < 6 and success is False:
        try:
            model = sm.tsa.ARMA(test_sample,(12, tries)).fit() # 12 Lags seems to be enough to get an accurate prediction.
            success = True
        except:
            tries += 1

    if success is False:
        return False

    # Creates the prediction based on the proceeding two weeks through one week into the future:
    prediction = model.predict(twoweeksago, str(nextweek), dynamic = False)

    # The fitting of the prediction to the actual looked about 1 day off, so...
    prediction = prediction.shift(-1)
    # print prediction

    # Finds the average of the Goldstein scores for the coming week:
    predicts = round(prediction.ix[tomorrow:str(nextweek)].mean(), 1)

    suggestion = round(((predicts - 1) * -1), 1)

    # make sure suggestion and prediction are never the same
    if suggestion == predicts:
        suggestion += .1

    gsDescription   = random.choice(gsCodes.loc[suggestion].values[0].split(';')).strip()
    prediction_text = random.choice(gsCodes.loc[predicts].values[0].split(';')).strip()

    if predicts == 1.0:
        gsDescription = "You're doing great. Stay the course."

    print '==================='
    print name + "'s Forecast: ", predicts, prediction_text
    print name + "'s Suggested Action: ", (predicts - 1) * -1
    print "Suggested actions for the coming week:\n" + gsDescription
    print '==================='

    graph_file = draw_graph(plot_sample, prediction, predicts, suggestion, name, gsDescription, leader)

    image = compose.draw(name=name, country=country, suggestion=gsDescription, prediction=prediction_text, graph_file=graph_file)
    final_image = 'images/' + name + '_' + tomorrow + '.png'
    image.save(final_image, 'PNG')

    return {'username': username, 'suggestion': gsDescription, 'image': final_image}

##############################################################

def draw_graph(plot_sample, prediction, predicts, suggestion, name, gsDescription, leader):

    country = leader['display_country'].values[0]

    last_index_plot = plot_sample.index[-1]
    prediction = prediction.loc[last_index_plot:]

    fontsize = 16

    startdate = datetime.date(plot_sample.index[0])
    enddate = dt.date.today() + dt.timedelta(days = 8)
    daterange = [startdate + dt.timedelta(days = x) for x in range(0, (enddate - startdate).days)]

    plt.rcParams['ytick.major.pad'] = '18'
    plt.rcParams['xtick.major.pad'] = '18'
    # plt.rcParams['ylabel.major.pad'] = '18'

    #fig = plt.figure(figsize = (13.5, 4))
    fig = plt.figure(figsize = (3.51*4, 4))
    ax = fig.add_subplot(111)
    ax.set_frame_on(False)
    ax.yaxis.labelpad = 18

    plt.tick_params(axis = "both", which = "both", bottom = "off", top = "off",
                    labelbottom = "on", left = "off", right = "off", labelleft = "on")

    yticks = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]
    plt.yticks(yticks,
        alpha = 0.5,
        fontproperties = font,
        fontsize = fontsize,
        rotation = 0,
        color = 'white')

    ax.get_yaxis().set_visible(False)

    # blue = (26/255., 37/255., 229/255.)
    red =(224/255., 86/255., 74/255.)
    green = (148/255., 166/255., 58/255.)
    gray = (170/255., 170/255., 170/255.)
    lightgray = (228/255., 228/255., 228/255.)
    darkgray = (129/255., 129/255., 129/255.)

    # Draws the horizontal dashed lines:
    horizontal_lines = np.linspace(-10, 10, num = 9)
    for y in horizontal_lines:
        plt.plot(daterange,
                [y] * len(daterange),
                '--',
                dashes = [12, 8],
                linewidth = 1.0,
                color = darkgray,
                clip_on = False,
                label = None)

    # Plot the main data:
    plot_sample.plot(kind = 'line',
            ax = ax,
            color = lightgray,
            linewidth = 5.0,
            ylim = (-10, 10),
            legend = False,
            zorder = 10)

    prediction.plot(
            kind = 'line',
            ax = ax,
            color = red,
            dashes = [12, 4],
            linewidth = 5.0,
            label = 'prediction',
            zorder = 11,
            legend = False,
            grid = False)

    plt.ylabel('Clifton-Lavigne-Goldstein Scale',
            color = gray,
            fontproperties = font,
            fontsize = fontsize - 2)

    plt.arrow(dt.date.today() + dt.timedelta(days = 3), 12, 0.0, -12 + predicts + 2,
            length_includes_head = True,
            facecolor = red,
            edgecolor = red,
            antialiased = True,
            head_width = 2.5,
            head_length = 1.5,
            linewidth = 4.0,
            clip_on = False,
            zorder = 99)

    # red horizontal line
    plt.axhline(
            y = 12,
            xmin = -0.085,
            xmax = 0.976,
            clip_on = False,
            alpha = 1.0,
            linewidth = 4.0,
            color = red)

    # tiny verticle line
    plt.axvline(
            x = startdate - dt.timedelta(days = 18),
            #x = startdate,
            ymin = 1.1,
            ymax = 1.15,
            clip_on = False,
            alpha = 0.0,
            linewidth = 4.0,
            color = red)

    # green horizontal line at 1.0
    plt.axhline(
            y = 1,
            xmin = 0.0025,
            # xmin = 0,
            # xmax = 0.985,
            xmax = 0.9925,
            clip_on = False,
            alpha = 1.0,
            linewidth = 5.0,
            color = green)

    # below is an alternate way to drawy lines
    #ax2 = plt.axes([0,0,1,1], axisbg=(1,1,1,0))
    #x,y = np.array([[0.0, 1.0], [0.0, 0.0]])
    #line = lines.Line2D(x, y, lw=5., color='r', alpha=0.4)
    #ax2.add_line(line)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

    plt.xticks(
            fontproperties = font,
            fontsize = fontsize,
            color = darkgray,
            rotation = 0,
            horizontalalignment='center')

    plt.savefig(image_name(name),
            bbox_inches = 'tight',
            dpi = 300,
            transparent = True)

    return image_name(name)

##############################################################

def image_name(name):
    return IMG_PATH + name + '_prediction.png'

##############################################################

if __name__ == '__main__' :
    scrape()
    create_schedule()
