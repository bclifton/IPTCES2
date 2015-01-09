#!/usr/bin/env python
import sys
import os
import glob
import json
import datetime as dt

import numpy as np
import pandas as pd
from pandas import *
from pandas.io.json import json_normalize

from bigquery import get_client

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import statsmodels.api as sm
from statsmodels.tsa import *
tsa = sm.tsa

##############################################
# BigQuery variables
with open('config/settings.json') as json_data:
    settings = json.load(json_data)
client = get_client(settings['PROJECT_ID'], service_account=settings['SERVICE_ACCOUNT'], private_key=settings['KEY'], readonly=True)

# Font variables
FONT_LOCATION = '/Users/brianclifton/Desktop/Fonts 2/A/LL Akkurat Family OTStd/AkkuratStd-Regular.otf'
font = fm.FontProperties(fname=FONT_LOCATION)

# Path variables
IMG_PATH = 'graphs/'
CSV_PATH = 'csv/'

# File variables
GS_PATH = 'assets/goldstein_suggestions.tsv'

# Time variables
tomorrow = str(dt.date.today()) 
twoweeksago = str(dt.date.today() - dt.timedelta(days = 14))
nextweek = dt.date.today() + dt.timedelta(days = 8)


# Temp variables
leaders = ['Ellen Johnson Sirleaf']
date_start = '19790101'
date_end = '20150108'
country = 'LBR'



##############################################################

def get_leaders_from_bigquery():
	
	print date_start, date_end

	# This will have to be modified:
	for leader in leaders:

		job_id, results = client.query('SELECT SQLDATE, Actor1Name, Actor1CountryCode, GoldsteinScale, NumMentions FROM [gdelt-bq:full.events] WHERE Actor1Name CONTAINS \"'+ leader +'\" AND Actor1CountryCode=\"'+ country +'\" AND SQLDATE>'+ date_start +' AND SQLDATE<'+ date_end +' IGNORE CASE;')

		complete, row_count = client.check_job(job_id)
		results = client.get_query_rows(job_id)

		df = json_normalize(results)

		name = '_'.join(leader.split(' '))

		df.to_csv(CSV_PATH + name + '.csv', encoding='utf-8')
		
		print 'Saved: ' + name + '.csv'


##############################################################

def open_files():

	gsCodes = pd.read_csv(GS_PATH, index_col='code', sep='\t')

	for f in glob.glob(CSV_PATH + '*.csv'):
	    perform_analysis(f, gsCodes)

##############################################################

def perform_analysis(data, gsCodes):
	df = pd.read_csv(data, index_col='SQLDATE', parse_dates=True)

	name = df['Actor1Name'][0].title()

	# These steps incorporate the number of mentions a Goldstein score is associated with, reducing the impact of error in the event encoding, making the average better reflect the event's presence in the GDELT.
	df['GoldMentions'] = df['GoldsteinScale'] * df['NumMentions']
	goldstein = df.groupby([df.index.date]).agg({'GoldMentions': np.sum, 'NumMentions': np.sum})
	goldstein['GoldAverage'] = goldstein['GoldMentions'] / goldstein['NumMentions']

	full_daterange = pd.date_range(start = min(df.index), end = max(df.index))

	# ffill() takes care of days that do not have a entry / Goldstein score in GDELT:
	goldstein = goldstein.reindex(full_daterange).ffill() 

	# Creates a rolling_mean using a 30-day window:
	goldstein['sma-30'] = pd.rolling_mean(goldstein['GoldAverage'], 30)

	# The first 30 entries in the rolling_mean become NaN, so...
	grm = goldstein['sma-30'].dropna()

	test_sample = pd.DataFrame(grm[-400:]) # 200 entries seems to be enough to determine stationarity.
	test_sample.index = pd.to_datetime(test_sample.index)
	test_sample.columns = ['Goldstein daily average']

	plot_sample = pd.DataFrame(grm[-200:])
	plot_sample.index = pd.to_datetime(plot_sample.index)
	plot_sample.columns = ['Goldstein daily average']

	# Creates the forcasting model using Autoregressive Moving Average (ARMA):
	model = sm.tsa.ARMA(test_sample,(12, 0)).fit() # 12 Lags seems to be enough to get an accurate prediction.

	# Creates the prediction based on the proceeding two weeks through one week into the future:
	print twoweeksago
	print nextweek

	prediction = model.predict(twoweeksago, str(nextweek), dynamic = False)

	# The fitting of the prediction to the actual looked about 1 day off, so...
	prediction = prediction.shift(-1)

	# Finds the average of the Goldstein scores for the coming week:
	predicts = round(prediction.ix[tomorrow:str(nextweek)].mean(), 1)

	suggestion = round(((predicts - 1) * -1), 1)
	gsDescription = gsCodes.loc[suggestion].values[0]

	print '==================='
	print name.replace('_', ' ') + "'s Forecast: ", predicts
	print name.replace('_', ' ') + "'s Suggested Action: ", (predicts - 1) * -1
	print "Suggested actions for the coming week:\n" + gsDescription
	print '==================='

	draw_graph(plot_sample, prediction, predicts, suggestion, name, gsDescription)

##############################################################

def draw_graph(plot_sample, prediction, predicts, suggestion, name, gsDescription):

	startdate = datetime.date(plot_sample.index[0])
	enddate = dt.date.today() + dt.timedelta(days = 8)
	daterange = [startdate + dt.timedelta(days = x) for x in range(0, (enddate - startdate).days)]


	fig = plt.figure(figsize = (14, 8))
	ax = fig.add_subplot(111)
	ax.set_frame_on(False)
	plt.tick_params(axis = "both", which = "both", bottom = "off", top = "off",  
	                labelbottom = "on", left = "off", right = "off", labelleft = "on") 

	yticks = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]
	plt.yticks(yticks, alpha = 0.5, fontproperties = font, rotation = 0)
	ax.get_yaxis().set_visible(False)

	plt.xticks(alpha = 0.5, fontproperties = font, rotation = 0)

	blue = (26/255.,37/255.,229/255.)
	green = (12/255., 170/255.,12/255.)

	# Draws the horizontal dashed lines:
	horizontal_lines = np.linspace(-10, 10, num = 9)
	for y in horizontal_lines:
	    plt.plot(daterange, 
	         [y] * len(daterange),
	         '--', 
	         lw = 0.5, 
	         color = 'black',
	         alpha = 0.3,
	         label = None)
	    
	plt.plot(daterange,
	         [1] * len(daterange),
	         alpha = 0.57,
	         color = green)

	# Plot the main data:
	plot_sample.plot(kind = 'line', 
	            ax = ax, 
	            color = blue, 
	            ylim = (-10, 10),
	            legend = False)

	prediction.plot(kind = 'line', 
	                ax = ax,
	                color = 'red', 
	                label = 'prediction',
	                legend = False,
	                grid = False)
	# Title
	plt.text(startdate,
	     13,
	     name.replace('_', ' ') + ' (' + country + ')' + ': Goldstein Trend and Prediction',
	     fontsize = 24,
	     fontproperties = font,
	     color = 'black',
	     ha = 'left')

	# Legend
	plt.text(str(nextweek),
	     9,
	     "Trend", 
	     fontsize = 14,
	     fontproperties = font,
	     color = blue,
	     ha = "right")
	plt.text(str(nextweek),
	     8,
	     "Target", 
	     fontsize = 14,
	     fontproperties = font,
	     color = green,
	     ha = "right")
	plt.text(str(nextweek),
	     7,
	     "Prediction", 
	     fontsize = 14,
	     fontproperties = font,
	     color = "red",
	     ha = "right")

	# Prediction Number
	plt.text(str(nextweek),
	     predicts, 
	     nextweek.strftime("%b. %d, %Y") + "", 
	     fontsize = 14,
	     fontproperties = font,
	     color = "red",
	     ha = "left")
	plt.text(str(nextweek),
	     predicts -2, 
	     "  " + str(predicts), 
	     fontsize = 36,
	     fontproperties = font,
	     color = "red",
	     ha = "left")  

	# Credits
	plt.text(startdate, 
	     -15,
	     "Original data provided by GDELT (http://gdeltproject.org/)"
	     "\nData source: http://storage.googleapis.com/gdelt_bc/"+ name +".csv"  
	     "\nAuthor: Brian Clifton (briancliftonstudio.com / @BrianClifton1)"
	     "\n\n___________________________________________________________"
	     "_______________________________________________________________",
	     fontsize = 10,
	     fontproperties = font,
	     alpha = 0.5)

	# Suggested Actions
	plt.text(startdate,
	         -17,
	         "Suggested actions for the coming week:",
	         fontsize = 20,
	         fontproperties = font,
	         color='black')
	plt.text(startdate,
	         -20,
	         "" + gsDescription + ""
	         "\n(Goldstein Scale "+ str(suggestion) + ")",
	         fontsize = 24,
	         fontproperties = font,
	         color=green)

	plt.savefig(IMG_PATH + name + '_prediction.png', bbox_inches = 'tight', dpi = 300)

##############################################################

def main():
	get_leaders_from_bigquery()


	open_files()







##############################################################

if __name__ == '__main__' :
    main()













