import datetime
import pickle
import json
from twython import Twython

with open('config/settings.json') as json_data:
    settings = json.load(json_data)

twitter = Twython(settings['APP_KEY'], settings['APP_SECRET'], settings['OAUTH_TOKEN'], settings['OAUTH_TOKEN_SECRET'])

def send_tweet(username, suggestion, filename):
    message = "Dear {}, this week we suggest you: {}".format(username, suggestion)
    message = message[0:117]
    photo = open(filename, 'rb')
    try:
        twitter.update_status_with_media(status=message, media=photo)
        print 'TWEETED: ' + message
        return True
    except:
        print 'Failed to tweet to ' + username
        return False


if __name__ == '__main__' :
    with open('tweet_schedule.p', 'rb') as f:
        tweets = pickle.load(f)

    today = datetime.date.today()
    start_of_day = datetime.datetime(today.year, today.month, today.day, 9, 0)

    for tweet in tweets:
        tweet_time = start_of_day + datetime.timedelta(seconds=tweet['seconds'])
        if 'sent' not in tweet and datetime.datetime.now() > tweet_time:
            tweet['sent'] = True
            send_tweet(tweet['username'], tweet['suggestion'], tweet['image'])

    with open('tweet_schedule.p', 'wb') as f:
        pickle.dump(tweets, f)
