# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:00:19 2020

@author: Chuck
"""

import tweepy
import pandas as pd
from datetime import datetime

# user_id = ["@realDonaldTrump","@JoeBiden"]

num_days = 7

# base = datetime.datetime.today()
# date_list = [base - datetime.datetime.timedelta(days=x) for x in range(num_days)]

api_key = 'Bip1Xolcap3jMc55NZavYw4Ow'
api_secret = 'WVOn8KqZVyS8j4MPXj2xAkSkGIp2ohMt6p2n0CwuSxCMdQGIyQ'
# access_token = '1082111439769264128-7tfopo0aLHtdWPbEgMzY0yhQyUJ8EI'
# access_token_secret = 'yDcFR6JU8rxDcI13H40k669Ut75n5nkr6NO8l7XoOuM0w'

auth = tweepy.AppAuthHandler(api_key, api_secret)

# auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth,wait_on_rate_limit=True)

data = []

search = ["#election2020","#republican","#democrat",'#donaldtrump','#joebiden']

for i in range(0,len(search)):
    tweets = tweepy.Cursor(api.search,q=search[i]+" -filter:retweets",lang="en",tweet_mode='extended').items(5000)
    for x in tweets:
        data.append({'tweet_category':search[i],'tweet_date':x.created_at,'tweet':x.full_text,'retweets_count':x.retweet_count})
    
df = pd.DataFrame(data)

csv = 'C:/Users/Charles/Desktop/twitter_data_{}.csv'.format(datetime.today().date())

df.to_csv(csv,index=False)