# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 15:28:43 2020

@author: Chuck
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:00:19 2020

@author: Chuck
"""

import tweepy
import pandas as pd
from datetime import datetime

user_id = ["@realDonaldTrump","@JoeBiden"]

api_key = 'Bip1Xolcap3jMc55NZavYw4Ow'
api_secret = 'WVOn8KqZVyS8j4MPXj2xAkSkGIp2ohMt6p2n0CwuSxCMdQGIyQ'
# access_token = '1082111439769264128-7tfopo0aLHtdWPbEgMzY0yhQyUJ8EI'
# access_token_secret = 'yDcFR6JU8rxDcI13H40k669Ut75n5nkr6NO8l7XoOuM0w'

auth = tweepy.AppAuthHandler(api_key, api_secret)

# auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth,wait_on_rate_limit=True)

data = []

for i in range(0,len(user_id)):
    if user_id[i] == '@realDonaldTrump':
        tweets = tweepy.Cursor(api.user_timeline, id=user_id[i],tweet_mode='extended').items(5000)
    else:
        tweets = tweepy.Cursor(api.user_timeline, id=user_id[i],tweet_mode='extended').items(1000)
    for x in tweets:
        data.append({'user_id':user_id[i],'tweet_date':x.created_at,'tweet':x.full_text,'retweets_count':x.retweet_count})
    
df = pd.DataFrame(data)

csv = 'C:\\Users\\Charles\\Downloads\\twitter_data_users_{}.csv'.format(datetime.today().date())

df.to_csv(csv,index=False)