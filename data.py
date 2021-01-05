# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 08:12:24 2020

@author: Charles
"""
# new model

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
# from sklearn.linear_model import SGDClassifier
import pandas as pd
import re, string
from nltk.corpus import twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
# from nltk import FreqDist
# from nltk.tokenize import word_tokenize
# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
* Title: Confusion Matrix Visualization
* Author: Dennis T
* Date: Oct 11 2019
* Code version: n/a
* Availability: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
"""

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    box_labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_labels,group_counts)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
        
"""
* Title: Visualising Top Features in Linear SVM with Scikit Learn and Matplotlib
* Author: Aneesha Bakharia
* Date: Jan 31 2016
* Code version: n/a
* Availability: https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
"""
        
def plot_coefficients(classifier, feature_names, top_features=20):
 coef = classifier.coef_.ravel()
 top_positive_coefficients = np.argsort(coef)[-top_features:]
 top_negative_coefficients = np.argsort(coef)[:top_features]
 top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
 # create plot
 plt.figure(figsize=(15, 5))
 colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
 plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
 feature_names = np.array(feature_names)
 plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
 plt.show()

load_more_data = 'C:\\Users\\Charles\\Desktop\\training.1600000.processed.noemoticon.csv'

large_df = pd.read_csv(load_more_data,names=["target", "id", "date", "flag",'user','text'],encoding='ISO-8859-1')

large_df = large_df[['text','target']]

pos_df_large = large_df[large_df['target'] == 4]

pos_df_large['target'] = 1

neg_df_large = large_df[large_df['target'] == 0]

large_df = pd.concat([pos_df_large,neg_df_large]).reset_index(drop=True)

stop_words = stopwords.words('english')

def clean_str(tweet):
    
    cleaned_tokens = []
    
    for token in tweet.split():
        
        token = re.sub(r"http\S+",'', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        token = token.replace("b'",'')
        token = token.replace('\\n','')
        token = token.replace("''",'')
        token = token.replace('http','')
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token)
        
        if len(token) >= 3 and token not in string.punctuation and token.lower() not in stop_words and token.lower() not in ['realdonaldtrump','joebiden','trump','biden','joe','donald','election2020','republican','democrat','donaldtrump'] and token.lower() != "n't" and not token.lower().startswith('//t.co'):
            cleaned_tokens.append(token.lower())
    return ' '.join(cleaned_tokens)

positive_tweets = twitter_samples.strings('positive_tweets.json')

pos_df = pd.DataFrame(positive_tweets,columns=['text'])
pos_df['target'] = 1

negative_tweet = twitter_samples.strings('negative_tweets.json')
neg_df = pd.DataFrame(negative_tweet,columns=['text'])
neg_df['target'] = 0

dataset_df = pd.concat([pos_df,neg_df,large_df]).reset_index(drop=True)

count_vect = CountVectorizer(stop_words='english',ngram_range=(1,3),analyzer='word',token_pattern=r'\w{2,}')

tfidf = TfidfTransformer()

x = []

for i in dataset_df['text']:
    
    x.append(clean_str(i))
 
X_train, X_test, y_train, y_test = train_test_split(x, dataset_df['target'], random_state = 0)

x_count = count_vect.fit_transform(X_train)

X_train_tfidf = tfidf.fit_transform(x_count)

svm_classifier_test = LinearSVC().fit(X_train_tfidf,y_train)

print(svm_classifier_test.score(count_vect.transform(X_test),y_test))

#---------------------------------------------------------------- Confusion Matrix

from sklearn.metrics import confusion_matrix

y_pred = svm_classifier_test.predict(count_vect.transform(X_test))

cf_matrix = confusion_matrix(y_test, y_pred)

print(cf_matrix)

labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['0','1']
make_confusion_matrix(cf_matrix, 
                      group_names=labels,
                      categories=categories, 
                      cmap='binary')


plot_coefficients(svm_classifier_test, count_vect.get_feature_names())

#----------------------------------------------------------------

x_train = x
y_train = dataset_df['target']

x_count = count_vect.fit_transform(x_train)

X_train_tfidf = tfidf.fit_transform(x_count)

svm_classifier = LinearSVC().fit(X_train_tfidf,y_train)

#----------------------------------------------------------------

df_1 = pd.read_csv('C:\\Users\\Charles\\Downloads\\twitter_data_users_2020-10-05.csv')

dataset = []

for i in df_1['tweet']:
    
    dataset.append(clean_str(i))
    
#for i in df['Unified_Product_Name']:
#    
#    appname.append(clean_str(i))

results_set = []

for i in range(0,len(dataset)):
    
    results = svm_classifier.predict(count_vect.transform([dataset[i]]))[0]
    
    results_set.append(results)

results_df = pd.DataFrame(results_set,columns=['category'])

dataset_final = df_1.join(results_df)

csv_save = 'C:\\Users\\Charles\\Downloads\\twitter_data_classified_v2.csv'

dataset_final.to_csv(csv_save,index=False)

#-----------------------------------------------------------------

df_1_tweets = pd.read_csv('C:\\Users\\Charles\\Desktop\\twitter_data_2020-09-19.csv')
df_2_tweets = pd.read_csv('C:\\Users\\Charles\\Downloads\\twitter_data_2020-09-12.csv')
df_3_tweets = pd.read_csv('C:\\Users\\Charles\\Desktop\\twitter_data_2020-09-26.csv')
df_4_tweets = pd.read_csv('C:\\Users\\Charles\\Desktop\\twitter_data_2020-10-03.csv')
df_5_tweets = pd.read_csv('C:\\Users\\Charles\\Desktop\\twitter_data_2020-10-10.csv')
df_6_tweets = pd.read_csv('C:\\Users\\Charles\\Desktop\\twitter_data_2020-10-19.csv')
df_7_tweets = pd.read_csv('C:\\Users\\Charles\\Desktop\\twitter_data_2020-10-25.csv')
df_8_tweets = pd.read_csv('C:\\Users\\Charles\\Desktop\\twitter_data_2020-10-31.csv')
df_9_tweets = pd.read_csv('C:\\Users\\Charles\\Desktop\\twitter_data_2020-11-07.csv')

df_merge = pd.concat([df_1_tweets,df_2_tweets,df_3_tweets,df_4_tweets,df_5_tweets,df_6_tweets,df_7_tweets,df_8_tweets,df_9_tweets]).reset_index(drop=True)

dataset = []

for i in df_merge['tweet']:
    
    dataset.append(clean_str(i))
    
results_set = []

for i in range(0,len(dataset)):
    
    results = svm_classifier.predict(count_vect.transform([dataset[i]]))[0]
    
    results_set.append(results)

results_df = pd.DataFrame(results_set,columns=['category'])

dataset_final = df_merge.join(results_df)

csv_save = 'C:\\Users\\Charles\\Downloads\\twitter_data_classified_non_users_final.csv'

dataset_final.to_csv(csv_save,index=False)

#-----------------------------------------------------------------

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

df_1 = pd.read_csv('C:\\Users\\Charles\\Downloads\\twitter_data_users_2020-10-05.csv')
df_2 = pd.read_csv('C:\\Users\\Charles\\Downloads\\twitter_data_users_2020-10-25.csv')
df_3 = pd.read_csv('C:\\Users\\Charles\\Downloads\\twitter_data_users_2020-10-31.csv')
df_4 = pd.read_csv('C:\\Users\\Charles\\Downloads\\twitter_data_users_2020-11-07.csv')

df_merge = pd.concat([df_1,df_2,df_3,df_4]).reset_index(drop=True)

df_merge = df_merge.drop(['retweets_count'],axis=1)

df_merge = df_merge.drop_duplicates()

df_merge = df_merge.reset_index(inplace=True)

dataset = []

for i in df_merge['tweet']:
    
    dataset.append(clean_str(i))
    
results_set = []

for i in range(0,len(dataset)):
    
    results = svm_classifier.predict(count_vect.transform([dataset[i]]))[0]
    
    results_set.append(results)

results_df = pd.DataFrame(results_set,columns=['category'])

dataset_final = df_merge.join(results_df)

csv_save = 'C:\\Users\\Charles\\Downloads\\twitter_data_classified_users_final.csv'

dataset_final.to_csv(csv_save,index=False)

search = ["#election2020","#republican","#democrat",'#donaldtrump','#joebiden']

dataset_trump = []
dataset_biden = []
dataset_election = []

df_trump = df_merge[df_merge['tweet_category'] == '#donaldtrump']
df_biden = df_merge[df_merge['tweet_category'] == '#joebiden']
df_election = df_merge[df_merge['tweet_category'] == '#election2020']

for i in df_trump['tweet']:
    
    cleaned = clean_str(i)
    
    dataset_trump.append(cleaned.split())
    
for j in df_biden['tweet']:
    
    clean = clean_str(j)
    
    dataset_biden.append(clean.split())
    
for k in df_election['tweet']:
    
    clean = clean_str(k)
    
    dataset_election.append(clean.split())
    
sid = SentimentIntensityAnalyzer()
trump_pos_word_list=[]
trump_neg_word_list=[]

biden_pos_word_list=[]
biden_neg_word_list=[]

election_pos_word=[]
election_neg_word=[]

for x in dataset_trump:
    for word in x:
        if (sid.polarity_scores(word)['compound']) >= 0.5:
            trump_pos_word_list.append(word)
        elif (sid.polarity_scores(word)['compound']) <= -0.5:
            trump_neg_word_list.append(word)

for y in dataset_biden:        
    for word in y:
        if (sid.polarity_scores(word)['compound']) >= 0.5:
            biden_pos_word_list.append(word)
        elif (sid.polarity_scores(word)['compound']) <= -0.5:
            biden_neg_word_list.append(word)
            
for z in dataset_election:        
    for word in z:
        if (sid.polarity_scores(word)['compound']) >= 0.5:
            election_pos_word.append(word)
        elif (sid.polarity_scores(word)['compound']) <= -0.5:
            election_neg_word.append(word)
        
df_pos_trump = pd.DataFrame(trump_pos_word_list,columns=['words'])
df_neg_trump = pd.DataFrame(trump_neg_word_list,columns=['words'])

df_pos_biden = pd.DataFrame(biden_pos_word_list,columns=['words'])
df_neg_biden = pd.DataFrame(biden_neg_word_list,columns=['words'])

df_pos_election = pd.DataFrame(election_pos_word,columns=['words'])
df_neg_election = pd.DataFrame(election_neg_word,columns=['words'])


df_pos_trump.to_csv('C:\\Users\\Charles\\Downloads\\trump_#_pos_words.csv',index=False)
df_neg_trump.to_csv('C:\\Users\\Charles\\Downloads\\trump_#_neg_words.csv',index=False)

df_pos_biden.to_csv('C:\\Users\\Charles\\Downloads\\biden_#_pos_words.csv',index=False)
df_neg_biden.to_csv('C:\\Users\\Charles\\Downloads\\biden_#_neg_words.csv',index=False)

df_pos_election.to_csv('C:\\Users\\Charles\\Downloads\\election_pos_words.csv',index=False)
df_neg_election.to_csv('C:\\Users\\Charles\\Downloads\\election_neg_words.csv',index=False)