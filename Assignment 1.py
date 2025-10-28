import pandas as pd
import numpy as np
import nltk

fox_health = pd.read_csv('foxnewshealth.txt', names=['text'])
cnn_health = pd.read_csv('cnnhealth.txt', names=['text'])
fox_health['source'] = 'Fox News'
cnn_health['source'] = 'CNN'
health_news_df = pd.concat([fox_health, cnn_health], ignore_index=True)
print(health_news_df.head())
