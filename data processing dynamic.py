from datetime import datetime, timedelta
import jieba
import pandas as pd
import re
import spacy

nlp = spacy.load("zh_core_web_md")


def generate_date_list(start_date, end_date):
    """
    generate a list of dates (YYYY-MM-DD) from a date to another
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_list = []
    current_date = start
    while current_date <= end:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    return date_list


def cn_word_tokenize(doc, stw):
    clean_doc = re.sub(r'[^\u4e00-\u9fa5]', '', doc)
    tokens = jieba.cut(clean_doc)
    tokens = [el for el in tokens if el not in stw and len(el) > 1]
    return tokens


with open('cn_stopwords.txt', mode='r', encoding='utf8') as f:
    cn_stw = [stw.strip() for stw in f.readlines()]
df = pd.read_csv('2019xinmin.csv')
date_list = generate_date_list('2019-01-01', '2019-12-31')
df = df[df['date'].isin(date_list)]
df = df[df['article'].str.len() > 50]
df['tokenized_article'] = df['article'].apply(lambda x: cn_word_tokenize(x, cn_stw))
df['corpus'] = df['tokenized_article'].apply(lambda x: ' '.join(x))
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df.to_csv("example_data_dynamic.csv", index=False)
