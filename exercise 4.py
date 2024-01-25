import jieba
import pandas as pd
import spacy
import gensim

nlp = spacy.load("zh_core_web_md")


def cn_word_tokenize(doc, stw):
    tokens = jieba.cut(doc)
    return [el for el in tokens if el not in stw and len(el) > 1]


with open('cn_stopwords.txt', mode='r', encoding='utf8') as f:
    cn_stw = [stw.strip() for stw in f.readlines()]

df = pd.read_csv('example_data_dynamic.csv')


dictionary = gensim.corpora.Dictionary(df['corpus'].apply(lambda x: x.split()))
corpus = [dictionary.doc2bow(text) for text in
          df['corpus'].apply(lambda x: x.split())]
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
time_slices = df['month'].value_counts().sort_index().tolist()
dtm_model = gensim.models.ldaseqmodel.LdaSeqModel(corpus=corpus, time_slice=time_slices, num_topics=5, id2word=dictionary)
dtm_model.print_topics(time=0)
