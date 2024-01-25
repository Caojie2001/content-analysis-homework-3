import jieba
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, \
    TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import ward, dendrogram, fcluster
import sklearn
import spacy
import gensim

nlp = spacy.load("zh_core_web_md")


def cn_word_tokenize(doc, stw):
    tokens = jieba.cut(doc)
    return [el for el in tokens if el not in stw and len(el) > 1]


def plotSilhouette(n_clusters, X):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    silhouette_avg = sklearn.metrics.silhouette_score(X, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = sklearn.metrics.silhouette_samples(X,
                                                                  cluster_labels)

    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[
            cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        cmap = matplotlib.cm.get_cmap("nipy_spectral")
        color = cmap(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    pca = PCA(n_components=2).fit(newsgroupsTFVects.toarray())
    cmap = matplotlib.cm.get_cmap("nipy_spectral")
    reduced_data = pca.transform(newsgroupsTFVects.toarray())
    colors = cmap(float(i) / n_clusters)
    ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], marker='.', s=30, lw=0,
                alpha=0.7,
                c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    projected_centers = pca.transform(centers)
    # Draw white circles at cluster centers
    ax2.scatter(projected_centers[:, 0], projected_centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(projected_centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.savefig('silhouette.png')
    plt.show()
    print(
        "For n_clusters = {}, The average silhouette_score is : {:.3f}".format(
            n_clusters, silhouette_avg))


with open('cn_stopwords.txt', mode='r', encoding='utf8') as f:
    cn_stw = [stw.strip() for stw in f.readlines()]

df = pd.read_csv('example_data.csv')

dictionary = gensim.corpora.Dictionary(df['corpus'].apply(lambda x: x.split()))
corpus = [dictionary.doc2bow(text) for text in
          df['corpus'].apply(lambda x: x.split())]
gensim.corpora.MmCorpus.serialize('news.mm', corpus)
newsmm = gensim.corpora.MmCorpus('news.mm')
newslda = gensim.models.ldamodel.LdaModel(corpus=newsmm, id2word=dictionary,
                                          num_topics=2, alpha='auto',
                                          eta='auto')
ldaDF = pd.DataFrame({
    'name': df['news_title'],
    'topics': [newslda[dictionary.doc2bow(l)] for l in
               df['corpus'].apply(lambda x: x.split())]
})
topicsProbDict = {i: [0] * len(ldaDF) for i in range(newslda.num_topics)}
for index, topicTuples in enumerate(ldaDF['topics']):
    for topicNum, prob in topicTuples:
        topicsProbDict[topicNum][index] = prob
for topicNum in range(newslda.num_topics):
    ldaDF['topic_{}'.format(topicNum)] = topicsProbDict[topicNum]
# ldaDF.to_csv("example_lda.csv", index=False)
ldaDFV = ldaDF[:10][['topic_%d' % x for x in range(2)]]
ldaDFVisN = ldaDF[:10][['name']]
ldaDFVis = ldaDFV.values
ldaDFVisNames = ldaDFVisN.values
'''
N = 10
ind = np.arange(N)
K = newslda.num_topics
ind = np.arange(N)
width = 0.5
plots = []
height_cumulative = np.zeros(N)

for k in range(K):
    color = plt.cm.coolwarm(k/K, 1)
    if k == 0:
        p = plt.bar(ind, ldaDFVis[:, k], width, color=color)
    else:
        p = plt.bar(ind, ldaDFVis[:, k], width, bottom=height_cumulative, 
        color=color)
    height_cumulative += ldaDFVis[:, k]
    plots.append(p)


plt.ylim((0, 1))
plt.ylabel('Topics')

plt.title('Topics in Press Releases')
plt.xticks(ind+width/2, ldaDFVisNames, rotation='vertical')

plt.yticks(np.arange(0, 1, 10))
topic_labels = ['Topic #{}'.format(k) for k in range(K)]
plt.legend([p[0] for p in plots], topic_labels, loc='center left', 
frameon=True,  bbox_to_anchor = (1, .5))
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.tight_layout()
plt.savefig('lda.png')
plt.show()
'''
'''
topic_labels = ['Topic #{}'.format(k) for k in range(2)]
plt.pcolor(ldaDFVis, norm=None, cmap='Blues')
plt.yticks(np.arange(ldaDFVis.shape[0]) + 0.5, ldaDFVisNames)
plt.xticks(np.arange(ldaDFVis.shape[1]) + 0.5, topic_labels)

plt.gca().invert_yaxis()

plt.xticks(rotation=90)

plt.colorbar(cmap='Blues')
plt.tight_layout()
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.savefig('heat.png')
plt.show()
'''
topicsDict = {}
for topicNum in range(newslda.num_topics):
    topicWords = [w for w, p in newslda.show_topic(topicNum)]
    topicsDict['Topic_{}'.format(topicNum)] = topicWords

wordRanksDF = pd.DataFrame(topicsDict)
topic1_df = pd.DataFrame(newslda.show_topic(1, topn=50))
'''
plt.figure()
topic1_df.plot.bar(legend = False)
plt.title('Probability Distribution of Words, Topic 1')
plt.savefig('probability_distribution_words_1.png')
plt.show()
'''
newslda1 = gensim.models.ldamodel.LdaModel(corpus=newsmm, id2word=dictionary, num_topics=10, eta = 0.00001)
newslda2 = gensim.models.ldamodel.LdaModel(corpus=newsmm, id2word=dictionary, num_topics=10, eta = 0.9)
topic11_df = pd.DataFrame(newslda1.show_topic(1, topn=50))
topic21_df = pd.DataFrame(newslda2.show_topic(1, topn=50))

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
topic11_df.plot.bar(legend = False, ax = ax1, title = '$\eta$  = 0.00001')
topic21_df.plot.bar(legend = False, ax = ax2, title = '$\eta$  = 0.9')
plt.savefig('distribution_contrast.png')
plt.show()
'''
lower n leads to a flatter distribution of words.
'''
