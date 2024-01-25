import jieba
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, \
    TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sklearn
import spacy

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

ng_count_vectorizer = CountVectorizer()
news_groups_vects = ng_count_vectorizer.fit_transform(df['corpus'])

newsgroupsTFTransformer = TfidfTransformer().fit(news_groups_vects)
newsgroupsTF = newsgroupsTFTransformer.transform(news_groups_vects)

ngTFVectorizer = TfidfVectorizer(max_df=0.5, max_features=1000, min_df=3,
                                 norm='l2')
newsgroupsTFVects = ngTFVectorizer.fit_transform(df['corpus'])


numClusters = 2
km = KMeans(n_clusters=numClusters, init='k-means++')
km.fit(newsgroupsTFVects)
df['kmeans_predictions'] = km.labels_

terms = ngTFVectorizer.get_feature_names_out()
print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(numClusters):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print('\n')

'''
Top terms per cluster:
Cluster 0:
 病例
 新增
 确诊
 输入
 境外
 出院
 治愈
 报告
 死亡
 现有


Cluster 1:
 上海
 疫情
 一个
 中国
 没有
 工作
 发展
 生活
 美国
 文化
'''

'''
X = newsgroupsTFVects.toarray()
plotSilhouette(2, X)
# the average silhouette score shows that 2 is the best cluster number
# There are two major clusters in the corpora, in which group 0 is about covid
# and group 1 is about others
'''

