import os
from django.utils.timezone import now
from .objects import ImageCluster, get_cluster_titles
from mmkg.text.text_analyzer import generate_wordcloud

def get_cluster_wordcloud(source, keyword):

    # use saved wordcloud if it is valid
    cluster = ImageCluster.objects.get(source=source, keyword=keyword)
    #print "Cluster: ", cluster
    if cluster.wordcloud is not None:
        print("wordcloud created: ", cluster.wordcloud_created)
        print("keyword searched: ", cluster.last_search)
        diff = cluster.wordcloud_created - cluster.last_search
        mindiff = (diff.days*24*60) + (diff.seconds/60)
        print("timediff.minutes: ", mindiff)
        if 0 < mindiff:
            return cluster.wordcloud

    titles = get_cluster_titles(source, keyword)
    text = " ".join(titles)

    # Generate a word cloud image
    wordcloud = generate_wordcloud(source, keyword, text)
    cluster.wordcloud = wordcloud
    cluster.wordcloud_created = now()
    cluster.save()
    return wordcloud
