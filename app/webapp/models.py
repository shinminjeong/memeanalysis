from __future__ import unicode_literals

import os, json
import numpy as np
import imagehash, PIL
from itertools import groupby
from dateutil import parser
from datetime import datetime

from operator import itemgetter
from collections import Counter
from langdetect import detect

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

from mmkg.crawler.twitter_crawler import fetch_tweets_from_file
from mmkg.model.cnn import extract_features
from mmkg.model.cluster import cluster_AP, cluster_Kmeans, multiD_scale
from mmkg.text.text_analyzer import get_most_common_tokens, get_w2v_vector, get_t2v_vector, count_label_freq, remove_emoji
from mmkg.text.text_analyzer import get_tokens_tfidf_aggregate, get_tokens_tfidf_average, get_language, get_related_entity

from .objects import *
from .util import extract_article

NUM_WORDS = 10 # number of words to show
IGNORE_FILES = ["README.txt", ".DS_Store"]

def reverse_search(url, download=True):
    infos = download_similar_images(url, download)
    titles = []
    images = []
    for info in infos:
        try:
            if detect(info["title"]) == "en":
                titles.append(info["title"])
        except:
            print("No features in text:", info["title"])
        images.append((info["title"], info["url"], info["user_link"]))
    tokens = get_most_common_tokens("", titles, 5)
    return images, tokens


def is_news_media(url):
    ignore_keywords = [
        "twitter.com",
        "youtube.com",
        "youtu.be",
        "google.com"
    ]
    for keyword in ignore_keywords:
        if keyword in url:
            return False
    return True


def get_included_urls(account):
    flist = get_interaction_tweets(account)
    newslist = []
    for f in flist:
        for l in f["urls"]:
            url = l["expanded_url"]
            if is_news_media(url):
                newslist.append(url)
    return list(set(newslist))


def get_twitter_user_info(screen_name):
    info = get_twitter_user(screen_name)
    name = info["name"]
    loc = info["location"]
    desc = info["description"]
    url = info["url"]
    image = info["profile_image_url"]
    followers = info["followers_count"]
    followings = info["friends_count"]
    tws = info["statuses_count"]
    return name, screen_name, loc, desc, url, image, followers, followings, tws


def get_twitter_friends(account):
    flist = get_interaction_tweets(account)
    sorted_flist = sorted(flist, key=itemgetter("user_screen_name"))
    result = []
    for key, value in groupby(sorted_flist, key=itemgetter("user_screen_name")):
        vlist = list(value)
        group = vlist[0]
        if account == key or len(vlist) < 3:
            continue
        twlist = [(0, v["text"], parser.parse(v["created"])) for v in vlist]
        return_tweets = get_interaction_tweets(group["user_screen_name"], account)
        twlist.extend([(1, v["text"], parser.parse(v["created"])) for v in return_tweets])
        sorted_tw = sorted(twlist, key=itemgetter(2), reverse=True)

        result.append((group["user_screen_name"], group["user_name"],
                        group["user_profile_image"], len(twlist), sorted_tw))
    return list(zip(*sorted(result, key=itemgetter(3), reverse=True)))


def extract_feature(keyword, images):
    paths = [img.path for img in images]
    X = extract_features(paths)
    for img, x in zip(images, X):
        img.fvector = json.dumps(x[0].tolist())
        img.labels = json.dumps(x[1])
        img.save()


def count_w2v_label(keyword, images, num_group):
    titles = []
    for imgcluster in images:
        tl = [p[1] for p in imgcluster]
        titles.append(tl)
    words, X = get_w2v_vector(keyword, titles)

    labels, num, pos = cluster_Kmeans(X, num_group)
    freq_plot = count_label_freq(words, num, labels, titles, num_group)
    return freq_plot


def find_similar_images(source, keyword, p):
    imcluster = get_keyword_cluster(source, keyword)
    images = get_original_images(imcluster)
    comp = images.filter(path=p)[0]
    sim = []
    for img in images:
        dist = calculate_image_dist(comp, img)
        if 0 < dist:
            sim.append((img.path, img.title, dist))
    sorted_list = sorted(sim, key=itemgetter(2))
    avg = np.mean([s[2] for s in sim])
    std = np.std([s[2] for s in sim])
    thres = avg + std*(-2) # < -2sigma
    threslist = [s for s in sorted_list if s[2] < thres]
    return [(s[0], s[1]) for s in threslist], sorted_list, thres


def make_image_groups(images, labels, num):
    clusters = [[] for y in range(num)]
    for i in range(len(labels)):
        clusters[labels[i][0]].append((images[i], labels[i][1]))
    # sort images by the distance to the cluster center
    sorted_clusters = []
    for c in clusters:
        sorted_clusters.append([p[0] for p in sorted(c, key=itemgetter(1))])
    return sorted_clusters


def make_clusters(keyword, images, labels, num):
    # sort images by the distance to the cluster center
    sorted_clusters = make_image_groups(images, labels, num)

    titleclusters = []
    for c in sorted_clusters:
        titleclusters.append([i.title for i in c])

    # token tfidf value
    #tfidf = get_tokens_tfidf_aggregate(keyword, titleclusters, num, NUM_WORDS)
    tfidf = get_tokens_tfidf_average(keyword, titleclusters, num, NUM_WORDS)

    # token wordcount
    wc = []
    for i in range(num):
        tokens = get_most_common_tokens(keyword, titleclusters[i], NUM_WORDS)
        wc.append(tokens)
    scores = get_dist_from_centroid(num, labels)
    return wc, tfidf, scores, sorted_clusters


def save_twitter_infos(imcluster, keyword, infos, source="twitter"):
    exists = 0
    original_count = 0
    for info in infos:
        if check_image_exists(imcluster, info["id"]):
            exists += 1
            continue
        image = Image(imageid=info["id"], source=source, keyword=keyword, title=info["title"].encode("utf8"),\
                    path=info["path"], url=info["url"], user=info["user"].encode("utf8"),\
                    user_link=info["user_link"], rawdata=json.dumps(info["rawdata"]))
        image.hashkey = str(imagehash.dhash(PIL.Image.open(image.path)))
        image.created = parser.parse(info["created"])
        try:
            image.save()
        except Exception as e:
            pass

        # if if has the parent tweet
        orig = info["original"]
        if orig != None:
            original_count += 1
            if check_image_exists(imcluster, orig["id"]):
                orig_image = imcluster.images.get(imageid=orig["id"])
            else:
                orig_image = Image(imageid=orig["id"], source=source, keyword=keyword,\
                            title=orig["title"].encode("utf8"), path=info["path"], url=orig["url"],\
                            user=orig["user"].encode("utf8"), user_link=orig["user_link"],\
                            rawdata=json.dumps(orig["rawdata"]))
                orig_image.hashkey = str(imagehash.dhash(PIL.Image.open(orig_image.path)))
                orig_image.created = parser.parse(orig["created"])
                try:
                    orig_image.save()
                    imcluster.images.add(orig_image)
                except Exception as e:
                    pass

            try:
                image.original = False
                image.parent = orig_image
                orig_image.retweets.add(image)
                orig_image.save()
            except Exception as e:
                pass

        try:
            image.save()
            imcluster.images.add(image)
        except Exception as e:
            pass
    imcluster.save()
    print("num of image downloaded: %d, original_images: %d" % (imcluster.images.count(), original_count))
    return exists


def fetch_from_file(flist, keyword):
    filename = flist[0]
    return fetch_twitter_data(filename, keyword)


def fetch_twitter_data(filename, keyword):
    # save images to the imagecluster
    #print "image cluster:", keyword
    imcluster = get_keyword_cluster("twitter", keyword, create=True, update_time=True)

    num, infos = fetch_tweets_from_file(filename)
    exists = save_twitter_infos(imcluster, keyword, infos)

    # use all images in the imagecluster
    images = get_original_images(imcluster)
    total = images.count()
    extract_feature(keyword, images)

    return keyword, total-exists, total


def get_dist_from_centroid(num, labels):
    dist = [[] for y in range(num)]
    for c, d in labels:
        dist[c].append(d)
    return [np.average(dist[y]) for y in range(num)]


def get_feature_matrix(images):
    mat = []
    for img in images:
        v = np.array(json.loads(img.fvector))
        mat = np.concatenate((mat, np.squeeze(v)))
    mat = mat.reshape(len(images), 2048)
    return mat


def cluster_eval(source, model, keyword):
    imcluster = get_keyword_cluster(source, keyword)
    assert imcluster != None

    images = get_original_images(imcluster)
    X = get_feature_matrix(images)
    total = images.count()

    res_num = []
    scores = []
    if model == "ap":
        pref = range(-1200, -100, 100)
        for p in pref:
            labels, num, _ = cluster_AP(X, p)
            scores.append(np.average(get_dist_from_centroid(num, labels)))
            res_num.append(num)
    if model == "kmeans":
        pref = range(1, 16, 1)
        for p in pref:
            labels, num, _ = cluster_Kmeans(X, p)
            scores.append(np.average(get_dist_from_centroid(num, labels)))
            res_num.append(num)

    return total, pref, res_num, scores


def get_model_pref(num):
    # TODO: choose reasonable number of clusters
    model = "kmeans"
    if num == 1:
        return model, 1
    pref = int(np.log2(num))
    return model, pref


def sort_images(source, keyword, num):
    imcluster = get_keyword_cluster(source, keyword)
    assert imcluster != None
    image_path = [image.path for image in get_original_images(imcluster)]
    img_counter = Counter(image_path)
    imglist = img_counter.most_common(num)

    imcluster.repimages_model = "sorted"
    imcluster.repimages_pref = num
    imcluster.repimages.clear()
    for p, c in imglist:
        im = list(Image.objects.filter(path=p)[:1])
        imcluster.repimages.add(im[0])
    imcluster.save()
    return imglist


def cluster_titles(source, keyword, model, pref):
    imcluster = get_keyword_cluster(source, keyword)
    assert imcluster != None
    o_images = get_original_images(imcluster)

    # find image duplication
    check_duplicated_images(o_images)
    images = [i for i in o_images if not i.duplicate]

    titles = [i.title for i in images]
    words, X = get_t2v_vector(keyword, titles)

    if model == "" or model == "sorted":
        model, pref = get_model_pref(images.count())
    if model == "ap":
        labels, num, centers = cluster_AP(X, pref)
    if model == "kmeans":
        labels, num, centers = cluster_Kmeans(X, pref)

    wc, tfidf, scores, sorted_clusters = make_clusters(keyword, images, labels, num)
    save_repimages(model, pref, imcluster, sorted_clusters)
    return wc, tfidf, scores, sorted_clusters


def save_repimages(model, pref, imcluster, sorted_clusters):
    imcluster.repimages_model = model
    imcluster.repimages_pref = pref
    imcluster.repimages.clear()
    for sc in sorted_clusters:
        im = list(Image.objects.filter(path=sc[0].path)[:1])
        imcluster.repimages.add(im[0])
    imcluster.save()


def cluster_images(images, model="", pref=0):
    X = get_feature_matrix(images)
    if model == "" or model == "sorted":
        model, pref = get_model_pref(len(images))
    if model == "ap":
        labels, num, centers = cluster_AP(X, pref)
    if model == "kmeans":
        labels, num, centers = cluster_Kmeans(X, pref)
    return labels, num, centers


def cluster_images_from_keyword(source, keyword, model, pref):
    imcluster = get_keyword_cluster(source, keyword)
    assert imcluster != None
    o_images = get_original_images(imcluster)

    # find image duplication
    check_duplicated_images(o_images)
    images = [i for i in o_images if not i.duplicate]

    if model == "" or model == "sorted":
        model, pref = get_model_pref(len(images))

    labels, num, centers = cluster_images(images, model, pref)
    wc, tfidf, scores, sorted_clusters = make_clusters(keyword, images, labels, num)
    save_repimages(model, pref, imcluster, sorted_clusters)
    return wc, tfidf, scores, sorted_clusters
