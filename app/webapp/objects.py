import os, json
import numpy as np
import itertools
from django.db import models, connection
from django.utils import timezone
from operator import itemgetter
from mmkg.text.text_analyzer import get_most_common_tokens

SIM_DIST = 10

class Image(models.Model):
    id = models.AutoField(primary_key=True)
    imageid = models.CharField(max_length=100, default="")
    source = models.CharField(max_length=20, default="")
    keyword = models.CharField(max_length=20, default="")
    title = models.CharField(max_length=200, default="")
    description = models.TextField(default="")
    user = models.CharField(max_length=100, default="")
    user_link = models.CharField(max_length=500, default="")
    created = models.DateTimeField(null=True)

    path = models.CharField(max_length=200, default="")
    url = models.TextField(default="")
    fvector = models.TextField(default="")
    labels = models.TextField(default="")
    rawdata = models.TextField(default="")

    hashkey = models.TextField(null=True, default=None)
    duplicate = models.BooleanField(default=False)

    # for twitter RT
    original = models.BooleanField(default=True)
    parent = models.ForeignKey('Image', null=True, on_delete=models.DO_NOTHING)
    retweets = models.ManyToManyField('Image', related_name="%(class)s_retweets")

    def __str__(self):
        return "%s %s" %(self.imageid, self.title)


class ImageCluster(models.Model):
    type = models.CharField(max_length=10, default="web")
    source = models.CharField(max_length=20, default="")
    keyword = models.CharField(max_length=20, default="")
    last_search = models.DateTimeField()
    images = models.ManyToManyField('Image', related_name="images")

    wordcloud = models.TextField(null=True)
    wordcloud_created = models.DateTimeField(null=True)

    repimages = models.ManyToManyField('Image', related_name="repimages")
    repimages_model = models.CharField(max_length=20, default="")
    repimages_pref = models.IntegerField(default=0)

    def __str__(self):
        return "%s %s" %(self.source, self.keyword.encode('ascii', 'ignore'))


def calculate_hamming_dist(h1, h2):
    if h1 == None or h2 == None:
        return 1000
    fh1 = format(int(h1, 16), '#066b')
    fh2 = format(int(h2, 16), '#066b')
    return sum(c1 != c2 for c1, c2 in zip(fh1, fh2))

def check_duplicated_images(images):
    for i, j in itertools.combinations(images, 2):
        if calculate_hamming_dist(i.hashkey, j.hashkey) <= SIM_DIST:
            j.duplicate = True

def find_duplicate(img, original_filter=False):
    if original_filter:
        images = Image.objects.filter(source=img.source, keyword=img.keyword, original=True)
    else:
        images = Image.objects.filter(source=img.source, keyword=img.keyword)
    return [i for i in images if calculate_hamming_dist(img.hashkey, i.hashkey) <= SIM_DIST]

def count_image_label(info):
    labels = {}
    for i in info:
        for k, v in i[3]:
            if k in labels:
                labels[k] += v
            else:
                labels[k] = v
    sorted_labels = sorted(labels.items(), key=itemgetter(1), reverse=True)
    return [(k, v/len(info)) for k, v in sorted_labels][:10]

def get_intersection(s_tfidf, s_wc):
    intersection = []
    for tfidf, wc in zip(s_tfidf, s_wc):
        tfidf_keys = [k for k,v in tfidf]
        its = [(k,v) for k,v in wc if k in tfidf_keys]
        intersection.append(its)
    return intersection

def get_image_info(img):
    labels = []
    llist = json.loads(img.labels)
    for name, score in llist:
        if score > 0.05:
            labels.append((name.split(',')[0], 100.0*score))
    return (img.path, img.title, len(find_duplicate(img)), labels, img.created)

def calculate_image_dist(img1, img2):
    v1 = np.array(json.loads(img1.fvector))
    v2 = np.array(json.loads(img2.fvector))
    dist = np.mean((v1-v2)**2)
    return dist


def get_cluster_repimages(source, keyword):
    cluster = ImageCluster.objects.get(source=source, keyword=keyword)
    model = cluster.repimages_model
    pref = cluster.repimages_pref
    return model, pref, [(i.path, i.title) for i in cluster.repimages.all()]


def get_cluster_titles(source, keyword):
    cluster = ImageCluster.objects.get(source=source, keyword=keyword)
    titles = [i.title for i in cluster.images.all()]
    return titles


def get_keyword_list(source):
    klist = []
    if ImageCluster._meta.db_table in connection.introspection.table_names():
        if source == "all":
            clusters = ImageCluster.objects.all()
        elif source == "web":
            clusters = ImageCluster.objects.filter(type="web")
        elif source == "es":
            clusters = ImageCluster.objects.filter(type="es")
        else:
            clusters = ImageCluster.objects.filter(source=source)
        for c in clusters:
            klist.append((c.keyword, c.source, c.last_search, c.type, c.images.filter(original=True).count()))
    return sorted(klist, key=itemgetter(2), reverse=True)


def check_keyword_exists(source, keyword):
    if ImageCluster._meta.db_table in connection.introspection.table_names():
        return ImageCluster.objects.filter(source=source, keyword=keyword).count() > 0
    return False


def check_image_exists(imcluster, id):
    if imcluster.images.filter(imageid=id).count() > 0:
        return True
    return False


def get_keyword_cluster(source, keyword, create=False, update_time=False, type="web"):
    if check_keyword_exists(source, keyword):
        imcluster = ImageCluster.objects.get(source=source, keyword=keyword)
        imcluster.last_search=timezone.now()
        return imcluster
    elif create:
        imcluster = ImageCluster(source=source, keyword=keyword, last_search=timezone.now(), type=type)
        imcluster.save()
        return imcluster
    else:
        print("Table %s/%s NOT exists - Download first!" % (source, keyword))
        return None


def remove_keyword_cluster(source, keyword, delete_image=True):
    if not check_keyword_exists(source, keyword):
        return False
    imcluster = ImageCluster.objects.get(source=source, keyword=keyword)
    images = imcluster.images.all()
    for img in images:
        if delete_image:
            if img.path != None and os.path.isfile(img.path):
                os.remove(img.path)
        img.delete()
    imcluster.delete()
    return True


def get_original_images(imcluster):
    return imcluster.images.filter(original=True).all()


def get_image(path):
    img = Image.objects.filter(path=path)[0]
    return img

def get_media_source(path):
    img = Image.objects.filter(path=path)[0]
    return img.source, img.keyword


def get_media_info(path):
    infos = []
    imginfo = Image.objects.filter(path=path)
    for img in imginfo:
        infos.append((img.title, img.user, img.user_link, img.description))
    return infos

def get_media_info_twitter(path):
    infos = {}
    imginfo = Image.objects.filter(path=path)
    for img in imginfo:
        if not img.original:
            o_img = img.parent
            if (o_img.title, o_img.user) in infos.keys():
                infos[(o_img.title, o_img.user)] += ((img.title, img.user),)
            else:
                infos[(o_img.title, o_img.user)] = ((img.title, img.user),)
        else:
            if not (img.title, img.user) in infos.keys():
                infos[(img.title, img.user)] = ()
    return [(k[0], k[1], infos[k]) for k in infos.keys()]
