from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template import Context
from django.template.context_processors import csrf
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.conf import settings

import os, json
import timeit
from datetime import datetime, timedelta
from operator import itemgetter
from collections import OrderedDict

from .models import *
from .charts import *
from .objects import *
from .util import encode_ascii, lookup_dbpedia, extract_article
from .wordcloud import get_cluster_wordcloud
from .cached_page import CachedPage

saved_page = CachedPage()
cur_path = os.path.dirname(os.path.abspath(__file__))

def download(request):
    type = request.GET.get("type")
    print("download", request.GET)

    keyword = ""
    status = ""
    indices = []
    # try:
    #     _, indices = get_es_indices()
    # except Exception as e:
    #    indices = [["Failed to get indices from ES", "Error"]]
    # print(indices)

    today = datetime.now()
    timediff = timedelta(days=7)
    startrange = today-timediff
    daterange = startrange.strftime("%Y-%m-%d") + " ~ " + today.strftime("%Y-%m-%d")

    checkbox = OrderedDict()
    checkbox["title"] = True
    checkbox["description"] = True
    checkbox["hashtags"] = False
    checkbox["entities"] = False

    if "fetch" in request.POST:
        filetype = request.POST.get("filetype")
        keyword = request.POST.get("keyword")
        flist = request.FILES.getlist("infile")
        keyword, new, total = fetch_from_file(filetype, flist, keyword)
        source = "twitter"

    if "remove" in request.GET:
        remove = request.GET.get("remove")
        print("remove " + remove)
        source, keyword = remove.split('|')
        if source == "bing" or source == "twitter" or source == "facebook":
            remove_keyword_cluster(source, keyword)
        else:
            remove_keyword_cluster(source, keyword, delete_image=False)
        status = "%s has been removed" % remove

    if "websearch" in request.GET:
        source = request.GET.get("source")
        keyword = request.GET.get("keyword")
        count = int(request.GET.get("count"))
        print("search", source, keyword, count)

        new, total = search_images(source, keyword, count)
        status = "[%s from %s] %d new images are downloaded, total: %d"\
                % (keyword, source, new, total)

    if "search" in request.GET:
        index = request.GET.get("index")
        keyword = request.GET.get("keyword")
        daterange = request.GET.get("daterange")
        for t in checkbox.keys():
            checkbox[t] = True if request.GET.get(t) == "on" else False
        # print("index: %s keyword: %s daterange: %s" %(index, keyword, daterange))
        # print("checkbox: %s" %(checkbox))
        checked = [k for k in checkbox if checkbox[k]]
        count, _ = search_es_keyword(keyword, index, checked, daterange)
        status = "[%s] found %d documents from keyword: %s" % (index, count, keyword)
        if count > 0:
            es_info = save_es_images(keyword, index, checked, daterange)

    return render(request, "download.html", {"type": type, "indices": indices,
                    "status": status, "daterange": daterange, "option": checkbox.items()})

def account(request):
    status = ""
    if "search" in request.GET:
        sname = request.GET.get("screenname")
        print("Accont search", sname)

        user_info = get_twitter_user_info(sname)

        # get included media
        links = []
        for l in get_included_urls(sname):
            date, keywords, title, summary, img = extract_article(l)
            links.append((l, date, keywords, title, summary, img))

        # get interaction
        flist = get_twitter_friends(sname)
        interaction = None
        if len(flist) > 0:
            fname, fsname, url, count, tw = flist
            num_words = 10
            twclusters = []
            wc = []
            for tws in tw:
                tlist = [t[1] for t in tws]
                twclusters.append(tlist)
                wctokens = get_most_common_tokens(sname, tlist, num_words)
                wc.append(wctokens)
            tfidf = get_tokens_tfidf_average(sname, twclusters, len(twclusters), num_words)
            interaction = zip(wc, tfidf, fname, fsname, url, count, tw)
        return render(request, "account.html", {"userinfo": user_info,
                        "links": links, "interaction": interaction})

    return render(request, "account.html")


def media(request):
    if "id" in request.GET:
        img_path = request.GET.get("id")
        duplicates = find_duplicate(get_image(img_path), True)

        infos = []
        for img in duplicates:
            source, keyword = get_media_source(img.path)
            if source == "twitter":
                infos.append((img.path, source, get_media_info_twitter(img.path)))
            else:
                infos.append((img.path, source, get_media_info(img.path)))

        img_url = "http://"+request.get_host()+settings.STATIC_URL+img_path
        print("reverse image search:", img_url)
        rev_images, tags = reverse_search(img_url)
        sim, dlist, thres = find_similar_images(source, keyword, img_path)
        values = get_distance_histogram(dlist, thres)

        return render(request, "media.html", {"img": img_path,
                            "infos": infos, "sim": sim,
                            "values": json.dumps(values),
                            "rev": rev_images, "tags": tags} )
    return render(request, "media.html")


def search(request):
    keyword = ""
    status = ""

    today = datetime.now()
    timediff = timedelta(days=7)
    startrange = today-timediff
    daterange = startrange.strftime("%Y-%m-%d") + " ~ " + today.strftime("%Y-%m-%d")

    if "remove" in request.GET:
        remove = request.GET.get("remove")
        print("remove " + remove)
        source, keyword = remove.split('|')
        if source == "bing" or source == "twitter" or source == "facebook":
            remove_keyword_cluster(source, keyword)
        else:
            remove_keyword_cluster(source, keyword, delete_image=False)
        status = "%s has been removed" % remove


    keylist = get_keyword_list("all")
    wc = [get_cluster_wordcloud(k[1],k[0]) for k in keylist]
    repimages = [get_cluster_repimages(k[1],k[0]) for k in keylist]
    cloud = [x + (y,) + z for x, y, z in zip(keylist, wc, repimages)]
    return render(request, "search.html", {"keylist": keylist, "cloud": cloud,
                        "status": status, "daterange": daterange})


def cluster_more(request):
    index = int(request.GET.get("index"))-1
    source = request.GET.get("source")
    keyword = request.GET.get("keyword")
    print("more: ", source, keyword, index)
    # print(saved_page.images[index-1])
    if keyword != saved_page.keyword or source != saved_page.source\
        or saved_page.w2v_labels == None:
        saved_page.w2v_labels = count_w2v_label(keyword, saved_page.images, 10) # num of label groups
    images = saved_page.images[index]
    words_tfidf = saved_page.tfidf[index]
    words_wc = saved_page.wc[index]

    values = get_w2v_freq_chart(saved_page.w2v_labels, index)
    timechart = get_timeline_chart(saved_page.images, index)
    return render("page/more_info.html", {"wc": words_wc,
                        "tf": words_tfidf, "idx": index,
                        "label": values, "times": timechart})

def cluster(request):
    global saved_page
    resultform = None
    images = None

    if "cluster" in request.GET:
        source = request.GET.get("source")
        model = request.GET.get("model")
        keyword = request.GET.get("keyword")
        pref = int(request.GET.get("pref"))
        ctype = request.GET.get("type")
        print(source, model, keyword, pref, ctype)

        if ctype == "text":
            wc, tfidf, scores, rawimages = cluster_titles(source, keyword, model, pref)
        else: # type=="image"
            wc, tfidf, scores, rawimages = cluster_images_from_keyword(source, keyword, model, pref)

        words_tfidf = encode_ascii(tfidf)
        words_wc = encode_ascii(wc)
        ncluster = len(rawimages)

        images = []
        imglabels = []
        for raw in rawimages:
            info = [get_image_info(i) for i in raw if i.original]
            if ctype == "image": # calculate the average of image labels
                imglabels.append(count_image_label(info))
            images.append(info)

        if ctype == "text": # get intersection of wc and tfidf
            imglabels = get_intersection(words_tfidf, words_wc)

        saved_page.save(ctype, source, keyword, pref, model, ncluster,
                        words_wc, words_tfidf, scores, images, imglabels)

    return render(request, "cluster.html", saved_page.get())
