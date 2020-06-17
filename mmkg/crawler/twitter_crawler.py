import os, sys, json, tweepy
import urllib, shutil
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

download_to = os.environ["MEME_DATA_PATH"] +"/images/"

def retrieve_tweet(status):
    user = status["user"]
    tw = {
        "text": status["text"],
        "created": status["created_at"],
        "user_name": user["name"],
        "user_screen_name": user["screen_name"],
        "user_profile_image": user["profile_image_url"],
        "urls": status["entities"]["urls"]
    }
    return tw


def get_twitter_media_info(status, media):
    user = status["user"]
    tm = {
        "title": status["text"],
        "url": media["media_url"],
        "image_id": media["id"],
        "id": status["id"],
        "user": "%s (@%s)" %(user["name"], user["screen_name"]),
        "user_link": ""
    }
    return tm


def download_image(imageid, url):
    download_path = os.path.join(download_to, str(imageid))
    try:
        resp = urllib.request.urlopen(url)
        with open(download_path, "wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception as e:
        print("image inaccessible: " + str(e) + url)
        return None
    return download_path


def twitter_image_retrieve(status):
    rt_info = None
    if "retweeted_status" in status.keys() and status["retweeted_status"] != None:
        rt_status = status["retweeted_status"]
        entities = rt_status["entities"]
        created = rt_status["created_at"]
        if "media" in entities.keys() and entities["media"] != None:
            media = entities["media"][0]
            tm = get_twitter_media_info(rt_status, media)
            # download_path = download_image(tm["image_id"], tm["url"])
            # if download_path != None:
            rt_info = tm
            rt_info["created"] = created
            # rt_info["path"] = download_path
            rt_info["path"] = ""
            rt_info["rawdata"] = json.dumps(rt_status)
            rt_info["original"] = None

    entities = status["entities"]
    created = status["created_at"]
    if "media" in entities.keys() and entities["media"] != None:
        media = entities["media"][0]
        tm = get_twitter_media_info(status, media)
        download_path = download_image(tm["image_id"], tm["url"])
        if download_path != None:
            tm["created"] = created
            tm["path"] = download_path
            tm["rawdata"] = json.dumps(status)
            tm["original"] = rt_info
            return tm
    return


def get_included_media(account):
    infos = []
    num_tweet = 0
    max_id = 1
    while num_tweet < 1000:
        num_tweet += 100 # The number of tweets to return per page
        query = "from:%s url:news" % account
        statuses = [t._json for t in query_twitter(query, num_tweet, max_id-1)]
        res = p.map(retrieve_tweet, statuses)
        infos.extend(res)
        if len(statuses) > 0:
            max_id = min([status["id"] for status in statuses])
        else:
            break
        print("found %d tweets" % num_tweet)

    p.close()
    p.join()
    return infos


def fetch_tweets_from_file(filename):
    n_cpu = cpu_count()*2
    p = ThreadPool(n_cpu)
    print("Fetch images from Ingested twitter files: use %d threads" % n_cpu)

    fo = open(filename)
    statuses = [json.loads(l) for l in fo.readlines()]

    infos = []
    res = p.map(twitter_image_retrieve, statuses)
    infos.extend([r for r in res if r != None])
    num_image = len(infos)
    print("found %d images from %d tweets" %(num_image, len(statuses)))

    p.close()
    p.join()
    return num_image, infos

if __name__ == '__main__':
    # query, count, filepath = sys.argv[1:4]
    #download_and_save_tweets("@JulianRoepcke", 100000, "/mnt/MMKG/mmkg/data/JulianRoepcke_100K.dump")
    # download_and_save_tweets_account(query, int(count), filepath)
    fetch_tweets_from_file(sys.argv[1])
