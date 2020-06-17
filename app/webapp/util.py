import json, requests
from .objects import get_keyword_cluster
from mmkg.text.text_analyzer import get_most_common_tokens
from newspaper import Article
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from difflib import SequenceMatcher

DBPEDIA_LOOKUP_ADDR = "http://lookup.dbpedia.org/api/search/KeywordSearch"

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def _lookup_dbpedia(concept):
    payload = "QueryString=%s"%concept[0]
    response = requests.get(DBPEDIA_LOOKUP_ADDR, params=payload, headers={"Accept": "application/json"})
    # print('curl: ' + response.url)
    # print('return statue: ' + str(response.status_code))
    if response.status_code != 200:
        print("return statue: " + str(response.status_code))
        print("ERROR: problem with the request.")
        exit()

    result = json.loads(response.content.decode("utf-8"))
    if len(result["results"]) == 0:
        return (concept[1], concept[0], concept[0], "", [])

    first_match = result["results"][0]
    best_match = first_match
    max_similarity = 0
    for res in result["results"]:
        sim = similar(res["label"], concept[0])
        if sim > max_similarity:
            max_similarity = sim
            best_match = res
    # first_match = result["results"][0]
    classes = []
    if "classes" in best_match:
        classes = [(e["label"], e["uri"]) for e in best_match["classes"]]
    return (concept[1], concept[0].replace('_', ' '), best_match["label"], best_match["uri"], classes)

def lookup_dbpedia(conceptlist):
    n_cpu = cpu_count()*2
    p = ThreadPool(n_cpu)
    res = p.map(_lookup_dbpedia, conceptlist)
    return res


def extract_article(url):
    date = None
    keywords = []
    title = ""
    summary = ""
    image = ""
    try:
        article = Article(url)
        article.download()
        article.parse()
        date = article.publish_date
        article.nlp()
        keywords = article.keywords
        title = article.title
        summary = article.summary
        image = article.top_image
    except:
        print("error: extract article from url %s" %str(url))
    return date, keywords, title, summary, image


def encode_ascii(wc):
    l = []
    for cloud in wc:
        l.append([(w.encode("ascii", "xmlcharrefreplace"), c) for w,c in cloud])
    return l


def generate_json(source, keyword, index):
    ic = get_keyword_cluster(source, keyword)
    s = ic.images.all()[index]
    p = {}
    p["image_id"] = s.id
    p["image_url"] = s.url
    p["image_path"] = s.path
    p["title"] = s.title
    p["user"] = s.user
    p["createdAt"] = "%s"%s.created
    p["postedAt"] = "%s"%s.created
    p["keyword"] = s.keyword
    #tokens = get_most_common_tokens("", [s.title], 10)
    #p["tokens"] = [s for s, c in tokens]

    url = "http://130.56.254.119:9200/test/twitter/%d?pretty" % index
    headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}
    r = requests.post(url, data=json.dumps(p), headers=headers)
    print(r.url)
    print(r.status_code )
    return json.dumps(p)


def save_to_file(source, keyword):
    ic = get_keyword_cluster(source, keyword)
    arr = ["%s|||%s|||" %(s.title, s.path) for s in ic.images.all()]
    with open("output.dat", "w") as f:
        f.write(("\n".join(arr)).encode("utf-8"))
    f.close()
