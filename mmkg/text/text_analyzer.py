import os, re, requests
import nltk, string
import codecs, itertools
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

URL_DBPEDIA_PREFIX = 'http://model.dbpedia-spotlight.org/en/annotate'
Headers = {'Accept': 'application/json'}

stopwords_dir = os.environ["MEME_DATA_PATH"] + "/stopwords"
w2vglove_dir = os.environ["MEME_DATA_PATH"] + "/word2vec"
wordcloud_dir = os.environ["MEME_DATA_PATH"] + "/wordcloud"

class Tokenizer:

    __state = {}
    stopwords = None
    stopwords_contain = None

    def get_stopwords_from_file(self, fname):
        with codecs.open(fname, "r", "utf8") as f:
            content = f.readlines()
            return [x.rstrip('\n') for x in content]

    def tokenizer(self, t):
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(t.lower())
        filtered = [w for w in tokens\
                    if w not in stopwords.words("english")\
                    and w not in string.punctuation\
                    and w not in self.stopwords]
        filtered2 = [w for w, s in itertools.product(filtered, self.stopwords_contain)\
                     if s not in w]
        tagged = [word[0] for word in nltk.pos_tag(filtered2)]
        lemma = [wordnet_lemmatizer.lemmatize(tag) for tag in tagged]
        return lemma

    def __init__(self):
        # make it singleton class to read stopwords from file only once
        self.__dict__ = self.__state
        if self.stopwords is None:
            f_stopwords = os.path.join(stopwords_dir, "stopwords")
            self.stopwords = self.get_stopwords_from_file(f_stopwords)
        if self.stopwords_contain is None:
            f_contain = os.path.join(stopwords_dir, "stopwords_contain")
            self.stopwords_contain = self.get_stopwords_from_file(f_contain)


def generate_wordcloud(source, keyword, text):
    T = Tokenizer()
    stopwords = set(STOPWORDS)
    stopwords.add(keyword.split('@')[-1])
    for w in T.stopwords:
        stopwords.add(w)

    wc_path = ""
    try:
        # Generate a word cloud image
        wordcloud = WordCloud(
                background_color="white",
                max_words=50,
                width=400,
                height=400,
                stopwords=stopwords
                ).generate(text)

        wc_path = os.path.join(wordcloud_dir, "%s_%s.jpg"%(source, keyword))
        wordcloud.to_file(wc_path)
    except Exception as ex:
        print("generate_wordcloud", str(ex))

    return wc_path


def remove_emoji(string):
    if not string:
        return None
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def get_related_entity(input_text, confidence_level):
	payload = {'text': remove_emoji(input_text), 'confidence': confidence_level}
	response = requests.get(URL_DBPEDIA_PREFIX, params=payload, headers=Headers)
	# print('curl: ' + response.url)
	# print('return statue: ' + str(response.status_code))
	if response.status_code != 200:
		print('return statue: ' + str(response.status_code))
		print('ERROR: problem with the request.', input_text)
		return None
	return response.json()


def get_most_common_tokens(keyword, titles, num_tokens):
    wordset = Counter([])
    wordnet_lemmatizer = WordNetLemmatizer()
    T = Tokenizer()
    for t in titles:
        if t:
            tokens = T.tokenizer(t)
            words = [w for w in tokens if w not in keyword.lower()]
            wordset += Counter(words)
    return wordset.most_common(num_tokens)


def get_tokens_tfidf_aggregate(keyword, titlecluster, num_cluster, num_tokens):
    token_dict = {}
    for i in range(num_cluster):
        t = '\n'.join(titlecluster[i])
        token_dict[i] = t.lower()
    T = Tokenizer()
    tfidf = TfidfVectorizer(stop_words='english', tokenizer=T.tokenizer)
    tfs = tfidf.fit_transform(token_dict.values())

    result = []
    for i in range(num_cluster):
        t = token_dict[i]
        response = tfidf.transform([t])
        print(i)
        print(response)
        features = []
        feature_names = tfidf.get_feature_names()
        for col in response.nonzero()[1]:
            if feature_names[col] in keyword.lower(): continue
            features.append((feature_names[col], "%.5f" % response[0, col]))
        result.append(sorted(features, key=itemgetter(1), reverse=True)[0:num_tokens])
    return result


def get_tokens_tfidf_average(keyword, titlecluster, num_cluster, num_tokens):
    token_dict = []
    for i in range(num_cluster):
        for t in titlecluster[i]:
            token_dict.append(t.lower())
    T = Tokenizer()
    tfidf = TfidfVectorizer(stop_words='english', tokenizer=T.tokenizer)
    tfs = tfidf.fit_transform(token_dict)

    result = []
    for i in range(num_cluster):
        ressum = np.zeros(shape=(1, tfs.shape[1]))
        for t in titlecluster[i]:
            res = tfidf.transform([t.lower()])
            ressum += res
        response = np.divide(ressum, float(len(titlecluster[i])))

        features = []
        feature_names = tfidf.get_feature_names()
        for col in response.nonzero()[1]:
            if feature_names[col] in keyword.lower(): continue
            features.append((feature_names[col], "%.5f" % response[0, col]))
        result.append(sorted(features, key=itemgetter(1), reverse=True)[0:num_tokens])
    return result


def read_glove(fname):
    W = {}
    with open(fname) as f:
        content = f.readlines()
        for c in content:
            l = c.split()
            W[l[0]] = l[1:]
    f.close()
    return W


def get_t2v_vector(keyword, titles):
    # title to vector
    f_glove = os.path.join(w2vglove_dir, "glove.6B.50d.txt")
    W = read_glove(f_glove)
    words_avail = []
    vectors = []
    T = Tokenizer()
    for t in titles:
        v = np.zeros(50) # number of dim
        words = []
        l = T.tokenizer(t)
        for w in l:
            if w in W and w not in keyword:
                words.append(w)
                np.add(v, np.array(W[w], dtype='float64'), out=v)
        words_avail.append(words)
        vectors.append(v)
    return words_avail, vectors


def get_w2v_vector(keyword, titles):
    # word to vector
    f_glove = os.path.join(w2vglove_dir, "glove.6B.50d.txt")
    W = read_glove(f_glove)
    words = []
    T = Tokenizer()
    for cluster in titles:
        for t in cluster:
            l = T.tokenizer(t)
            words.extend(l)

    words_avail = []
    vectors = []
    for w in words:
        if w in W and w not in keyword:
            words_avail.append(w)
            vectors.append(W[w])
    return words_avail, vectors


def count_label_freq(words_avail, num, labels, titles, num_labels):
    wordclusters = [[] for y in range(num)]
    for i in range(len(labels)):
        wordclusters[labels[i][0]].append((words_avail[i], labels[i][1]))

    sorted_wordclusters = []
    for c in wordclusters:
        clist = Counter([p for p,q in sorted(c, key=itemgetter(1))])
        wc = [e[0] for e in clist.most_common(num_labels)]
        sorted_wordclusters.append(wc)

    T = Tokenizer()
    cluster_plots = []
    for cluster_titles in titles:
        c_words = T.tokenizer(" ".join(cluster_titles))
        scores = []
        for repwords in sorted_wordclusters:
            count = 0
            for cw in c_words:
                if cw in repwords:
                    count += 1
            scores.append((repwords, count))
        cluster_plots.append(scores)
    return cluster_plots


def get_language(text):
    ENG_STOPWORDS = set(nltk.corpus.stopwords.words('english'))
    NON_ENG_STOPWORDS = set(nltk.corpus.stopwords.words()) - ENG_STOPWORDS
    STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}
    words = set(nltk.wordpunct_tokenize(text.lower()))
    return max(((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()), key=itemgetter(1))[0]
