class CachedPage:
    w2v_labels = None
    source = None
    keyword = None
    def save(self, ctype, source, keyword, pref, model, ncluster,
                wc, tfidf, scores, images, imglabels, timechart):
        self.source = source
        self.keyword = keyword
        self.pref = pref
        self.model = model
        self.ncluster = ncluster
        self.wc = wc
        self.ctype = ctype
        self.tfidf = tfidf
        self.scores = scores
        self.images = images
        self.imglabels = imglabels
        self.w2v_labels = None
        self.timechart = timechart

    def get(self):
        c_idx = range(1, self.ncluster+1)
        numimages = [len(a) for a in self.images]
        return {"type": self.ctype,
                "s": self.source,
                "key": self.keyword,
                "pref": self.pref,
                "model": self.model,
                "ncluster": self.ncluster,
                "times": self.timechart,
                "words_images":zip(c_idx, numimages,
                    self.scores, self.images, self.imglabels)}
