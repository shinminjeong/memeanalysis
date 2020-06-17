from .models import cluster_eval
from collections import Counter
import json
import numpy as np
import scipy.stats

def get_timeline_chart(imagegroups, index):
    total_counter = Counter([])
    for images in imagegroups:
        t = [i[4].strftime("%Y-%m-%d") for i in images]
        t_counter = Counter(t)
        total_counter += t_counter

    t = [i[4].strftime("%Y-%m-%d") for i in imagegroups[index]]
    t_counter = Counter(t)
    t_list = [(c[0], t_counter[c[0]], c[1]-t_counter[c[0]]) for c in total_counter.items()]
    timelines = json.dumps(t_list)
    return timelines


def get_w2v_freq_chart(labels, index):
    values = []
    for q in labels[index]:
        values.append(("\n".join(q[0][:5]), q[1]))
    return json.dumps(values)


def get_linechart_cluster_eval(source, model, keyword):
    num_run = 0
    if model == "ap":
        total, xaxis, num, values = cluster_eval(source, model, keyword)
        names = ["%d (%d)" % (xaxis[i], num[i]) for i in range(len(xaxis))]
        data = []
        for n, v in zip(names, values):
            data.append([n, v])

    elif model == "kmeans":
        num_run = 5
        tvalues = []
        for run in range(num_run):
            total, xaxis, num, values = cluster_eval(source, model, keyword)
            tvalues.append(values)
        tvalues.insert(0, np.average(tvalues, axis=0).tolist())

        names = [str(x) for x in xaxis]
        data = []
        for n, tv in zip(names, np.array(tvalues).T):
            row = [v for v in tv]
            row.insert(0, n)
            data.append(row)

    return total, num_run, data


def get_distance_histogram(dlist, thres):
    vlist = [s[2] for s in dlist]
    minv = vlist[0]
    maxv = vlist[-1]
    mean = np.mean(vlist)
    std = np.std(vlist)

    numbins = 50
    bins = np.linspace(minv, maxv, numbins)
    scalef = len(dlist) * (bins[1]-bins[0])
    normdist = scipy.stats.norm(mean, std)
    norm = [normdist.pdf(b)*scalef for b in bins]
    hist, bin_edges = np.histogram(vlist, bins, (minv, maxv))

    diff = [abs(thres-b) for b in bins]
    sigma = [None for i in range(len(bins))]
    sigma_index = diff.index(min(diff)) # 2 sigma line
    sigma[sigma_index] = u"\u00b5-2\u03c3"

    values = zip(bins, sigma, hist.tolist(), norm)
    return list(values)
