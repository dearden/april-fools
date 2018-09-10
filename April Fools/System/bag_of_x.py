def get_bag_of_x(article, words, pos, sem):
    bow = get_bag_feats(article.wrd_fql, words)
    bop = get_bag_feats(article.pos_fql, pos)
    bos = get_bag_feats(article.sem_fql, sem)

    bag = dict(bow, **bop)
    bag = dict(bag, **bos)
    return bow


def get_bag_feats(tags, bag):
    features = dict()
    for tag in bag:
        if tag in tags:
            features[tag] = tags[tag] / tags['TOTAL']
        else:
            features[tag] = 0
    return features


def make_bag(freqs):
    sums = dict()
    for a in freqs:
        for w in a.keys():
            if w != 'TOTAL':
                if w in sums:
                    sums[w] += a[w]
                else:
                    sums[w] = a[w]
    s = [(k, sums[k]) for k in sorted(sums, key=sums.get, reverse=True)]
    bag = [x[0] for x in s[:1000]]
    return bag
