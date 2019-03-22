import re


# get a combined bag of words, pos, and semantic tags
def get_bag_of_x(article, words, pos, sem, kind='bow'):
    bow = get_bag_feats(article.wrd_fql, words)
    bop = get_bag_feats(article.pos_fql, pos)
    bos = get_bag_feats(article.sem_fql, sem)

    bag = dict(bow, **bop)
    bag = dict(bag, **bos)

    bag_dic = {"bow": bow, "bop": bop, "bos": bos, "bag": bag}

    out = bag_dic[kind]
    return out


# get a bag of x from a given set of tags to look for and also all the tags.
def get_bag_feats(tags, bag):
    features = dict()
    for tag in bag:
        if tag in tags:
            features[tag] = tags[tag] / tags['TOTAL']
        else:
            features[tag] = 0
    return features


# create a bag of the 1000 most popular tags.
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


# remove ditto tags given by CLAWS to multi-word expressions.
def remove_ditto(tags):
    reg_ditto = r'[A-Z]+[1-9]*\d\d'
    out = dict()
    for tag in tags:
        if re.fullmatch(reg_ditto, tag):
            simp = tag[:-2]
            if simp in out:
                out[simp] += tags[tag]
            else:
                out[simp] = tags[tag]
        else:
            out[tag] = tags[tag]
    return out
