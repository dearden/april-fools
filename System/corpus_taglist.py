from corpus import Corpus
from bag_of_x import get_bag_feats
from bag_of_x import remove_ditto


# Corpus for which each feature is the count of a certain tag, can be words or PoS, etc.
class CorpusTagList(Corpus):
    def __init__(self, articles, taglist, kind="wordlist"):
        Corpus.__init__(self, articles)
        for article in articles:
            if article.train:
                self.train_articles.append(article)     # keep a list of all articles in the training set.
            else:
                self.test_articles.append(article)      # keep a list of all articles in the testing set.

        for article in articles:
            if kind == 'wordlist':
                features = get_bag_feats(article.wrd_fql, taglist)
            elif kind == 'poslist':
                features = get_bag_feats(remove_ditto(article.pos_fql), taglist)
            if article.train:  # put feature dict in either testing or training.
                self.train_feats.append(features)
            else:
                self.test_feats.append(features)

        self.feat_names = features.keys()
