from corpus import Corpus
from bag_of_x import make_bag
from bag_of_x import get_bag_of_x


class CorpusBags(Corpus):
    def __init__(self, articles):
        Corpus.__init__(self, articles)
        for article in articles:
            if article.train:
                self.train_articles.append(article)     # keep a list of all articles in the training set.
            else:
                self.test_articles.append(article)      # keep a list of all articles in the testing set.

        wrd_bag = make_bag([a.wrd_fql for a in self.train_articles])
        pos_bag = make_bag([a.pos_fql for a in self.train_articles])
        sem_bag = make_bag([a.sem_fql for a in self.train_articles])

        for article in articles:
            features = get_bag_of_x(article, wrd_bag, pos_bag, sem_bag)
            if article.train:  # put feature dict in either testing or training.
                self.train_feats.append(features)
            else:
                self.test_feats.append(features)

        self.feat_names = features.keys()
