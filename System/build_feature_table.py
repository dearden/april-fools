# This file is responsible for building feature tables.
# If you run it, you will create all of the feature tables used in this work.


import os
import csv
from article import Article
from corpus_bags import CorpusBags
from corpus_feats import CorpusFeats
from corpus_taglist import CorpusTagList
import numpy as np


META_PATH = "../Tagged_Corpus/meta.csv"
CORPUS_PATH = "../Tagged_Corpus/"


# Builds a feature table with the given parameters.
# Takes the information about what type of corpus it is and where the corpus is stored.
def build_feature_table(name, kind='feats', meta_path=META_PATH, corpus_path=CORPUS_PATH, taglist=None):
    print("Begin")
    # Open the meta file.
    with open(meta_path) as meta_file:
        reader = csv.reader(meta_file, delimiter=';')
        meta = [line for line in reader]
    print("Read in metadata")

    articles = []

    # Create article objects based on the metadata.
    print("Begin creating articles")
    for item in meta:
        article = Article(item[0], corpus_path + str(item[0]), item[5], item[2], item[6])
        articles.append(article)
        print("Read in Article %s\t--\t%s" % (article.index, article.head))
    print("All articles created")

    # Tries to create the corpus, making a feature representation.
    print("Begin creating corpus")
    if kind == 'feats':
        corpus = CorpusFeats(articles)
    elif kind in ['wordlist', 'poslist'] and taglist:
        corpus = CorpusTagList(articles, taglist, kind)
    elif kind in ['bow', 'bop', 'bos', 'bag']:
        corpus = CorpusBags(articles, kind=kind)
    else:
        print("Do not understand what feature set you want.")
    print("Corpus created")

    # Gets the training and test tables. (it is worth noting that we do not use the test sets in this work)
    train = corpus.get_training_table().tolist()
    train_labels = corpus.get_training_labels()
    train_indices = corpus.get_training_indices()
    test = corpus.get_testing_table().tolist()
    test_labels = corpus.get_testing_labels()
    test_indices = corpus.get_testing_indices()
    feature_names = corpus.get_feature_names()

    print("Commence Training Table Creation")
    # Add feature names and labels to training feature table
    train_table = [feature_names] + train
    train_table[0] = ['index'] + train_table[0] + ['class']
    train_table[1:] = [[index] + row + [label] for index, row, label in
                       zip(train_indices, train_table[1:], train_labels)]

    # Write the feature table to a file.
    with open(name + "_train.csv", "w") as train_file:
        writer = csv.writer(train_file)
        writer.writerows(train_table)
    print("Training Table Complete")

    if len(test) > 0:
        print("Commence Testing Table Creation")
        # Add feature names and labels to testing feature table
        test_table = [feature_names] + test
        test_table[0] = ['index'] + test_table[0] + ['class']
        train_table[1:] = [[index] + row + [label] for index, row, label in
                           zip(test_indices, test_table[1:], test_labels)]
        with open(name + "_test.csv", "w") as test_file:
            writer = csv.writer(test_file)
            writer.writerows(test_table)
        print("Testing Table Complete")
    else:
        print("Nothing in test table so won't bother.")
    print("Finished")

    return corpus


if __name__ == "__main__":
    # set fake news corpus meta info.
    fake_news_meta_path = "../Tagged_Fake_News/meta.csv"
    fake_news_corp_path = "../Tagged_Fake_News/"

    build_feature_table("feats")                # Build hoax set table for AF.
    build_feature_table("bow", kind='bow')      # Build BoW table for AF.

    # Do the same for Fake News.
    build_feature_table("fake_news_feats", meta_path=fake_news_meta_path, corpus_path=fake_news_corp_path)
    build_feature_table("fake_news_bow", meta_path=fake_news_meta_path, corpus_path=fake_news_corp_path, kind='bow')


    # Read in Function Word list.
    # function word list from https://ieeexplore.ieee.org/document/6234420/
    with open("function_words.txt") as func_file:
        function_words = func_file.readlines()
        function_words = [line.strip() for line in function_words]

    # Read in CLAWS taglist.
    with open("../Word_Lists/pos.txt") as pos_file:
        claws_tags = pos_file.readlines()
        claws_tags = [line.strip() for line in claws_tags]

    # Build feature tables for AF for Function Words and BoPoS.
    build_feature_table("function_words", kind='wordlist', taglist=function_words)
    build_feature_table("bop", kind='poslist', taglist=claws_tags)

    # Build feature tables for Fake News for Function Words and BoPoS.
    build_feature_table("fake_news_function_words", meta_path=fake_news_meta_path, corpus_path=fake_news_corp_path, kind='wordlist', taglist=function_words)
    build_feature_table("fake_news_bop", meta_path=fake_news_meta_path, corpus_path=fake_news_corp_path, kind='poslist', taglist=claws_tags)



