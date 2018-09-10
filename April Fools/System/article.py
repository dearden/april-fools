import os
import re


def process_fql(text):
    pattern = re.compile('(\S+)\s+(\S+)')
    counts = dict()
    for line in text:
        line = line.strip()
        m = pattern.match(line)
        tag = m.group(1)
        count = int(m.group(2))
        counts[tag] = count
    return counts


def combine_freqs(freq1, freq2):
    freq = freq1.copy()
    for item in freq2.keys():
        if item in freq:
            freq[item] += freq2[item]
        else:
            freq[item] = freq2[item]
    return freq


class Article(object):
    def __init__(self, i, folder, af_val, headline, train_test):
        self.index = i
        self.head = headline
        self.folder = folder

        if train_test.lower() == 'train':
            self.train = True
        else:
            self.train = False

        if af_val.lower() == "y":
            self.label = 1
        else:
            self.label = 0

        head_folder = folder + "h"

        for subdir, dirs, files in os.walk(folder):
            for f in files:
                with open(os.path.join(subdir, f)) as file:
                    content = file.readlines()
                    if f == str(i) + ".txt":
                        self.body = "".join(content)
                    if f.__contains__("wrd.fql"):
                        self.body_wrd_fql = process_fql(content)
                    if f.__contains__("pos.fql"):
                        self.body_pos_fql = process_fql(content)
                    if f.__contains__("sem.fql"):
                        self.body_sem_fql = process_fql(content)

        for subdir, dirs, files in os.walk(head_folder):
            for f in files:
                with open(os.path.join(subdir, f)) as file:
                    content = file.readlines()
                    if f.__contains__("wrd.fql"):
                        self.head_wrd_fql = process_fql(content)
                    if f.__contains__("pos.fql"):
                        self.head_pos_fql = process_fql(content)
                    if f.__contains__("sem.fql"):
                        self.head_sem_fql = process_fql(content)

        self.wrd_fql = combine_freqs(self.body_wrd_fql, self.head_wrd_fql)
        self.pos_fql = combine_freqs(self.body_pos_fql, self.head_pos_fql)
        self.sem_fql = combine_freqs(self.body_sem_fql, self.head_sem_fql)

        self.full_text = self.head + "\n" + self.body
