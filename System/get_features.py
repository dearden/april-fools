from article import Article
import numpy as np
import re
from lxml import etree
import os
import spacy
from nltk.corpus import wordnet
from textstat.textstat import textstat
from nltk.tokenize import word_tokenize
from nltk import bigrams
import nltk
from string import punctuation
from nltk.corpus import wordnet as wn
from ap_features import get_ap_features
from enchant.checker import SpellChecker
from string import punctuation as str_punc
from nltk.corpus import wordnet as wn


# nltk.download('cmudict')
nlp = spacy.load('en_core_web_md', disable=['parser'])
print("Loaded Spacy")

spell_check = SpellChecker("en_GB")
print("Loaded Spell Checker")

# function word list from https://ieeexplore.ieee.org/document/6234420/
with open("function_words.txt") as func_file:
    function_words = func_file.readlines()
    function_words = [line.strip() for line in function_words]
print("Loaded Function Words")

# Profanity list from https://github.com/RobertJGabriel/Google-profanity-words/blob/master/list.txt
with open("profanity.txt") as prof_file:
    profanity = prof_file.readlines()
    profanity = [line.strip() for line in profanity]
print("Loaded F***ing Profanity List")

quote_regex = r' <\/?quote> '

superlatives = r'(DAT|JJT|RGT|RRT)(\d\d)?'
degree_adverbs = r'(RG(QV?|R|T)?)(\d\d)?'
comparative_adverbs = r'(RGR|RRR)(\d\d)?'
proper_nouns = r'(NP[12])(\d\d)?'
dates = r'(NP(D[12]|M[12]))(\d\d)?'
numbers = r'(M[CDF]\w*)(\d\d)?'
fp_pronouns = r'(PPI\w+)(\d\d)?'
negations = r'(XX)(\d\d)?'

all_emotion = r'E[\w\.\+-]+'
emotion_general = r'E1[\+-]'
pos_emotion = r'E[2-6](\.[12])?\++'
neg_emotion = r'E[2-6](\.[12])?-+'
time_related = r'T[1-4][\.\d\+-]*'
sense_words = r'X3[\.\d\+-]*'
movement_words = r'M[1-6]'
relationships = r'S3.*'

spatial_words = r'RL(\d\d)?|ND1(\d\d)?|NNL[12](\d\d)?|M[78]'

# Imagination subject to change
inf_conjunctions = r'CC(\d\d)?'
ima_conjunctions = r'(CCB|CS)(\d\d)?'
conjunctions = r'C\w+'
inf_verb = r'VVN(\d\d)?'
ima_verb = r'(VV[^N]\w?|VM)(\d\d)?'
verbs = r'V\w+'
prepositions = r'I\w+'
articles = r'A\w+'
ima_determiners = r'DA1|DB\w?'
determiners = r'D\w+'
adjectives = r'J\w+'
nouns = r'N\w+'
pronouns = r'P\w+'
adverbs = r'R\w+'

exaggeration = r'A13\.[237]'    # Boosters, Maximisers, and Minimisers.
vague_degree = r'A13\.[145]'    # Non-specific, Approximators, Compromisers

re_punctuation = r'[\.\,\"\'\“\”\`\!\?]+'


def get_features(article):
    features = dict()

    # CLAWS features
    features['superlatives'] = get_tag_group(article.body_pos_fql, superlatives)
    # features['degree_adverbs'] = get_tag_proportion(article.body_pos_fql, degree_adverbs, adverbs)
    features['degree_adverbs'] = get_tag_group(article.body_pos_fql, degree_adverbs)
    # features['comparative_adverbs'] = get_tag_proportion(article.body_pos_fql, comparative_adverbs, adverbs)
    features['comparative_adverbs'] = get_tag_group(article.body_pos_fql, comparative_adverbs)
    # features['proper_nouns'] = get_tag_proportion(article.body_pos_fql, proper_nouns, nouns)
    features['proper_nouns'] = get_tag_group(article.body_pos_fql, proper_nouns)
    features['dates'] = get_tag_group(article.body_pos_fql, dates)
    features['numbers'] = get_tag_group(article.body_pos_fql, numbers)
    # features['fp_pronouns'] = get_tag_proportion(article.body_pos_fql, fp_pronouns, pronouns)
    features['fp_pronouns'] = get_tag_group(article.body_pos_fql, fp_pronouns)
    features['negations'] = get_tag_group(article.body_pos_fql, negations)

    # USAS features
    features['pos_emotion'] = get_tag_group(article.body_sem_fql, pos_emotion)           # Possibly make proportion
    features['neg_emotion'] = get_tag_group(article.body_sem_fql, neg_emotion)
    # features['pos_proportion'] = get_tag_proportion(article.body_sem_fql, pos_emotion, all_emotion) # Uncomment to make proportion
    # features['neg_proportion'] = get_tag_proportion(article.body_sem_fql, neg_emotion, all_emotion) # Uncomment to make proportion
    # features['emotion_general'] = get_tag_group(article.body_sem_fql, emotion_general)   # Currently rubbish
    features['time_related'] = get_tag_group(article.body_sem_fql, time_related)
    features['sense_words'] = get_tag_group(article.body_sem_fql, sense_words)
    features['relationships'] = get_tag_group(article.body_sem_fql, relationships)

    # This is just bad in different ways. Not convinced by this feature.
    features['motion_words'] = get_pos_sem_combo(verbs, movement_words, article.index, article.folder, article.pos_fql['TOTAL'])

    # Couple of extra ones (May already be covered by some of the POS features)
    features['exaggeration'] = get_tag_group(article.body_sem_fql, exaggeration)
    features['vague_degree'] = get_tag_group(article.body_sem_fql, vague_degree)

    features['spatial_words'] = get_tag_group(dict(article.body_pos_fql, **article.body_sem_fql), spatial_words)

    # Imagination features
    # features['ima_conjunctions'] = get_tag_proportion(article.body_pos_fql, ima_conjunctions, conjunctions)
    features['ima_conjunctions'] = get_tag_group(article.body_pos_fql, ima_conjunctions)
    # features['inf_verb'] = get_tag_proportion(article.body_pos_fql, inf_verb, verbs)
    features['inf_verb'] = get_tag_group(article.body_pos_fql, inf_verb)
    # features['ima_verb'] = get_tag_proportion(article.body_pos_fql, ima_verb, verbs)
    features['ima_verb'] = get_tag_group(article.body_pos_fql, ima_verb)
    features['preposition'] = get_tag_group(article.body_pos_fql, prepositions)
    features['articles'] = get_tag_group(article.body_pos_fql, articles)
    # features['ima_determiners'] = get_tag_proportion(article.body_pos_fql, ima_determiners, determiners)
    features['ima_determiners'] = get_tag_group(article.body_pos_fql, ima_determiners)
    features['adjectives'] = get_tag_group(article.body_pos_fql, adjectives)

    # features['quote_proportion'] = get_quote_proportion(article.body)
    features['claws_ambiguity'] = get_claws_ambiguity(article.index, article.folder, article.pos_fql['TOTAL'])
    features['usas_ambiguity'] = get_usas_ambiguity(article.index, article.folder, article.sem_fql['TOTAL'])
    features['synset_ambiguity'] = get_synset_ambiguity(article.wrd_fql)

    # features['quote_proportion'] = get_quote_proportion(article.body)

    features['avg_sentence_len'] = get_avg_sentence_length(article.index, article.folder, article.pos_fql['TOTAL'])

    features['contextual_imbalance_head'] = get_contextual_imbalance(article.head)
    features['contextual_imbalance_body'] = get_contextual_imbalance(article.body)

    features['body_punctuation'] = get_punctuation(article.body)
    features['head_punctuation'] = get_punctuation(article.head)

    # 1 - reading ease (so that higher number means more complex)
    features['readability'] = 1 - abs(get_readability(article.full_text)) / 150

    ap_feats = get_ap_features(article.body, article.head)
    features['ap_num'] = ap_feats['numbers']
    features['ap_date'] = ap_feats['dates']
    features['ap_title'] = ap_feats['titles']
    # features['ap_style_feats'] = get_ap_features(article.body, article.head)

    # # These are also a bit dodgy.
    # features['sentiment_shifts'] = get_sentiment_shifts(article.body)
    # features['body_sentiment_score'] = get_sentiment_score(article.body)
    # features['head_sentiment_score'] = get_sentiment_score(article.head)

    # seems to take absolutely bloody ages. doesn't seem to add much.
    features['alliteration'] = get_alliteration(article.head, article.head_wrd_fql['TOTAL'])
    # features['rhyme'] = get_rhyme(article.head)
    features['profanity'] = get_word_list_counts(article.wrd_fql, profanity)

    features['lexical_diversity'] = get_lex_div(article.wrd_fql)
    features['lexical_density'] = get_lex_den(article.wrd_fql)

    features['function_words'] = get_word_list_counts(article.wrd_fql, function_words)

    # features['antonymy'] = get_antonymy(article.head)

    features['spelling_errors'] = get_spelling_errors(article.full_text, article.wrd_fql['TOTAL'])

    print("Completed Article %s\t--\t%s" % (article.index, article.head))

    return features


# gets a single feature value that combines the counts of several tags (USAS, CLAWS, or Words)
def get_tag_group(article_tags, regex):
    count = 0
    for t in article_tags.keys():
        if re.fullmatch(regex, t):
            count += article_tags[t]
    count = count / article_tags["TOTAL"]
    return count


# get a single feature value equal to the frequency of the subset tags over the frequency of the super set tags.
# For example: getting a value for the proportion of pronouns that are first person.
def get_tag_proportion(article_tags, reg_subset, reg_superset):
    sub_count = 0
    super_count = 0
    for t in article_tags.keys():
        if re.fullmatch(reg_subset, t):
            sub_count += article_tags[t]
        if re.fullmatch(reg_superset, t):
            super_count += article_tags[t]

    proportion = 0
    if super_count != 0 and sub_count != 0:
        proportion = sub_count / super_count

    return proportion


def get_pos_sem_combo(reg_pos, reg_sem, i, f, n):
    file_name = str(i) + ".txt.pos.sem"
    with open(os.path.join(f, file_name)) as file:
        text = file.readlines()
        text = [line.strip() for line in text]

    # Pattern of the lines from the POS file we are interested in.
    reg_line = re.compile(r'(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(.+)')

    # Number of ambiguous (>1 USAS tag <100% confidence)
    count = 0
    # Loop through lines of POS file.
    for line in text:
        line = line.strip()
        # match the pattern of the lines we are interesting and define groups within.
        m = reg_line.match(line)
        if m:
            pos = m.group(3)
            sems = m.group(5)                           # Get the group we're interested in (possible tags)
            sem = sems.split(" ")[0]                    # Split all candidate tags into a list
            sem = sem.split("/")                        # If it's in multiple categories, get both tags
            sem = [s.split("[")[0] for s in sem]        # Sort out the gubbins for multiword expressions.

            sem_match = False
            for s in sem:
                if re.match(reg_sem, s):
                    sem_match = True

            if re.match(reg_pos, pos) and sem_match:
                count += 1

    count = count / n  # Normalise to number of tags.
    return count


def get_word_list_counts(article_words, word_list):
    count = 0
    for word in word_list:
        if word in article_words:
            count += article_words[word]

    return count / article_words['TOTAL']


def get_quote_proportion(text):
    START = "<quote> "
    END = "</quote>"

    quotes = []
    qs = []
    # loop through the text
    for i in range(0, len(text) - len(END)):
        chunk = text[i:i+len(END)]              # check a chunk of text
        if chunk == START:                      # If chunk == <quote>, that's the start of a quote.
            qs.append(i)                        # Add index to list of starting points.
        elif chunk == END and len(qs):          # If chunk == </quote>, that's the end of a quote.
            end = i
            start = qs.pop() + len(START)       # Pop the last starting point off the list.
            quote = text[start:end]             # Get the quote.
            quotes.append(quote)
        if len(qs) > 1:
            True

    # Go through the quotes and remove quotes within quotes. (they will be included in parent quote)
    i = 0
    while i < len(quotes) - 2:
        if quotes[i] in quotes[i+1]:
            quotes.pop(i)
        i += 1

    # Remove all the quote tags from the quotes and add the number of characters to the tally.
    quote_chars = 0
    for quote in quotes:
        quote = re.sub(quote_regex, '', quote)
        quote_chars += len(quote)

    # Remove tags from the text and count the characters.
    quoteless_text = re.sub(quote_regex, '', text)
    text_chars = len(quoteless_text)

    # Calculate proportion of characters in text contained within quotes.
    quote_proportion = quote_chars / text_chars

    return quote_proportion


def get_claws_ambiguity(i, f, n):
    file_name = str(i) + ".txt.pos"
    with open(os.path.join(f, file_name)) as file:
        text = file.readlines()
        text = [line.strip() for line in text]

    # Pattern of the lines from the POS file we are interested in.
    reg_line = re.compile(r'(\d+)\s+(\d+)\s+(\S+)[\s<>]+(\d+)\s+(.+)')

    # Regex pattern of a chosen tag and its certainty value.
    reg_certainty = re.compile(r'\[.+\/(\d+)\]')

    # Number of ambiguous (>1 POS tag <100% confidence)
    ambiguous = 0
    # Loop through lines of POS file.
    for line in text:
        line = line.strip()
        # match the pattern of the lines we are interesting and define groups within.
        m = reg_line.match(line)
        if m:
            tags = m.group(5)               # Get the group we're interested in (possible tags)
            tags = tags.split(" ")          # Split all candidate tags into a list
            if len(tags) > 1:               # If there is >1 candidate POS tag.
                cert = reg_certainty.match(tags[0]).group(1)    # Find certainty that tag is the first candidate.
                if int(cert) < 100:                             # If certainty is <100, then it is ambiguous.
                    ambiguous += 1
    ambiguity = ambiguous / n
    return ambiguity


def get_usas_ambiguity(i, f, n):
    file_name = str(i) + ".txt.pos.sem"
    with open(os.path.join(f, file_name)) as file:
        text = file.readlines()
        text = [line.strip() for line in text]

    # Pattern of the lines from the POS file we are interested in.
    reg_line = re.compile(r'(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(.+)')

    # Number of ambiguous (>1 USAS tag <100% confidence)
    ambiguous = 0
    # Loop through lines of POS file.
    for line in text:
        line = line.strip()
        # match the pattern of the lines we are interesting and define groups within.
        m = reg_line.match(line)
        if m:
            tags = m.group(5)           # Get the group we're interested in (possible tags)
            tags = tags.split(" ")      # Split all candidate tags into a list
            if len(tags) > 1:           # If there is more than 1 candidate, it is ambiguous.
                ambiguous += 1
    ambiguity = ambiguous / n           # Normalise to number of tags.
    return ambiguity


def get_synset_ambiguity(wrds):
    avg_n = 0
    for word in wrds.keys():
        if word == "TOTAL":
            continue
        for w in word.split("_"):           # For multiword expressions, looks at each constituent word.
            n_syn = len(wn.synsets(w))      # Get number of synsets.
            avg_n += n_syn * wrds[word]     # Add to average number of synsets.
    avg_n = avg_n / wrds['TOTAL']
    avg_n = avg_n / 15                      # Divide by arbitrary number
    if avg_n > 1: avg_n = 1                 # Make sure it's not above 1
    return avg_n


# Might have to look at pos file for sentence lengths.
def get_avg_sentence_length(i, f, n):
    file_name = str(i) + ".txt.pos"

    # Pattern of the lines from the POS file we are interested in.
    reg_word_line = re.compile(r'(\d+)\s+(\d+)\s+(\S+)[\s<>]+(\d+)\s+(.+)')
    reg_sep_line = re.compile(r'(\d+)\s+(\d+)\s+-+')

    reg_end = re.compile(r'[\.\!\?]')
    reg_word = re.compile(r'[A-Za-z]+')

    with open(os.path.join(f, file_name)) as file:
        text = file.readlines()
        text = [line.strip() for line in text]

    words = []
    num_sent = 0

    # Loop through all words in POS file.
    for line in text[2:]:
        m = reg_word_line.match(line)               # Get the current line.
        if m:
            words.append(m.group(3))                # Add token to list of tokens so far.
            # print(words[-1])
            continue
        else:
            m = reg_sep_line.match(line)            # If line doesn't match, match the separating line pattern.

        if not words:
            continue                                # If there's no words, loop back immediately.

        i = 1
        while not reg_word.match(words[-i]):        # Keep checking the previous token while it isn't a word. (ie punct)
            if reg_end.match(words[-i]):            # If the token is an "end sentence" token.
                num_sent += 1                           # add to sentence count.
                break
            i += 1

    if num_sent == 0:
        True

    # Divide average by arbitrarily long sentence.
    avg_sent_len = n / num_sent / 150

    return avg_sent_len


def get_contextual_imbalance(text):
    # window = how many words adjacent to check for similarity.
    window = 3
    doc = nlp(text)
    # Keep only content words that are made of alphabet characters.
    doc = [token for token in doc if token.is_alpha and not token.is_stop and not(token.text.lower() in function_words) and len(token.text) > 1]
    av_sim = 0

    for w in range(1, window+1):
        # calculate average similarity between adjacent content words in the corpus.
        for i in range(len(doc) - w):
            # print("%s %s %f" % (doc[i].text, doc[i+w].text, doc[i].similarity(doc[i+w])))
            av_sim += doc[i].similarity(doc[i+w]) / (len(doc) * window)

    # Force the value to be between 1 and 0.
    av_sim = (av_sim + 1 )/ 2

    return av_sim


def get_punctuation(text):
    instances = re.findall(re_punctuation, text)
    return len(instances) / len(text)


# Returns the Flesch Reading Ease score of text. (Low is less readable)
def get_readability(text):
    standard = textstat.flesch_reading_ease(text)
    return standard


# problems with the Vader system. Maybe don't use.
def get_sentiment_shifts(text):
    anal = VaderBreakdown()
    words, breakdown = anal.sentiment_breakdown(text)
    sents = [bd for w, bd in zip(words, breakdown) if w not in function_words]

    curr_state = 0
    pn_shifts = 0
    al_shifts = 0
    prev = 0
    for sent in sents:
        if sent > 0:                            # if positive
            if curr_state == -1:                # if most recent state was negative
                pn_shifts += 1                     # add a shift
            curr_state = 1

        elif sent < 0:                          # if negative
            if curr_state == 1:                 # If most recent state was positive
                pn_shifts += 1                     # add a shift
            curr_state = -1

        if sent * prev <= 0 and (sent + prev) != 0:    # if there is any change, count a shift.
            al_shifts += 1

    if al_shifts > 0:
        return pn_shifts / al_shifts
    else:
        return 0


# problems with the Vader system. Maybe don't use.
def get_sentiment_score(text):
    anal = VaderBreakdown()
    score = anal.polarity_scores(text)["compound"]
    return score


def get_alliteration(t, n_words):
    text = remove_quotes(t)
    toks = word_tokenize(text)                          # Tokenise text.
    toks = [t for t in toks if t not in str_punc]
    bis = list(bigrams(toks))                           # Get Bigrams.
    allit = 0

    for bi in bis:
        if bi[0][0].lower() == bi[1][0].lower():        # If a both words in bigram start with same letter.
            allit += 1                                      # Count alliteration.

    return allit / n_words


# https://stackoverflow.com/questions/25714531/find-rhyme-using-nltk-in-python
def doTheyRhyme(word1, word2):
    # first, we don't want to report 'glue' and 'unglue' as rhyming words
    # those kind of rhymes are LAME
    if word1.find(word2) == len(word1) - len(word2):
      return False
    if word2.find(word1) == len(word2) - len(word1):
      return False

    return word1 in rhyme(word2, 1)


# https://stackoverflow.com/questions/25714531/find-rhyme-using-nltk-in-python
def rhyme(inp, level):
    entries = nltk.corpus.cmudict.entries()
    syllables = [(word, syl) for word, syl in entries if word == inp]
    rhymes = []
    for (word, syllable) in syllables:
        rhymes += [word for word, pron in entries if pron[-level:] == syllable[-level:]]
    return set(rhymes)


# Takes a literal eternity.
def get_rhyme(t):
    text = remove_quotes(t)
    words = [c for c in text if c not in punctuation]
    toks = word_tokenize(text)
    for i in range(len(toks) - 1):
        if doTheyRhyme(toks[i], toks[i+1]):
            print(toks[i], toks[i+1])
    return True


def get_lex_div(freqs):
    unique = len(freqs.keys()) - 1      # Get number of unique words.
    lex_div = unique / freqs['TOTAL']   # Divide by number of words.
    return lex_div


def get_lex_den(freqs):
    # Get list of unique words that aren't stopwords.
    unique_non_stop = [x for x in freqs.keys() if x not in function_words and x != 'TOTAL']
    # Lexical density = content words / total words.
    lex_den = sum([freqs[x] for x in unique_non_stop]) / freqs['TOTAL']
    return lex_den


# only finds 4 instances in the corpus. Maybe leave it.
def get_antonymy(t):
    count = 0
    text = remove_quotes(t)
    toks = word_tokenize(text)
    toks = [t for t in toks if t not in function_words]

    for curr in toks:
        curr_syns = wn.synsets(curr)
        antonyms = []
        for s in curr_syns:
            for l in s.lemmas():
                antonyms.extend(l.antonyms())

        for antonym in antonyms:
            if antonym.name() in toks:
                count += 1

    if count > 0:
        print(text)
    return count


def get_spelling_errors(t, n_words):
    n_errors = 0
    text = remove_quotes(t)
    spell_check.set_text(text)
    for error in spell_check:
        n_errors += 1
        # print(error.word)
    # print(n_errors)
    return n_errors / n_words

def remove_quotes(text):
    clean = re.sub(quote_regex, '', text)
    return clean
