import re

reg_num = r'\b\d+\b'
num_pattern = re.compile(reg_num)
no_hyphon = r'([Tt]wen|[Tt]hir|[Ff]our|[Ff]if|[Ss]ix|[Ss]even|[Ee]igh|[Nn]ine)ty ([Oo]ne|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive|[Ss]ix|[Ss]even|[Ee]ight|[Nn]ine)'
hyph_pattern = re.compile(no_hyphon)
sent_end = r'[\.\?\!]'
sent_end_patt = re.compile(sent_end)
reg_date_num = r'((?<=(May) )\d+|(?<=(June|July) )\d+|(?<=(March|April) )\d+|(?<=(August) )\d+|(?<=(January|October) )\d+|(?<=(February|November|December) )\d+|((?<=(September)) )\d+)|\d+(?= ((January|February|March|April|May|June|July|August|September|October|November|December)))'
pat_date_num = re.compile(reg_date_num)

reg_th = r'(January|February|March|April|May|June|July|August|September|October|November|December) (\d+)(st|nd|rd|th)'
th_pattern = re.compile(reg_th)
reg_d_of_m = r'(\d+)(st|nd|rd|th)?( of)? (January|February|March|April|May|June|July|August|September|October|November|December)'
of_pattern = re.compile(reg_d_of_m)
reg_yester = r'([Yy]ester|[Tt]o)day|[Tt]omorrow'
yester_pat = re.compile(reg_yester)

reg_word = r'\b\w+\b'
pat_word = re.compile(reg_word)
capitalised = r'[A-Z].*'
cap_pat = re.compile(capitalised)


def get_ap_features(body, head):
    feats = dict()
    feats['numbers'] = get_ap_numbers(body)
    feats['dates'] = get_ap_dates(body)
    # feats['months'] = get_ap_months(body)
    feats['titles'] = get_ap_titles(head)
    # return sum(feats.values()) / len(feats)
    return feats


def get_ap_numbers(text):
    bad_numbers = 0

    dates = pat_date_num.finditer(text)
    # Find all numbers < 10 that haven't been spelt out.
    for m in num_pattern.finditer(text):
        if int(m.group(0)) < 10:
            for d in dates:
                if m.start() != d.start():
                    # print("Bad Number Spelled: " + m.group(0))
                    bad_numbers += 1

    # Find non written out numbers starting sentences.
    for m in num_pattern.finditer(text):
        s = m.start()
        p = text[s-2:s]
        if sent_end_patt.findall(p):
            # print("Bad Number Start: " + m.group(0))
            bad_numbers += 1

    # Find all non-hyphenated numbers.
    hyph_match = hyph_pattern.finditer(text)
    for m in hyph_match:
        # print("Bad Number Hyphen: " + m.group(0))
        bad_numbers += 1

    return int(bad_numbers > 0)


def get_ap_dates(text):
    bad_dates = 0

    # Count the dates with "st, nd, rd, th"
    for m in th_pattern.finditer(text):
        bad_dates += 1
        # print("Bad Date: " + m.group(0))

    # Count the dates that have the date before the month.
    for m in of_pattern.finditer(text):
        bad_dates += 1
        # print("Bad Date: " + m.group(0))

    # Count the instances of "yesterday, today, and tomorrow"
    for m in yester_pat.finditer(text):
        bad_dates += 1
        # print("Bad Date: " + m.group(0))

    return int(bad_dates > 0)


def get_ap_months(text):
    return 1


def get_ap_titles(text):
    bad_title = 0

    words = pat_word.findall(text)

    # Check first and last word of title are capitalised.
    if not (cap_pat.fullmatch(words[0]) and cap_pat.fullmatch(words[-1])):
        # print("Bad Title: %s %s" % (words[0], words[-1]))
        bad_title += 1

    # Check remainder of words.
    for word in words[1:-1]:
        # I is fine to be capitalised.
        if word == "I":
            continue
        # Words of < 4 letters should not be capitalised.
        if len(word) < 4 and cap_pat.fullmatch(word):
            bad_title += 1
            # print("Bad Title: %s" % (word))
        # Words of >= 4 letters should be capitalised.
        elif len(word) >= 4 and not cap_pat.fullmatch(word):
            bad_title += 1
            # print("Bad Title: %s" % (word))

    return int(bad_title > 0)


if __name__ == "__main__":
    perfect_title = "A Perfect Picture of a Storm for you and I and Me"
    perfect_body = "Twenty dogs were released from the kennels of Buckingham palace on May 5. This comes after the queen reached the age of 80 and decreed one hound should be released for every four years of her life."
    print(get_ap_features(perfect_body, perfect_title))

    bad_title = "the Worst title On this earth"
    bad_body = "22 dogs were let go yesterday. Twenty of them were big ones and 3 of them were small. The 5th of May is when this will happen. 5 May."
    print(get_ap_features(bad_body, bad_title))
