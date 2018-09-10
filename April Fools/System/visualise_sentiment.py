from vaderSentiment import vaderSentiment
import matplotlib.pyplot as plt
import matplotlib as mpl


class VaderBreakdown(vaderSentiment.SentimentIntensityAnalyzer):
    def sentiment_breakdown(self, text):
        """
        Create array showing breakdown of sentiment in a text.
        Code modified version of 'polarity_scores' method.
        :param text: The text (ideally tweet or sentence) to get breakdown of.
        :return: Array of sentiments over text.
        """
        sentitext = vaderSentiment.SentiText(text)

        sentiments = []
        words = sentitext.words_and_emoticons
        for item in words:
            valence = 0
            i = words.index(item)
            if (i < len(words) - 1 and item.lower() == "kind" and\
                words[i + 1].lower() == "of") or\
                item.lower() in vaderSentiment.BOOSTER_DICT:
                    sentiments.append(valence)
                    continue

            sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)

        sentiments = self._but_check(words, sentiments)
        return sentitext.words_and_emoticons, sentiments


# plot a line graph of the sentiment
def plot_sentence_graph(words, sentiments):
    plt.plot(sentiments)
    plt.xticks(range(len(words)), words, size='small')
    plt.show()
    return


# print a coloured sentence highlighting the different sentiment of words.
def print_coloured_sentence(sentence, index=''):
    anal = VaderBreakdown()
    words, sentiments = anal.sentiment_breakdown(sentence)

    colours = ['\033[94m' if c > 0 else '\033[91m' if c < 0 else '\033[93m' for c in sentiments]
    end = '\033[0m'
    sent = ''
    for w, c in zip(words, colours):
        sent = sent + c + w + end + " "

    print(index + '\t' + sent)
    return sent


if __name__ == "__main__":
    sentence = "I love how awful everything is right now"
    anal = VaderBreakdown()
    words, sentiments = anal.sentiment_breakdown(sentence)
    print_coloured_sentence(sentence)
    plot_sentence_graph(words, sentiments)
