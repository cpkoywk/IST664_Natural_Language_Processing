import nltk

# movie review sentences
from nltk.corpus import sentence_polarity
import random

## repeat the setup of the movie review sentences for classification
# for each sentence(document), get its words and category (positive/negative)
documents = [(sent, cat) for cat in sentence_polarity.categories()
    for sent in sentence_polarity.sents(categories=cat)]

random.shuffle(documents)

# get all words from all movie_reviews and put into a frequency distribution
#   note lowercase, but no stemming or stopwords
all_words_list = [word for (sent,cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
print(len(all_words))

# get the 1500 most frequently appearing keywords in the corpus (to save time in class)
word_items = all_words.most_common(1500)
word_features = [word for (word,count) in word_items]

# define features (keywords) of a document for a BOW/unigram baseline
# each feature is 'V_keyword' and is true or false depending
# on whether that keyword is in the document
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    return features

# get features sets for a document, including keyword features and category feature
featuresets = [(document_features(d, word_features), c) for (d, c) in documents]

## cross-validation with precision, recall and F-scores##
# use functions from the file NLTK_cross_validation_evaluation.py
# prints precision, recall and F-score for each label

# two labels
label_list = ['pos','neg']
num_folds = 5
cross_validate_evaluate(num_folds, featuresets, label_list)
