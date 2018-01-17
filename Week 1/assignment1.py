'''
Steps:
get the text with nltk.corpus.gutenberg.raw()
get the tokens with nltk.word_tokenize()
get the words by using w.lower() to lowercase the tokens
make the frequency distribution with FreqDist
get the 30 top frequency words with most_common(30) and print the word, frequency pairs
'''
#Import required modules
import nltk
from nltk import FreqDist
from nltk.corpus import brown

#check what file they've got in gutenberg
nltk.corpus.gutenberg.fileids()

#I will pick 'shakespeare-hamlet.txt'
file0 = nltk.corpus.gutenberg.fileids()[-3]
#file0 = 'shakespeare-hamlet.txt'

#1. get the text with nltk.corpus.gutenberg.raw()
hamlettext=nltk.corpus.gutenberg.raw(file0)

#2. Get the tokens with nltk.word_tokenize()
hamlettokens = nltk.word_tokenize(hamlettext)

#3. Get the words by using w.lower() to lowercase the tokens
hamletwords = [w.lower() for w in emmatokens]

#4. make the frequency distribution with FreqDist
fdist = FreqDist(hamletwords)
fdistkeys=list(fdist.keys())

#5. get the 30 top frequency words with most_common(30) and print the word, frequency pairs
top30keys=fdist.most_common(30)
for pair in top30keys:
    print (pair)
