
# coding: utf-8

# In[1]:

import nltk
from urllib import request


# In[2]:

# text from online gutenberg
url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
print(type(raw))
print(len(raw))
print(raw[:200])


# In[3]:

# text from online news article (see NLTK book chapter 3)
blondurl = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = request.urlopen(blondurl).read().decode('utf8')
html[:1000]


# In[4]:

from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'lxml')
# get all the text tags from the html
braw = soup.get_text()
btokens = nltk.word_tokenize(braw)
btokens[:100]


# In[5]:

# to get the path to the directory of the python interpreter:
import os
os.getcwd()
## put the file in that directory
f = open('desert.txt')
rawtext = f.read()
rawtext[:200]


# In[8]:

fin = open('/Users/njmccrac1/AAAdocs/NLPfall2017/labs/LabExamplesWeek3/desert.txt')
rawtext = fin.read()
rawtext[:200]


# In[9]:

## create tokens, and continue to use text
deserttokens = nltk.word_tokenize(rawtext)
text = nltk.Text(deserttokens)
text.concordance('pass')
# close file at the end
fin.close()


# In[11]:

### Stemming and Lemmatization
## get text from a file and create tokens (use \ on PCs, and / on Macs)
fin = open('CrimeAndPunishment.txt')
crimetext = fin.read()
crimetokens = nltk.word_tokenize(crimetext)
print(len(crimetokens))
print(crimetokens[:100])


# In[13]:

#use NLTK's stemmers (section 3.6 in NLTK book)
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()


# In[14]:

# compare Porter and Lancaster stemmers on the tokens
crimePstem = [porter.stem(t) for t in crimetokens]
print('Porter\n', crimePstem[:200])

crimeLstem = [lancaster.stem(t) for t in crimetokens]
print('Lancaster\n', crimeLstem[:200])


# In[15]:

# NLTK has a lemmatizer that uses WordNet as a dictionary
wnl = nltk.WordNetLemmatizer()
crimeLemma = [wnl.lemmatize(t) for t in crimetokens]
print('WordNet Lemmatizer\n', crimeLemma[:200])


# In[ ]:



