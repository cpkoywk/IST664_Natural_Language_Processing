
# coding: utf-8

# In[3]:

import nltk
# import wordnet and shorten its name to wn
from nltk.corpus import wordnet as wn


# In[6]:

# for each sense of a word, there is a synset with an id consisting of one of the words,
#    whether it is noun, verb, adj or adverb and a number among the synsets of that word
# given word "dog", returns the ids of the synsets
wn.synsets('dog')


# In[7]:

# given a synset id, find words/lemma names (the synonyms) of the first noun sense of "dog"
wn.synset('dog.n.01').lemma_names()


# In[8]:

# given a synset id, find lemmas of the synset (a lemma pairs a word with a synset)
wn.synset('dog.n.01').lemmas()


# In[9]:

# find synset of a lemma
wn.lemma('dog.n.01.domestic_dog').synset()


# In[10]:

# find lemma names for all senses of a word
for synset in wn.synsets('dog'):
	print (synset, ":  ", synset.lemma_names())


# In[11]:

# find definition of the first noun sense of dog, or namely, the dog.n.01 synset
wn.synset('dog.n.01').definition()


# In[12]:

# display an example of the synset
wn.synset('dog.n.01').examples()


# In[13]:

# or show the definitions for all the synsets of a word
for synset in wn.synsets('dog'):
	print (synset, ":  ", synset.definition())


# In[14]:

# or combine the synonyms/lemma names, definitions and examples
for synset in wn.synsets('dog'):
	print (synset, ":  ")
	print ('     ', synset.lemma_names())
	print ('     ', synset.definition())
	print ('     ', synset.examples())


# In[15]:

##  Lexical relations between synsets in WordNet
# find hypernyms of synsets
dog1 = wn.synset('dog.n.01')
dog1.hypernyms()


# In[16]:

# find hyponyms
dog1.hyponyms()


# In[17]:

# the most general hypernym of a synset
dog1.root_hypernyms()


# In[18]:

# from the wordnet browser, we see that dog1 has two more relations
dog1.part_meronyms()


# In[21]:

# what is this?  check it out 
print (wn.synset('flag.n.07').lemma_names(),wn.synset('flag.n.07').definition(), 
       wn.synset('flag.n.07').examples())


# In[22]:

dog1.member_holonyms()


# In[23]:

# look at another word, the adjective "good"
wn.synsets('good')


# In[25]:

# find antonyms, sometimes need to specify for which lemma the antonym is needed
good1 = wn.synset('good.a.01')
# display synonyms of this synset
good1.lemma_names()


# In[26]:

# the antonym function is defined only on the lemma, not the synset
# find antonym for the first lemma of the synset
print(good1.lemmas())
good1.lemmas()[0].antonyms() 


# In[27]:

# find entailments of verbs
print(wn.synset('walk.v.01').entailments())
print(wn.synset('eat.v.01').entailments())


# In[28]:

# trace paths of a synset by visiting its hypernyms
dog1.hypernyms()


# In[29]:

# number of paths from the synset to the root concept "entity"
paths=dog1.hypernym_paths()
print(len(paths) )
# look at the first path
paths[0]


# In[30]:

# or just list the names in the paths
#list the first path
[synset.name() for synset in paths[0]]


# In[31]:

#list the second path 
[synset.name() for synset in paths[1]] 


# In[32]:

# Word similarity

# define 3 different types of whales
right = wn.synset('right_whale.n.01')
minke = wn.synset('minke_whale.n.01')  
orca = wn.synset('orca.n.01') 


# In[33]:

# look at the paths of these three whales
print(right.hypernym_paths())
print(minke.hypernym_paths())
print(orca.hypernym_paths())


# In[34]:

# find the least ancestor of right and minke, and then right and orca
print(right.lowest_common_hypernyms(minke))
print(right.lowest_common_hypernyms(orca))


# In[35]:

# the function min_depth gives the length of a path from a word to the top of the hierarchy
print(right.min_depth() )
print(wn.synset('baleen_whale.n.01').min_depth() )
print(wn.synset('entity.n.01').min_depth())


# In[36]:

# the path similarity gives a similarity score between 0 and 1
print(right.path_similarity(minke) )
print(right.path_similarity(orca))


# In[37]:

# define 2 more words and look at their similarity
tortoise = wn.synset('tortoise.n.01')
novel = wn.synset('novel.n.01')
# note the least ancestor of these two words
print(right.lowest_common_hypernyms(tortoise))
print(right.lowest_common_hypernyms(novel))


# In[38]:

print(right.path_similarity(tortoise) )
print(right.path_similarity(novel))


# In[39]:

help(wn)


# In[40]:

# first get information content from a general corpus
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')


# In[41]:

# try Resnik Similarity
print(right.res_similarity(orca, brown_ic))
print(right.res_similarity(tortoise, brown_ic))
print(right.res_similarity(novel, brown_ic))


# In[45]:

## SentiWordNet
from nltk.corpus import sentiwordnet as swn
# each word judged to be made up of positive, negative and objective meaning


# In[43]:

# sentiwordnet has the same synsets as wordnet, use wn functions
print(list(swn.senti_synsets('breakdown')))
print(wn.synsets('breakdown'))


# In[46]:

# the print function gives the positive and negative scores
breakdown3 = swn.senti_synset('breakdown.n.03')
print (breakdown3)


# In[47]:

# there are also separate functions for all the scores
print(breakdown3.pos_score())
print(breakdown3.neg_score())
print(breakdown3.obj_score())


# In[49]:

# some more exploration of sentiment scores of words
dogswn1 = swn.senti_synset('dog.n.01')
print(dogswn1)
print(dogswn1.obj_score())


# In[50]:

goodswn1 = swn.senti_synset('good.a.01')
print(goodswn1)
print(goodswn1.obj_score())


# In[51]:

# not all words in WordNet have been scored for sentiment in SentiWordNet
#   but the most recent version has scored a lot more so I don't have an example right now
print(wn.synsets('exuberant'))
ex3 = swn.senti_synset('exuberant.s.03')
print(ex3)


# In[ ]:



