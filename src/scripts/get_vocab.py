# Script to build vocabulary
import pandas as pd
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import pickle
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer

min_count = 1
counts = Counter()
vocab = set()

stops = set(stopwords.words('english'))
stops.remove('no')
stops.remove('not')

def process_text(text):
    # Remove stop words or nah?
    text = text.strip().lower()
    text = contractions.fix(text)

    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    text = tokenizer.tokenize(text)
    #text = [w for w in text if w not in set(stopwords.words('english'))]

    return text

# Preprocess each text, count each word, add those above min_count to vocab
text_dirs = pd.read_csv('../data/mimic-cxr-jpg/cxr-study-list.csv')
for p in text_dirs.path:
    text = open(f'../data/mimic-cxr-jpg/{p}').read()
    text = process_text(text)
    counts.update(text)

    
vocab = set([w for w in counts if counts[w] > min_count] + ['<UNK>','<PAD>'])
    
# Save vocab and counts
with open('../data/vocab_raw.txt','wb') as f:
    pickle.dump(vocab, f)
with open('../data/vocab_index_raw.txt','wb') as f:
    vocab_index = {k: i for i, k in enumerate(vocab)}
    pickle.dump(vocab_index, f)
with open('../data/vocab_counts_raw.txt','wb') as f:
    pickle.dump(counts, f)
