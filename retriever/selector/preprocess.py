import os
import sys
import spacy
import copy
import json
import math
import glob
import wikiwords
import re
import jsonlines
import string
exclude = set(string.punctuation)
import pickle

from collections import Counter
from tqdm import tqdm 


class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """
        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups

class SpacyTokenizer():

    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        """
        model = kwargs.get('model', 'en')
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        nlp_kwargs = {'parser': False}
        if not {'lemma', 'pos', 'ner'} & self.annotators:
            nlp_kwargs['tagger'] = False
        if not {'ner'} & self.annotators:
            nlp_kwargs['entity'] = False
        self.nlp = spacy.load(model, **nlp_kwargs)

    def tokenize(self, text):
        # We don't treat new lines as tokens. Besides, need to remove space at the beginning.
        clean_text = text.replace('\n', ' ').replace('\t', ' ').replace('/', ' / ').strip()
        clean_text = re.sub(r'[^\x00-\x7F]+',' ', clean_text)
        clean_text = clean_text.lstrip(' ')
        # remove consecutive spaces
        if clean_text.find('  ') >= 0:
            clean_text = ' '.join(clean_text.split())
        tokens = self.nlp.tokenizer(clean_text)
        if {'lemma', 'pos', 'ner'} & self.annotators:
            self.nlp.tagger(tokens)
        if {'ner'} & self.annotators:
            self.nlp.entity(tokens)

        data = []
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i].idx
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1].idx
            else:
                end_ws = tokens[i].idx + len(tokens[i].text)

            data.append((
                tokens[i].text,
                text[start_ws: end_ws],
                (tokens[i].idx, tokens[i].idx + len(tokens[i].text)),
                tokens[i].tag_,
                tokens[i].lemma_,
                tokens[i].ent_type_,
            ))

        # Set special option for non-entity tag: '' vs 'O' in spaCy
        return Tokens(data, self.annotators, opts={'non_ent': ''})


TOK = None

def init_tokenizer():
    global TOK
    TOK = SpacyTokenizer(annotators={'pos', 'lemma', 'ner'})


digits2w = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three',
            '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
def replace_digits(words):
    global digits2w
    return [digits2w[w] if w in digits2w else w for w in words]

def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        'words': replace_digits(tokens.words()),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(), # keep capitalization is good for NER?
    }
    return output

# 
from utils import is_stopword, is_punc, is_science_term
def compute_features(q_dict, c_dict):
    # in_c, lemma_in_c, tf
    c_words_set = set([w.lower() for w in c_dict['words']])
    in_c = [int(w.lower() in c_words_set and not is_stopword(w) and not is_punc(w)) for w in q_dict['words']]
    
    c_words_set = set([w.lower() for w in c_dict['lemma']])
    lemma_in_c = [int(w.lower() in c_words_set and not is_stopword(w) and not is_punc(w)) for w in q_dict['lemma']]

    # tf = [0.1 * math.log(wikiwords.N * wikiwords.freq(w.lower()) + 10) for w in q_dict['words']] 
    tf = [wikiwords.freq(w.lower()) for w in q_dict['words']]
    # tf = [float('%.2f' % v) for v in tf]
    
    q_words = Counter(filter(lambda w: not is_stopword(w) and not is_punc(w), q_dict['words']))
    from conceptnet import concept_net
    q_c_relation = concept_net.p_q_relation(q_dict['words'], c_dict['words'])
    assert len(lemma_in_c) == len(in_c) and len(tf) == len(in_c)
    assert len(tf) == len(q_c_relation)
    
    q_is_science_term = [is_science_term(w) for w in q_dict['words']]
    q_is_cand = [1 if not is_punc(w) and not is_stopword(w) else 0 for w in q_dict['words']]
    
    return {
        'in_c': in_c,
        'lemma_in_c': lemma_in_c,
        'tf': tf,
        'q_c_relation': q_c_relation,
        'q_is_science_term': q_is_science_term,
        'q_is_cand': q_is_cand
    }

def get_example(d_id, q_id, c_id, d_dict, q_dict, c_dict, label):
    return {
            'id': d_id + '_' + q_id + '_' + c_id,
            'd_words': ' '.join(d_dict['words']),
            'd_pos': d_dict['pos'],
            'd_ner': d_dict['ner'],
            'q_words': ' '.join(q_dict['words']),
            'q_pos': q_dict['pos'],
            'c_words': ' '.join(c_dict['words']),
            'label': label
        }

## do not use d_dict (i.e. information from paragraph)
def get_selector_example(q_id, q_dict, c_dict, label):
    return {
        'id': q_id,
        'q_words': ' '.join(q_dict['words']),
        'q_pos': q_dict['pos'],
        'q_ner': q_dict['ner'],
        'c_words': ' '.join(c_dict['words']),
        'label': label
        #'answer': answer
    }

           
def preprocess_selector_dataset(path, is_test_set=False):
    filename = path.split('/')[-1]
    # write to ./data
    writer = open('./data/' + filename.replace('.jsonl', '') + '-processed.json', 'w', encoding='utf-8')
    ex_cnt = 0
    with jsonlines.open(path) as reader:
        for q_id, d in enumerate(reader):
            question_toks = tokenize(d['question']) # tokenize the question
            term_toks = [] # tokenize each words
            words = d['words']
            scores = d['scores']
            for word in words:
                term_toks.append(tokenize(word))
            t1 = [w for w in question_toks['words']]
            t2 = []
            scores2 = [] # new scores
            for toks, score in zip(term_toks, scores):
                for w in toks['words']:
                    t2.append(w)
                    scores2.append(score)
            # modify scores for punctuation        
            for i,w in enumerate(t1):
                if w in exclude:
                    scores2[i] = 0.0
            c_dict = tokenize(d['choices'])
            label = [score > 0.5 for score in scores2] 
            q_dict = question_toks
            example = get_selector_example(q_id, q_dict, c_dict, label)
            example.update(compute_features(q_dict, c_dict)) # compute features
            writer.write(json.dumps(example))
            writer.write('\n')
            ex_cnt += 1
    print('Found %d examples in %s...' % (ex_cnt, path))
    writer.close()   
    

def preprocess_arc_dataset(path, is_test_set=False):
    filename = path.split('/')[-1]
    writer = open('./data/' + filename.replace('.jsonl', '') + '-processed.json', 'w', encoding='utf-8')
    ex_cnt = 0
    with jsonlines.open(path) as reader:
        for q_id, d in enumerate(reader):
            question_toks = tokenize(d['question']['stem']) # tokenize the question
            choices = [c['text'] for c in d['question']['choices']]
            c_dict = tokenize(" ".join(choices))
            label = [1] * len(question_toks['words']) # question_toks['words'] is words
            q_dict = question_toks
            example = get_selector_example(q_id, q_dict, c_dict, label)
            example.update(compute_features(q_dict, c_dict)) # compute features
            writer.write(json.dumps(example))
            writer.write('\n')
            ex_cnt += 1
    print('Found %d examples in %s...' % (ex_cnt, path))
    writer.close()   


    
# build from all json file of multiple datasets
def build_vocab_from_raw_dataset():
    import utils
    word_cnt = Counter()
    
    arc_path = '../arc-solver/data/ARC-V1-Feb2018/ARC-Challenge'
    selector_path = './data'

    # read arc
    print("load arc data...")
    for mode in ['Train','Dev','Test']:

        '''    
        data_path = os.path.join(arc_path, 'ARC-Challenge-' + mode + '_with_paras_default.jsonl')
        with jsonlines.open(data_path) as reader:
            for obj in tqdm(reader):
                d_dict = tokenize(obj['para'].replace('\n', ' '))
                word_cnt += Counter(d_dict['words'])
                choices = obj['question']['choices']
                q_dict = tokenize(obj['question']['stem'])
                word_cnt += Counter(q_dict['words'])
                for c_id, choice in enumerate(choices):
                    choice_text = choice['text']
                    c_dict = tokenize(choice_text)
                    word_cnt += Counter(c_dict['words'])
        '''
        data_path = os.path.join(arc_path, 'ARC-Challenge-' + mode + '.jsonl')
        with jsonlines.open(data_path) as reader:
            for obj in tqdm(reader):
                choices = obj['question']['choices']
                q_dict = tokenize(obj['question']['stem'])
                word_cnt += Counter(q_dict['words'])
                for c_id, choice in enumerate(choices):
                    choice_text = choice['text']
                    c_dict = tokenize(choice_text)
                    word_cnt += Counter(c_dict['words'])
       
 
    # read selector dataset
    selector_word_cnt = Counter()
    print("load selector data...")
   
    for mode in ['train','dev','test']:
        data_path = os.path.join(selector_path , mode + '-terms.jsonl')
        with jsonlines.open(data_path) as reader:
            for obj in tqdm(reader):
                q_dict = tokenize(obj['question'])
                selector_word_cnt += Counter(q_dict['words'])
                c_dict = tokenize(obj['choices'])
                selector_word_cnt += Counter(c_dict['words'])
         
    # read science term
    print("load science term...")
    with open(os.path.join('./data', '9144-Science-Terms.txt'), 'r') as f:
        lines = f.readlines()
        terms = []
        for term in lines:
            terms.extend(term.strip().split()) 
        
    # combine words in all datasets
    for key, val in word_cnt.most_common():
        utils.vocab.add(key)
    for key, val in selector_word_cnt.most_common():
        utils.vocab.add(key)
    for term in terms:
        utils.vocab.add(term)
    
    print('Vocabulary size: %d' % len(utils.vocab))
    writer = open('./data/vocab', 'w', encoding='utf-8')
    writer.write('\n'.join(utils.vocab.tokens()))
    writer.close()

    
def preprocess_conceptnet(path):
    import utils
    # build vocab from arc dataset
    build_vocab_from_raw_dataset()
    
    print("vocab built...")
    writer = open('./data/concept.filter', 'w', encoding='utf-8')
    
    def _get_lan_and_w(arg):
        arg = arg.strip('/').split('/')
        return arg[1], arg[2]
    for line in open(path, 'r', encoding='utf-8'):
        fs = line.split('\t')
        relation, arg1, arg2 = fs[1].split('/')[-1], fs[2], fs[3]
        lan1, w1 = _get_lan_and_w(arg1)
        if lan1 != 'en' or not all(w in utils.vocab for w in w1.split('_')):
            continue
        lan2, w2 = _get_lan_and_w(arg2)
        if lan2 != 'en' or not all(w in utils.vocab for w in w2.split('_')):
            continue
        obj = json.loads(fs[-1])
        if obj['weight'] < 1.0:
            continue
        writer.write('%s %s %s\n' % (relation, w1, w2))
    writer.close()

    
if __name__ == '__main__':
    init_tokenizer()
    
    # preprocess conceptnet
    if len(sys.argv) > 1 and sys.argv[1] == 'conceptnet':
        # will build vocab inside 
        concept_path = '../../data/conceptnet'
        print("preprocess conceptnet...") 
        preprocess_conceptnet(os.path.join(concept_path, 'conceptnet-assertions-5.6.0.csv'))
    
    # preprocess arc challenge
    elif len(sys.argv) > 1 and sys.argv[1] == 'challenge': 
        selector_path = '../../data/ARC-Challenge'
        print("preprocess arc challenge dataset...")  
        preprocess_arc_dataset(os.path.join(selector_path, 'ARC-Challenge-Train.jsonl'))
        preprocess_arc_dataset(os.path.join(selector_path, 'ARC-Challenge-Dev.jsonl'))
        preprocess_arc_dataset(os.path.join(selector_path, 'ARC-Challenge-Test.jsonl'), is_test_set=True)

    # preprocess arc easy    
    elif len(sys.argv) > 1 and sys.argv[1] == 'easy': 
        selector_path = '../../data/ARC-Easy'
        print("preprocess arc easy dataset...")        
        preprocess_arc_dataset(os.path.join(selector_path, 'ARC-Easy-Train.jsonl'))
        preprocess_arc_dataset(os.path.join(selector_path, 'ARC-Easy-Dev.jsonl'))
        preprocess_arc_dataset(os.path.join(selector_path, 'ARC-Easy-Test.jsonl'), is_test_set=True)

    # preprocess the label data
    else:
        selector_path = './data'        
        print("preprocess given vocab...")        
        preprocess_selector_dataset(os.path.join(selector_path, 'train-terms.jsonl'))
        preprocess_selector_dataset(os.path.join(selector_path, 'dev-terms.jsonl'))
        preprocess_selector_dataset(os.path.join(selector_path, 'test-terms.jsonl'), is_test_set=True)

        print("load data...") 
        import utils
        train_data = utils.load_data(os.path.join(selector_path, 'train-terms-processed.json'))
        dev_data = utils.load_data(os.path.join(selector_path, 'dev-terms-processed.json'))
        test_data = utils.load_data(os.path.join(selector_path, 'test-terms-processed.json'))
        
        print("build other vocab...") 
        utils.build_vocab(train_data + dev_data + test_data) # word vocab already exists, run to build pos/ner/rel vocab
