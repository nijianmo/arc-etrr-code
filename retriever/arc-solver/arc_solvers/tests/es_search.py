'''
Author: Jianmo
Add multi-thread search
'''

from elasticsearch import Elasticsearch
from multiprocessing.pool import ThreadPool
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List

import json
import nltk
import os
import re
import string

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))
from arc_solvers.processing.convert_to_entailment import convert_qa_to_entailment


class EsHit:
    def __init__(self, score: float, position: int, text: str, type: str, query: str):
        """
        Basic information about an ElasticSearch Hit
        :param score: score returned by the query
        :param position: position in the retrieved results (before any filters are applied)
        :param text: retrieved sentence
        :param type: type of the hit in the index (by default, only documents of type "sentence"
        will be retrieved from the index)
        """
        self.score = score
        self.position = position
        self.text = text
        self.type = type
        self.query = query


class EsSearch:
    def __init__(self,
                 es_client: str = "localhost",
                 indices: str = "arc_corpus",
                 max_question_length: int = 1000,
                 max_hits_retrieved: int = 500,
                 max_hit_length: int = 300,
                 max_hits_per_choice: int = 100):
        """
        Class to search over the text corpus using ElasticSearch
        :param es_client: Location of the ElasticSearch service
        :param indices: Comma-separated list of indices to search over
        :param max_question_length: Max number of characters used from the question for the
        query (for efficiency)
        :param max_hits_retrieved: Max number of hits requested from ElasticSearch
        :param max_hit_length: Max number of characters for accepted hits
        :param max_hits_per_choice: Max number of hits returned per answer choice
        """
        self._es = Elasticsearch([es_client], retries=3)
        self._indices = indices
        self._max_question_length = max_question_length
        self._max_hits_retrieved = max_hits_retrieved
        self._max_hit_length = max_hit_length
        self._max_hits_per_choice = max_hits_per_choice
        # Regex for negation words used to ignore Lucene results with negation
        self._negation_regexes = [re.compile(r) for r in ["not\\s", "n't\\s", "except\\s"]]
        
        # calculate tfidf 
        self.tfidf = self.build_tfidf()
        self.feature_names = self.tfidf.get_feature_names()

    def build_tfidf(self):
        def stem_tokens(tokens, stemmer):
            stemmed = []
            for item in tokens:
                stemmed.append(stemmer.stem(item))
            return stemmed

        def tokenize(text):
            tokens = nltk.word_tokenize(text)
            return tokens
            #stems = stem_tokens(tokens, stemmer)
            #return stems
    
        PATH = "/data2/jianmo/reading_comprehension/ARC-V1-Feb2018/ARC-Challenge"
        train_path = os.path.join(PATH, "ARC-Challenge-Train.jsonl")
        
        token_dict = []
        stemmer = PorterStemmer()
        
        with open(train_path, 'r') as f:
            for l in f.readlines():
                d = json.loads(l.strip())
                text = d['question']['stem']
                lowers = text.lower()
                no_punctuation = lowers.translate(str.maketrans('','',string.punctuation))
                token_dict.append(no_punctuation)

        #this can take some time
        tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
        tfs = tfidf.fit_transform(token_dict)    
        return tfidf
        
    # Given a question, extract the top tfidf words
    def get_top_tfidf_words(self, q):
        response = self.tfidf.transform([q])
        candidates = []
        for col in response.nonzero()[1]:
            candidates.append((self.feature_names[col], response[0, col]))
        # sort 
        sorted_candidates = sorted(candidates, key=lambda x:-x[1])
        # aggregate words
        top_words = []
        for c in sorted_candidates[:10]: # top 10 words?
            top_words.append(c[0])
            
        return " ".join(top_words)

    def get_hits_for_question(self, question: str, choices: List[str], method='common') -> Dict[str, List[EsHit]]:
        """
        :param question: Question text
        :param choices: List of answer choices
        :param method: Method to construct query
        :return: Dictionary of hits per answer choice
        """
        choice_hits = dict()
        for choice in choices:
            choice_hits[choice] = self.filter_hits(self.get_hits_for_choice(question, choice, method))
        return choice_hits

    
    def get_hits_for_question_rl(self, queries: List[str]) -> Dict[str, List[EsHit]]:
        """
        :param query: search query
        :return: Dictionary of hits per answer choice
        """
        choice_hits = dict()
        for i, query in enumerate(queries):
            initial_hits = self.get_hits_for_choice_rl(query)
            choice_hits[i] = self.filter_hits(initial_hits)
        return choice_hits    
    

    # Constructs an ElasticSearch query from the input query
    def construct_query(self, query):
        return {"from": 0, "size": self._max_hits_retrieved,
                "query": {
                    "bool": {
                        "must": [
                            {"match": {
                                "text": query
                            }}
                        ]
                    }
                }}   

    # Constructs an ElasticSearch query from the input question and choice
    # Uses the last self._max_question_length characters from the question and requires that the
    # text matches the answer choice and the hit type is a "sentence"
    def construct_qa_query(self, question, choice, method='common', use_filter=True):
        if method == 'common':
            query = question[-self._max_question_length:] + " " + choice
        elif method == 'entailment':
            query = convert_qa_to_entailment(question, choice) # convert qa into hypothesis and use as query
        elif method == 'tfidf':
            top_words = self.get_top_tfidf_words(question)
            query = top_words + " " + choice # construct an ES query from tfidf: tfidf words of question + whole choice
           
        if use_filter:
            return {"from": 0, "size": self._max_hits_retrieved,
                "query": {
                    "bool": {
                        "must": [
                            {"match": {
                                "text": query
                            }}
                        ],
                        "filter": [
                            {"match": {"text": choice}},
                            {"type": {"value": "sentence"}}
                        ]
                    }
                }}
        else:
            return {"from": 0, "size": self._max_hits_retrieved,
                "query": {
                    "bool": {
                        "must": [
                            {"match": {
                                "text": query
                            }}
                        ]
                    }
                }}
    
    
    # Retrieve unfiltered hits for input question and answer choice
    def get_hits_for_choice(self, question, choice, method='common'):
        query = self.construct_qa_query(question, choice, method)
        # query = self.construct_qa_query(question, choice, method='entailment') # entailment baseline
        # query = self.construct_qa_query(question, choice, method='tfidf') # tfidf baseline
        res = self._es.search(index=self._indices, body=query)
        
        hits = []
        for idx, es_hit in enumerate(res['hits']['hits']):
            es_hit = EsHit(score=es_hit["_score"],
                           position=idx,
                           text=es_hit["_source"]["text"],
                           type=es_hit["_type"],
                           query=query)
            hits.append(es_hit)
        return hits

    # Retrieve unfiltered hits for input query
    def get_hits_for_choice_rl(self, query):
        query = self.construct_query(query)
        res = self._es.search(index=self._indices, body=query)
        
        hits = []
        for idx, es_hit in enumerate(res['hits']['hits']):
            es_hit = EsHit(score=es_hit["_score"],
                           position=idx,
                           text=es_hit["_source"]["text"],
                           type=es_hit["_type"],
                           query=query)
            hits.append(es_hit)
        return hits
    

    # Remove hits that contain negation, are too long, are duplicates, are noisy.
    def filter_hits(self, hits: List[EsHit]) -> List[EsHit]:
        filtered_hits = []
        selected_hit_keys = set()
        for hit in hits:
            hit_sentence = hit.text
            hit_sentence = hit_sentence.strip().replace("\n", " ")
            if len(hit_sentence) > self._max_hit_length:
                continue
            for negation_regex in self._negation_regexes:
                if negation_regex.search(hit_sentence):
                    # ignore hit
                    continue
            if self.get_key(hit_sentence) in selected_hit_keys:
                continue
            if not self.is_clean_sentence(hit_sentence):
                continue
            filtered_hits.append(hit)
            selected_hit_keys.add(self.get_key(hit_sentence))
        return filtered_hits[:self._max_hits_per_choice]

    # Check if the sentence is not noisy
    def is_clean_sentence(self, s):
        # must only contain expected characters, should be single-sentence and only uses hyphens
        # for hyphenated words
        return (re.match("^[a-zA-Z0-9][a-zA-Z0-9;:,\(\)%\-\&\.'\"\s]+\.?$", s) and
                not re.match(".*\D\. \D.*", s) and
                not re.match(".*\s\-\s.*", s))

    # Create a de-duplication key for a HIT
    def get_key(self, hit):
        # Ignore characters that do not effect semantics of a sentence and URLs
        return re.sub('[^0-9a-zA-Z\.\-^;&%]+', '', re.sub('http[^ ]+', '', hit)).strip().rstrip(".")
