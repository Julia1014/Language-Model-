# project.py


import pandas as pd
import numpy as np
import os
import re
import requests
import time

from collections import Counter


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------

# CHECK: do u use regex for this? am I approaching this the right way? how do I implement a "pause" in my request? 
def get_book(url):    	
    # API request
    request = requests.get(url)
    
    # CHECK: Handle the "pause" in my request, is this right??
    time.sleep(2) 

    # extracting the content between START and END comments using regex 
    start_marker = re.findall('\*\*\* START OF THE PROJECT GUTENBERG EBOOK [\d\w\s].+\*\*\*', request.text)[0]
    end_marker = re.findall('\*\*\* END OF THE PROJECT GUTENBERG EBOOK [\d\w\s].+\*\*\*', request.text)[0]
    # excluding the START and END comments from the content 
    start_idx = request.text.find(start_marker) + len(start_marker)
    end_idx = request.text.find(end_marker)
    content = request.text[start_idx : end_idx]
    # replacing any Windows newline characters (`'\r\n'`) with standard newline characters (`'\n'`)
    content = content.replace('\r\n', '\n')
    
    return content


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(text):
    result_string = re.sub(r'\n{2,}', ' \x03 \x02 ', text)
    result_string = re.sub(r'\n', ' ', result_string)
    tokenized = re.findall(r'(\x02|\x03|\b\w+\b|\S)', result_string)
    if tokenized[0] == '\x03':
        tokenized = tokenized[1:]
    elif tokenized[0] != '\x02':
        tokenized = ['\x02'] + tokenized
    if tokenized[-1] == '\x02':
        tokenized = tokenized[:len(tokenized) - 1]
    elif tokenized[0] != '\x03':
        tokenized =  tokenized + ['\x03']
    return tokenized

# def tokenize(book_string):
#     # start of paragraph
#     new_string = re.sub(r'\n{2,}', ' \x03 \x02 ', book_string)
#     # separating punctuation
#     punctuation_pattern = r"[.,'!?;:(){}\[\]<>\/\\|_+=*&^%$#@~-]"
#     separate_punc_string = re.sub(punctuation_pattern, r' \g<0> ', new_string)

#     result_list = re.findall(r'\S+', separate_punc_string)

#     # handling edge cases of extra '\x03' and '\x02'
#     if result_list[0] == '\x03':
#         result_list = result_list[1:]
#     elif result_list[0] != '\x02':
#         result_list = ['\x02'] + result_list
#     if result_list[-1] == '\x02':
#         result_list = result_list[:len(result_list) - 1]
#     elif result_list[0] != '\x03':
#         result_list =  result_list + ['\x03']
#     return result_list

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        # finding unique tokens
        unique_tokens_lst = list(set(tokens))
        # making probabilities 
        total_unique_tokens = len(unique_tokens_lst)
        prob_lst = [1 / total_unique_tokens] * total_unique_tokens
        # zip to dict to series
        dictionary = dict(zip(unique_tokens_lst, prob_lst))
    
        return pd.Series(dictionary)
    
    def probability(self, words):
        total_prob = 1
        for word in words:
            if word not in self.mdl.index: 
                # total probability becomes 0 
                return 0
            else:
                prob = self.mdl[word]
                total_prob *= prob

        return total_prob
        
    def sample(self, M):
        unique_tokens = np.array(self.mdl.index)
        prob_lst = np.array(self.mdl.values)
        random_words_lst = np.random.choice(a= unique_tokens, size= M, p= prob_lst)

        return ' '.join(random_words_lst)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------

class UnigramLM(object):
    
    def __init__(self, tokens):

        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        tokens_and_counts = np.unique(tokens, return_counts=True)
        # gets list of unique tokens 
        index = tokens_and_counts[0]
        # gets list of probabilities
        values = tokens_and_counts[1]/len(tokens)
        return pd.Series(values, index)
    
    def probability(self, words):
        total_prob = 1
        for word in words:
            if word not in self.mdl.index: 
                # total probability becomes 0 
                return 0
            else:
                prob = self.mdl[word]
                total_prob *= prob

        return total_prob
        
    def sample(self, M):
        unique_tokens = np.array(self.mdl.index)
        prob_lst = np.array(self.mdl.values)

        random_words_lst = np.random.choice(a= unique_tokens, size= M, p= prob_lst)
        return ' '.join(random_words_lst)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        # added below
        self.tokens = tokens
        # added above
        
        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def shorten_tuples(self, tuple_list, target_length):
        shortened_tuples = [tuple[:target_length-1] for tuple in tuple_list]
        return shortened_tuples

    def create_ngrams(self, tokens):
        return [tuple(tokens[i : i+self.N]) for i in range(len(tokens) - self.N + 1)]
    
    def train(self, ngrams):
        dictionary = {}
        ngrams_lst = []
        n1grams_lst = []
        prob_lst = []

        ngrams_lst = list(set(ngrams))
        n1grams_lst = self.shorten_tuples(ngrams_lst, self.N)

        n1grams_all = self.shorten_tuples(ngrams, self.N)
        countN = Counter(ngrams)
        countN1 = Counter(n1grams_all)

        dictionary['ngram'] = ngrams_lst
        dictionary['n1gram'] = n1grams_lst

        for i in range(len(ngrams_lst)):
            probN = countN[ngrams_lst[i]]
            probN1 = countN1[n1grams_lst[i]]
            prob_lst.append(probN/probN1)

        dictionary['prob'] = prob_lst
        df = pd.DataFrame(dictionary)
        self.df = df

        return df
    
    def probability(self, words):
        all_probs = 1
        words_short = words

        # while loop to figure out when to use n, tri-, bi-, or uni-gram
        while len(words_short) >= self.N:
            a = self.df.index[self.df['ngram'] == tuple(words_short[-(self.N):])].tolist()
            if not a:
                return 0
            else:
                this_prob = self.df.loc[a[0], 'prob']
                all_probs = all_probs * this_prob
                words_short = words_short[:-1]
        # recursion
        all_probs = all_probs * self.prev_mdl.probability(words_short)
        return all_probs
    
    def sample(self, M):
        # Initialize the sampled sequence with the starting token '\x02'
        sampled_sequence = ['\x02']

        # 
        counter = 2
        while len(sampled_sequence) < (M + 1): # orginal: for _ in range(M - 1):
            # getting current ngram & ngram df
            if new_N == self.N:
                new_N = self.N
            else:
                new_N = self.N - (self.N - counter)
                currGram = NGramLM(new_N, self.tokens)
                currGram_df =currGram.mdl

            # Get the most recent (N-1)-gram from the sampled sequence
            
            prev_ngram = sampled_sequence[-(self.N - 1):]
            print(prev_ngram)
            # Find all N-grams that match the most recent (N-1)-gram
            print(tuple(prev_ngram))
            matching_ngrams = currGram_df[currGram_df['n1gram'] == tuple(prev_ngram)]
            # print(matching_ngrams)
            if not matching_ngrams.empty:
                # Choose the next word based on probabilities
                next_word = np.random.choice(matching_ngrams['ngram'], p=matching_ngrams['prob'])
                sampled_sequence.append(next_word[-1])
            else:
                # If no matching N-grams, add '\x03' and break the loop
                sampled_sequence.append('\x03')
            counter += 1    

        # Add the final '\x03' token
        sampled_sequence.append('\x03')

        # Convert the list of tokens to a string
        sampled_string = ' '.join(sampled_sequence)

        return sampled_string
    

    # def sample(self, M):
    #     output_string = ['\x02']
    #     for j in range(M-1):
    #         b = self
    #         if self.N > 2:
    #             b = self.prev_mdl
    #         for i in range(self.N - 3 - j):
    #             b = b.prev_mdl

    #         condition_tup = tuple(output_string)
    #         try:
    #             charlie = b.mdl[b.mdl['n1gram'] == condition_tup[-b.N+1:]]
    #             random_row = np.random.choice(charlie.index, p=charlie['prob'])
    #             output_string.append(charlie['ngram'].loc[random_row][-1])
    #         except:
    #             output_string.append('\x03')

    #     output_string.append('\x03')
    #     # how do we join the string with punctuation? like would ['katy', 's', 'pie'] return "katy ' s pie" or "katy's pie"?
    #     return ' '.join(output_string)