import re
from collections import defaultdict


class BPETokenizer:
    def __init__(self, vocab_size, word_list=None):
        self.vocab_size = vocab_size
        self.word_list = word_list
        self.marker = "_"
        self.vocab, self.charset = self.initialize_vocabulary()
        self.merges = []
        self.train()
    
    def initialize_vocabulary(self):
        vocabulary = defaultdict(int)
        charset = set()
        for word in word_list:
            word_with_marker = self.marker + word
            characters = list(word_with_marker)
            charset.update(characters)
            tokenized_word = " ".join(characters)
            vocabulary[tokenized_word] += 1
        
        return vocabulary, charset

    def train(self):
        while True:
            pair_counts = self.get_pair_counts()
            if not pair_counts:
                break
            most_frequent_pair = max(pair_counts, key=pair_counts.get)
            self.merges.append(most_frequent_pair)
            self.vocab = self.merge_pair(most_frequent_pair)
    
    def get_pair_counts(self):
        pair_counts = defaultdict(int)
        for tokenized_word, count in self.vocab.items():
            tokens = tokenized_word.split()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += count
        return pair_counts
    
    def merge_pair(self, pair):
        new_vocab = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        
        for tokenized_word, count in self.vocab.items():
            new_tokenized_word = pattern.sub("".join(pair), tokenized_word)
            new_vocab[new_tokenized_word] = count
        return new_vocab
    
    def tokenize_word(self, word, unk_token="<UNK>"):
        word = self.marker + word
        tokens = [char if char in self.charset else unk_token for char in word]
        original_tokens = tokens.copy()
        
        for left, right in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i:i+2] == [left, right]:
                    tokens[i:i+2] = [left + right]
                else:
                    i += 1
        return tokens
    
    def tokenize_sentence(self, sentence):
        return [self.tokenize_word(word) for word in sentence.split()]
    

if __name__ == "__main__":
    data = "the quick brown fox jumps over the lazy dog."
    vocab_size = 50

    word_list = data.split()
    tokenizer = BPETokenizer(vocab_size, word_list)
    tokens = tokenizer.tokenize_sentence("the dog was lazy but still quick")
    print(tokens)