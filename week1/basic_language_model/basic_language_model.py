import re
import math
import random
import gradio as gr
import numpy as np
from collections import defaultdict


class BasicLanguageModel:
    def __init__(self, n_params=5):
        random.seed(42)
        self.n_params = n_params
        self.state = [{} for _ in range(n_params)]
        self.train_data, self.test_data = self.get_data()
        self.num_train_tokens = len(self.train_data)

    def get_data(self):
        with open('data/weather_data.txt', 'r', encoding='utf-8') as file:
            corpus = file.read()
        tokens = self.tokenize(corpus)
        split_index = int(len(tokens) * 0.90)
        train_corpus = tokens[:split_index]
        test_corpus = tokens[split_index:]
        return train_corpus, test_corpus

    def train(self):
        tokens = self.train_data
        for ind in range(1, self.n_params + 1):
            counts = self.state[ind - 1]
            for i in range(len(tokens) - ind + 1):
                context = tuple(tokens[i:i + ind - 1])
                next_token = tokens[i + ind - 1]
                if context not in counts:
                    counts[context] = defaultdict(int)
                counts[context][next_token] = counts[context][next_token] + 1
    
    def predict_next_token(self, context):
        for n in range(self.n_params, 1, -1):
            if len(context) >= n - 1:
                context_n = tuple(context[-(n - 1):])
                counts = self.state[n - 1].get(context_n)
                if counts:
                    return max(counts.items(), key=lambda x: x[1])[0]
        unigram_counts = self.state[0].get(())
        if unigram_counts:
            return max(unigram_counts.items(), key=lambda x: x[1])[0]
        return None

    def generate_text(self, context, num_tokens=10):
        if type(context) is str:
            context = self.tokenize(context)
        generated = list(context)

        while len(generated) - len(context) < num_tokens:
            next_token = self.predict_next_token(generated[-(self.n_params - 1):])
            generated.append(next_token)

        return " ".join(generated)

    def get_probability(self, token, context):
        for n_gram in range(self.n_params, 1, -1):
            if len(context) >= n_gram - 1:
                context_ngram = tuple(context[-(n_gram - 1):])
                counts = self.state[n_gram - 1].get(context_ngram)
                if counts:
                    total = sum(counts.values())
                    count = counts.get(token, 0)
                    if count > 0:
                        return count / total
        unigram_counts = self.state[0].get(())
        count = unigram_counts.get(token, 0)
        V = len(unigram_counts)
        return (count + 1) / (self.num_train_tokens + V)

    def compute_perplexity(self):
        tokens = self.test_data
        num_tokens = len(tokens)
        log_likelihood = 0

        for i in range(num_tokens):
            context_start = max(0, i - self.n_params)
            context = tuple(tokens[context_start:i])
            token = tokens[i]
            probability = self.get_probability(token, context)
            log_likelihood += math.log(probability)

        average_log_likelihood = log_likelihood / num_tokens

        perplexity = math.exp(-average_log_likelihood)
        return perplexity

    def predict_next_token_with_temperature(self, context, temperature=0):
        for n in range(self.n_params, 1, -1):
            if len(context) >= n - 1:
                context_n = tuple(context[-(n - 1):])
                counts = self.state[n - 1].get(context_n)
                if counts:
                    if temperature == 0:
                        return max(counts.items(), key=lambda x: x[1])[0]
                    else:
                        tokens, counts_list = zip(*counts.items())
                        counts_array = np.array(counts_list, dtype=np.float64)
                        
                        logits = np.log(counts_array)
                        logits_t = logits / temperature
                        
                        exp_logits = np.exp(logits_t - np.max(logits_t)) 
                        probs = exp_logits / np.sum(exp_logits)
                        
                        return np.random.choice(tokens, p=probs)

        unigram_counts = self.state[0].get(())
        if unigram_counts:
            return max(unigram_counts.items(), key=lambda x: x[1])[0]
        return None

    def generate_text_with_temperature(self, context, num_tokens, temperature=0):
        generated = list(context)

        while len(generated) - len(context) < num_tokens:
            next_token = self.predict_next_token_temperature(generated[-(self.n_params - 1):], temperature)
            generated.append(next_token)

            if len(generated) - len(context) >= num_tokens and next_token == ".":
                break

        return " ".join(generated)

    def tokenize(self, text):
        return re.findall(r"\b[a-zA-Z0-9]+\b|[.]", text.lower())

if __name__ == "__main__":
    model = BasicLanguageModel()
    model.train()
    model.generate_text("weather report")
    def chat(message, history):
        return model.generate_text(message, 10)
    perplexity = model.compute_perplexity()
    gr.ChatInterface(fn=chat, type="messages").launch()