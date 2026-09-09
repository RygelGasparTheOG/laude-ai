#!/usr/bin/env python3
"""
Laude AI
Run: python3 laude.py
Access: http://localhost:8000
"""

import http.server
import socketserver
import json
import pickle
import os
from datetime import datetime
from collections import defaultdict

PORT = 8000
MODEL_FILE = 'laude_model.pkl'
TRAINING_FILE = 'laude_dataset.json'
CONFIDENCE_THRESHOLD = 7.5

# Words so common they appear in most training entries and carry almost no
# discriminative value ("is" alone matched 96% of a 26.8k-entry dataset).
# Excluded from candidate generation only -- they're still used for scoring
# (word_jaccard, dice, etc.) once a real candidate set exists.
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'what', 'when', 'where', 'who', 'why', 'how', 'do', 'does', 'did',
    'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'but', 'with',
    'i', 'you', 'it', 'this', 'that', 'my', 'your',
}


class LaudeModel:
    def __init__(self):
        self.version = '1.1'
        self.last_trained = datetime
        self.word_index = defaultdict(list)
        self.training_embeddings = []
        self.ngram_cache = {}  # Cache for performance
        # NEW: trigram -> set of indexed words that contain it.
        # Lets fuzzy lookup skip words that share nothing with the query,
        # instead of comparing against every word ever seen.
        self.word_trigram_index = defaultdict(set)
        # NEW: trigram -> set of training example indices, used to bound
        # the worst-case fallback scan instead of touching every example.
        self.trigram_to_examples = defaultdict(set)

    def train(self, training_data):
        """Train the model on question-answer pairs"""
        self.word_index.clear()
        self.training_embeddings = []
        self.ngram_cache.clear()
        self.word_trigram_index.clear()
        self.trigram_to_examples.clear()

        for idx, item in enumerate(training_data):
            normalized_input = item['input'].lower().strip()
            words = normalized_input.split()
            char_trigrams = self.get_ngrams(normalized_input, n=3)

            self.training_embeddings.append({
                'input': normalized_input,
                'original_input': item['input'],
                'response': item['response'],
                'words': set(words),
                'word_list': words,
                'char_trigrams': char_trigrams,
                'length': len(words)
            })

            for word in words:
                self.word_index[word].append(idx)
                for tri in self.get_ngrams(word, n=3):
                    self.word_trigram_index[tri].add(word)

            for tri in char_trigrams:
                self.trigram_to_examples[tri].add(idx)

        self.last_trained = datetime.now().isoformat()
        print(f"OK Model trained on {len(training_data)} examples")

    def get_ngrams(self, text, n=3):
        """Generate character n-grams from text"""
        if text in self.ngram_cache:
            return self.ngram_cache[text]
        padded = ' ' * (n - 1) + text + ' ' * (n - 1)
        ngrams = set()
        for i in range(len(padded) - n + 1):
            ngrams.add(padded[i:i + n])
        self.ngram_cache[text] = ngrams
        return ngrams

    def jaccard_similarity(self, set1, set2):
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def dice_coefficient(self, set1, set2):
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        return (2.0 * intersection) / (len(set1) + len(set2))

    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def word_similarity(self, word1, word2):
        if word1 == word2:
            return 1.0
        len_diff = abs(len(word1) - len(word2))
        if len_diff > 3:
            return 0.0
        ngrams1 = self.get_ngrams(word1, n=3)
        ngrams2 = self.get_ngrams(word2, n=3)
        jaccard = self.jaccard_similarity(ngrams1, ngrams2)
        max_len = max(len(word1), len(word2))
        edit_dist = self.levenshtein_distance(word1, word2)
        normalized_edit = 1.0 - (edit_dist / max_len) if max_len > 0 else 0.0
        return 0.6 * jaccard + 0.4 * normalized_edit

    def find_similar_words(self, word, threshold=0.7):
        # Only consider words that share at least one character trigram
        # with the query word, instead of scanning every indexed word.
        # A pair of similar words (esp. by n-gram/edit-distance) almost
        # always shares at least one trigram, so this preserves results
        # while cutting comparisons from O(vocab) to O(shared-trigram words).
        candidates = set()
        for tri in self.get_ngrams(word, n=3):
            candidates.update(self.word_trigram_index.get(tri, ()))

        similar = []
        for indexed_word in candidates:
            similarity = self.word_similarity(word, indexed_word)
            if similarity >= threshold:
                similar.append((indexed_word, similarity))
        similar.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in similar]

    def calculate_match_score(self, user_words, user_input, embedding):
        score = 0.0
        exact_matches = len(user_words.intersection(embedding['words']))
        score += exact_matches * 10.0

        fuzzy_score = 0.0
        for user_word in user_words:
            if user_word not in embedding['words']:
                best_similarity = 0.0
                for emb_word in embedding['words']:
                    similarity = self.word_similarity(user_word, emb_word)
                    best_similarity = max(best_similarity, similarity)
                fuzzy_score += best_similarity
        score += fuzzy_score * 5.0

        user_word_list = user_input.split()
        emb_word_list = embedding['word_list']
        for i in range(len(user_word_list) - 1):
            bigram = f"{user_word_list[i]} {user_word_list[i+1]}"
            for j in range(len(emb_word_list) - 1):
                emb_bigram = f"{emb_word_list[j]} {emb_word_list[j+1]}"
                if bigram == emb_bigram:
                    score += 3.0

        user_trigrams = self.get_ngrams(user_input, n=3)
        trigram_similarity = self.jaccard_similarity(user_trigrams, embedding['char_trigrams'])
        score += trigram_similarity * 15.0

        if embedding['input'] in user_input or user_input in embedding['input']:
            score += 8.0

        len_diff = abs(len(user_words) - embedding['length'])
        if len_diff == 0:
            score += 3.0
        elif len_diff == 1:
            score += 1.5

        word_jaccard = self.jaccard_similarity(user_words, embedding['words'])
        score += word_jaccard * 12.0

        word_dice = self.dice_coefficient(user_words, embedding['words'])
        score += word_dice * 8.0

        return score

    def predict(self, user_input):
        if not self.training_embeddings:
            return "I haven't been trained yet. Please add training data first!"

        normalized_input = user_input.lower().strip()
        words = set(normalized_input.split())

        for embedding in self.training_embeddings:
            if embedding['input'] == normalized_input:
                return embedding['response']

        candidate_indices = set()
        # Stopwords are excluded from candidate generation only: a word
        # like "is" appearing in almost every entry tells us nothing about
        # which entry the user means, and pulling in its full posting list
        # is what made near-every query touch near-every training example.
        # Scoring below still uses the full `words` set, stopwords included.
        content_words = words - STOPWORDS

        for word in content_words:
            if word in self.word_index:
                candidate_indices.update(self.word_index[word])

        for word in content_words:
            if len(word) > 3:
                similar_words = self.find_similar_words(word, threshold=0.7)
                for similar_word in similar_words[:5]:
                    if similar_word in self.word_index:
                        candidate_indices.update(self.word_index[similar_word])

        if len(candidate_indices) < 10:
            # Use the trigram->example index to only score examples that
            # share at least one character trigram with the query, instead
            # of computing trigram similarity against every example in the
            # dataset. This is what made no-match queries scale with total
            # dataset size before.
            user_trigrams = self.get_ngrams(normalized_input, n=3)
            fallback_pool = set()
            for tri in user_trigrams:
                fallback_pool.update(self.trigram_to_examples.get(tri, ()))
            for idx in fallback_pool:
                embedding = self.training_embeddings[idx]
                trigram_sim = self.jaccard_similarity(user_trigrams, embedding['char_trigrams'])
                if trigram_sim > 0.3:
                    candidate_indices.add(idx)

        best_match = None
        highest_score = 0.0
        for idx in candidate_indices:
            embedding = self.training_embeddings[idx]
            score = self.calculate_match_score(words, normalized_input, embedding)
            if score > highest_score:
                highest_score = score
                best_match = embedding['response']

        confidence_threshold = CONFIDENCE_THRESHOLD
        if highest_score >= confidence_threshold and best_match:
            return best_match

        return "I'm not sure how to answer that. Could you rephrase your question?"


def load_training_data():
    if os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"OK Loaded {len(data)} training examples")
        return data
    print(f"WARN {TRAINING_FILE} not found, creating empty file")
    save_training_data([])
    return []


def save_training_data(data):
    with open(TRAINING_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
