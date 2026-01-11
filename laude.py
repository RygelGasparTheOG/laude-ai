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
class LaudeModel:
    
    def __init__(self):
        self.version = '1.0'
        self.last_trained = datetime
        self.word_index = defaultdict(list)
        self.training_embeddings = []
        self.ngram_cache = {}  # Cache for performance
        
    def train(self, training_data):
        """Train the model on question-answer pairs"""
        self.word_index.clear()
        self.training_embeddings = []
        self.ngram_cache.clear()
        
        for idx, item in enumerate(training_data):
            normalized_input = item['input'].lower().strip()
            words = normalized_input.split()
            
            self.training_embeddings.append({
                'input': normalized_input,
                'original_input': item['input'],
                'response': item['response'],
                'words': set(words),
                'word_list': words,
                'char_trigrams': self.get_ngrams(normalized_input, n=3),
                'length': len(words)
            })
            
            for word in words:
                self.word_index[word].append(idx)
        
        self.last_trained = datetime.now().isoformat()
        print(f"✓ Model trained on {len(training_data)} examples")
    
    def get_ngrams(self, text, n=3):
        """Generate character n-grams from text"""
        if text in self.ngram_cache:
            return self.ngram_cache[text]
        
        # Pad text for edge cases
        padded = ' ' * (n-1) + text + ' ' * (n-1)
        ngrams = set()
        for i in range(len(padded) - n + 1):
            ngrams.add(padded[i:i+n])
        
        self.ngram_cache[text] = ngrams
        return ngrams
    
    def jaccard_similarity(self, set1, set2):
        """Calculate Jaccard similarity between two sets"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def dice_coefficient(self, set1, set2):
        """Calculate Dice coefficient (more lenient than Jaccard)"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        return (2.0 * intersection) / (len(set1) + len(set2))
    
    def levenshtein_distance(self, s1, s2):
        """Calculate edit distance between two strings"""
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
        """Calculate similarity between two words"""
        if word1 == word2:
            return 1.0
        
        # Quick reject if too different in length
        len_diff = abs(len(word1) - len(word2))
        if len_diff > 3:
            return 0.0
        
        # Use character trigrams for similarity
        ngrams1 = self.get_ngrams(word1, n=3)
        ngrams2 = self.get_ngrams(word2, n=3)
        
        # Combine Jaccard and edit distance
        jaccard = self.jaccard_similarity(ngrams1, ngrams2)
        
        # Normalize edit distance to 0-1 scale
        max_len = max(len(word1), len(word2))
        edit_dist = self.levenshtein_distance(word1, word2)
        normalized_edit = 1.0 - (edit_dist / max_len) if max_len > 0 else 0.0
        
        # Weighted combination
        return 0.6 * jaccard + 0.4 * normalized_edit
    
    def find_similar_words(self, word, threshold=0.7):
        """Find words in index similar to given word"""
        similar = []
        for indexed_word in self.word_index.keys():
            similarity = self.word_similarity(word, indexed_word)
            if similarity >= threshold:
                similar.append((indexed_word, similarity))
        
        # Sort by similarity, return words only
        similar.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in similar]
    
    def calculate_match_score(self, user_words, user_input, embedding):
        """Calculate comprehensive match score"""
        score = 0.0
        
        # 1. Exact word matches (highest weight)
        exact_matches = len(user_words.intersection(embedding['words']))
        score += exact_matches * 10.0
        
        # 2. Fuzzy word matches
        fuzzy_score = 0.0
        for user_word in user_words:
            if user_word not in embedding['words']:
                best_similarity = 0.0
                for emb_word in embedding['words']:
                    similarity = self.word_similarity(user_word, emb_word)
                    best_similarity = max(best_similarity, similarity)
                fuzzy_score += best_similarity
        score += fuzzy_score * 5.0  # Less weight than exact matches
        
        # 3. Word order bonus (partial sequence matching)
        user_word_list = user_input.split()
        emb_word_list = embedding['word_list']
        
        # Check for common subsequences
        for i in range(len(user_word_list) - 1):
            bigram = f"{user_word_list[i]} {user_word_list[i+1]}"
            for j in range(len(emb_word_list) - 1):
                emb_bigram = f"{emb_word_list[j]} {emb_word_list[j+1]}"
                if bigram == emb_bigram:
                    score += 3.0
        
        # 4. Character n-gram similarity (catches typos and variations)
        user_trigrams = self.get_ngrams(user_input, n=3)
        trigram_similarity = self.jaccard_similarity(user_trigrams, embedding['char_trigrams'])
        score += trigram_similarity * 15.0
        
        # 5. Substring matching bonus
        if embedding['input'] in user_input or user_input in embedding['input']:
            score += 8.0
        
        # 6. Length similarity bonus (prefer similar length inputs)
        len_diff = abs(len(user_words) - embedding['length'])
        if len_diff == 0:
            score += 3.0
        elif len_diff == 1:
            score += 1.5
        
        # 7. Jaccard similarity on word sets
        word_jaccard = self.jaccard_similarity(user_words, embedding['words'])
        score += word_jaccard * 12.0
        
        # 8. Dice coefficient (more forgiving than Jaccard)
        word_dice = self.dice_coefficient(user_words, embedding['words'])
        score += word_dice * 8.0
        
        return score
    
    def predict(self, user_input):
        """Find best matching response using advanced fuzzy matching"""
        if not self.training_embeddings:
            return "I haven't been trained yet. Please add training data first!"
        
        normalized_input = user_input.lower().strip()
        words = set(normalized_input.split())
        
        # Quick exact match check
        for embedding in self.training_embeddings:
            if embedding['input'] == normalized_input:
                return embedding['response']
        
        # Find candidates using multiple strategies
        candidate_indices = set()
        
        # Strategy 1: Exact word matches
        for word in words:
            if word in self.word_index:
                candidate_indices.update(self.word_index[word])
        
        # Strategy 2: Fuzzy word matches (typo tolerance)
        for word in words:
            if len(word) > 3:  # Only fuzzy match longer words
                similar_words = self.find_similar_words(word, threshold=0.7)
                for similar_word in similar_words[:5]:  # Limit to top 5
                    if similar_word in self.word_index:
                        candidate_indices.update(self.word_index[similar_word])
        
        # Strategy 3: If still no candidates, broaden search with lower threshold
        if len(candidate_indices) < 10:
            user_trigrams = self.get_ngrams(normalized_input, n=3)
            for idx, embedding in enumerate(self.training_embeddings):
                # Use character-level similarity as fallback
                trigram_sim = self.jaccard_similarity(user_trigrams, embedding['char_trigrams'])
                if trigram_sim > 0.3:
                    candidate_indices.add(idx)
        
        # Score all candidates
        best_match = None
        highest_score = 0.0
        
        for idx in candidate_indices:
            embedding = self.training_embeddings[idx]
            score = self.calculate_match_score(words, normalized_input, embedding)
            
            if score > highest_score:
                highest_score = score
                best_match = embedding['response']
        
        # Threshold for returning a match
        confidence_threshold = CONFIDENCE_THRESHOLD
        
        if highest_score >= confidence_threshold and best_match:
            return best_match
        
        return "I'm not sure how to answer that. Could you rephrase your question?"

def load_model():
    """Load trained model from disk or create new one"""
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
                print(f"✓ Loaded model (trained: {model.last_trained})")
                return model
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
    
    print("✓ Creating new model")
    return LaudeModel()

def save_model(model):
    """Save trained model to disk"""
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

def load_training_data():
    """Load training data from JSON file"""
    if os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✓ Loaded {len(data)} training examples")
            return data
    
    print(f"⚠ {TRAINING_FILE} not found, creating empty file")
    save_training_data([])
    return []

def save_training_data(data):
    """Save training data to JSON file"""
    with open(TRAINING_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def retrain_model():
    """Reload training data and retrain model"""
    training_data = load_training_data()
    model = load_model()
    model.train(training_data)
    save_model(model)
    return model

class LaudeHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for Laude AI server"""
    
    model = None
    
    def log_message(self, format, *args):
        """Suppress default request logging"""
        pass
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.serve_html()
        elif self.path == '/api/training':
            self.serve_training_data()
        elif self.path == '/api/model-info':
            self.serve_model_info()
        else:
            self.send_error(404)
    
    def serve_html(self):
        """Serve the main HTML page"""
        try:
            with open('laude_index.html', 'r', encoding='utf-8') as f:
                html_content = f.read()
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html_content.encode('utf-8'))
        except FileNotFoundError:
            self.send_error(500, "laude_index.html not found")
    
    def serve_training_data(self):
        """Serve training data as JSON"""
        training_data = load_training_data()
        self.send_json({'training_data': training_data})
    
    def serve_model_info(self):
        """Serve model information"""
        if LaudeHandler.model is None:
            LaudeHandler.model = load_model()
        
        training_data = load_training_data()
        info = {
            'last_trained': LaudeHandler.model.last_trained,
            'training_count': len(training_data),
            'version': LaudeHandler.model.version
        }
        self.send_json(info)
    
    def do_POST(self):
        """Handle POST requests"""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return
        
        if self.path == '/api/chat':
            self.handle_chat(data)
        elif self.path == '/api/train':
            self.handle_train(data)
        elif self.path == '/api/retrain':
            self.handle_retrain()
        else:
            self.send_error(404)
    
    def handle_chat(self, data):
        """Handle chat message"""
        if LaudeHandler.model is None:
            LaudeHandler.model = load_model()
            training_data = load_training_data()
            LaudeHandler.model.train(training_data)
        
        message = data.get('message', '').strip()
        if not message:
            self.send_json({'response': 'Please send a message!'})
            return
        
        response = LaudeHandler.model.predict(message)
        self.send_json({'response': response})
    
    def handle_train(self, data):
        """Add new training data"""
        user_input = data.get('input', '').strip()
        response = data.get('response', '').strip()
        
        if not user_input or not response:
            self.send_json({'error': 'Both input and response are required'}, 400)
            return
        
        training_data = load_training_data()
        training_data.append({
            'input': user_input,
            'response': response
        })
        save_training_data(training_data)
        
        self.send_json({
            'message': 'Training data added! Click "Retrain" to update the model.',
            'count': len(training_data)
        })
    
    def handle_retrain(self):
        """Retrain the model"""
        LaudeHandler.model = retrain_model()
        count = len(LaudeHandler.model.training_embeddings)
        self.send_json({
            'message': f'Model retrained successfully on {count} examples!'
        })
    
    def send_json(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

def main():
    """Start the Laude AI server"""
    print("=" * 60)
    print("Starting Laude AI")
    print("=" * 60)
    print(f"Model file: {MODEL_FILE}")
    print(f"Training file: {TRAINING_FILE}")
    print(f"Server: http://localhost:{PORT}")
    print("\nInstructions:")
    print("  1. Open http://localhost:{} in your browser".format(PORT))
    print("  2. Add training data via the 'Train Model' button")
    print("  3. Click 'Retrain' to update the model")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    # Initialize model on startup
    model = load_model()
    training_data = load_training_data()
    model.train(training_data)
    save_model(model)
    LaudeHandler.model = model
    
    # Start server
    with socketserver.TCPServer(("", PORT), LaudeHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n✓ Shutting down server...")

if __name__ == '__main__':
    main()
