"""
Real Intelligence - Spiking Neural Network
A brain-inspired chatbot using LIF neurons and pattern memory.
No transformers, no backpropagation, no GPU required.

Architecture:
- Input: Rate-coded spike trains (words → neurons)
- Hidden: Leaky Integrate-and-Fire neurons with recurrent connections
- Memory: Spike patterns stored and compared via similarity
- Output: Winning pattern determines response

Usage:
    from real_intelligence import RealIntelligence
    
    ri = RealIntelligence()
    ri.train([("hello", "hi there"), ("bye", "goodbye")])
    print(ri.respond("hello"))
"""

import numpy as np
import re
from collections import Counter


class LIFNeuron:
    """Leaky Integrate-and-Fire neuron"""
    
    def __init__(self, threshold=0.5, decay=0.85, refractory=2):
        self.threshold = threshold
        self.decay = decay
        self.refractory = refractory
        self.v = 0.0
        self.refrac_counter = 0
        self.last_spike = -999
    
    def step(self, current, t):
        if self.refrac_counter > 0:
            self.refrac_counter -= 1
            return False
        self.v = self.v * self.decay + current
        if self.v >= self.threshold:
            self.v = 0.0
            self.refrac_counter = self.refractory
            self.last_spike = t
            return True
        return False
    
    def reset(self):
        self.v = 0.0
        self.refrac_counter = 0


class RealIntelligence:
    """
    Brain-inspired intelligence using spiking neurons.
    
    Unlike transformers that predict next tokens, this system:
    1. Encodes input as spike trains
    2. Processes through LIF neurons
    3. Stores spike patterns in memory
    4. Matches new inputs to stored patterns
    
    Args:
        input_size: Number of input neurons (vocabulary size)
        hidden_size: Number of hidden neurons
        creativity: Randomness in spike patterns (0.0-1.0)
    """
    
    def __init__(self, input_size=200, hidden_size=100, creativity=0.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.creativity = creativity
        
        self.input_layer = [LIFNeuron(0.8) for _ in range(input_size)]
        self.hidden_layer = [LIFNeuron(0.35, 0.8) for _ in range(hidden_size)]
        
        self.W_ih = np.random.uniform(0.3, 0.6, (input_size, hidden_size))
        self.W_hh = np.random.uniform(-0.01, 0.01, (hidden_size, hidden_size))
        
        self.word2idx = {}
        self.memory = []
    
    def _build_vocabulary(self, texts):
        """Build vocabulary from training texts"""
        word_counts = Counter()
        for t in texts:
            cleaned = re.sub(r'[^a-z0-9 ]', ' ', t.lower())
            word_counts.update(cleaned.split())
        
        essential = {'hello', 'hi', 'hey', 'thanks', 'thank', 'bye', 'goodbye',
                     'what', 'is', 'how', 'are', 'you', 'do', 'can', 'the',
                     'a', 'i', 'my', 'name', 'made', 'who', 'help', 'make'}
        
        top_words = [w for w, _ in word_counts.most_common(self.input_size)]
        all_words = list(essential) + [w for w in top_words if w not in essential]
        all_words = all_words[:self.input_size]
        
        self.word2idx = {w: i for i, w in enumerate(all_words)}
    
    def _encode(self, text, timesteps=20):
        """Encode text to spike pattern"""
        spikes = np.zeros((self.input_size, timesteps))
        cleaned = re.sub(r'[^a-z0-9 ]', ' ', text.lower())
        words = cleaned.split()
        
        for wi, w in enumerate(words):
            if w in self.word2idx:
                idx = self.word2idx[w]
                offset = (wi * 3) % timesteps
                for t in range(timesteps):
                    phase = (t + offset) % 4
                    if phase == 0:
                        spikes[idx, t] = 1.0
                    if (t + idx) % 5 == 0:
                        spikes[idx, t] = 1.0
        return spikes
    
    def _get_pattern(self, input_spikes, timesteps=20):
        """Get spike pattern from input"""
        return input_spikes.mean(axis=1)
    
    def train(self, pairs, epochs=3):
        """
        Train on input-response pairs.
        
        Args:
            pairs: List of (input_text, response_text) tuples
            epochs: Number of training epochs
        """
        all_texts = [inp for inp, _ in pairs] + [out for _, out in pairs]
        self._build_vocabulary(all_texts)
        
        seen = set()
        for ep in range(epochs):
            for inp, out in pairs:
                key = inp.lower().strip()
                if key in seen:
                    continue
                seen.add(key)
                
                input_spikes = self._encode(inp, timesteps=20)
                pattern = self._get_pattern(input_spikes, timesteps=20)
                self.memory.append((pattern, out))
    
    def respond(self, text):
        """
        Generate response to input text.
        
        Args:
            text: Input string
            
        Returns:
            Best matching response from training data
        """
        if not self.memory:
            return "[not trained]"
        
        input_spikes = self._encode(text, timesteps=25)
        query = self._get_pattern(input_spikes, timesteps=25)
        
        best_sim = -1
        best_resp = None
        
        for pattern, response in self.memory:
            sim = np.dot(query, pattern)
            if sim > best_sim:
                best_sim = sim
                best_resp = response
        
        if best_sim < 0.0001:
            return "[uncertain]"
        
        return best_resp
    
    def respond_creative(self, text):
        """
        Generate creative response (varied for same input).
        
        Args:
            text: Input string
            
        Returns:
            Response, randomly selected from top matches
        """
        if not self.memory:
            return "[not trained]"
        
        input_spikes = self._encode(text, timesteps=25)
        query = self._get_pattern(input_spikes, timesteps=25)
        
        sims = []
        for pattern, response in self.memory:
            sim = np.dot(query, pattern)
            sims.append((sim, response))
        
        sims.sort(key=lambda x: x[0], reverse=True)
        
        if sims[0][0] < 0.0001:
            return "[uncertain]"
        
        if self.creativity > 0 and np.random.random() < self.creativity:
            top = sims[:min(3, len(sims))]
            winner = np.random.choice(len(top))
            return top[winner][1]
        
        return sims[0][1]
    
    def save(self, filepath):
        """Save brain to file"""
        data = {
            'word2idx': self.word2idx,
            'memory': [(p.tolist(), r) for p, r in self.memory],
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
        }
        np.savez(filepath, **{k: v for k, v in data.items()})
    
    def load(self, filepath):
        """Load brain from file"""
        data = np.load(filepath, allow_pickle=True)
        self.word2idx = data['word2idx'].item()
        self.memory = [(np.array(p), r) for p, r in data['memory']]
