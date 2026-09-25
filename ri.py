#!/usr/bin/env python3
"""
Real Intelligence - CLI Interface
A brain-inspired chatbot using spiking neurons.

Usage:
    python ri.py                    # Interactive chat
    python ri.py --train data.txt   # Train from file
    python ri.py --test             # Run test suite
"""

import argparse
import sys
from real_intelligence import RealIntelligence


def load_data(filepath):
    """Load training data from file (format: input -> response)"""
    pairs = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if ' -> ' in line:
                parts = line.strip().split(' -> ', 1)
                if len(parts) == 2:
                    pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def train_mode(data_file, epochs=3, output='ri_model.npz'):
    """Train and save model"""
    print(f"Loading data from {data_file}...")
    pairs = load_data(data_file)
    print(f"Loaded {len(pairs)} pairs")
    
    ri = RealIntelligence(input_size=200, hidden_size=100, creativity=0.3)
    ri.train(pairs, epochs=epochs)
    
    ri.save(output)
    print(f"Model saved to {output}")
    print(f"Vocabulary: {len(ri.word2idx)} words")
    print(f"Memory: {len(ri.memory)} patterns")


def chat_mode(model_file):
    """Interactive chat"""
    ri = RealIntelligence(input_size=200, hidden_size=100, creativity=0.3)
    ri.load(model_file)
    
    creative = False
    
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user:
            continue
        if user.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        if user.lower() == 'creative':
            creative = not creative
            print(f"Creativity: {'ON' if creative else 'OFF'}")
            continue
        
        if creative:
            response = ri.respond_creative(user)
        else:
            response = ri.respond(user)
        
        print(f"RI: {response}")


def test_mode(model_file):
    """Run test suite"""
    ri = RealIntelligence(input_size=200, hidden_size=100)
    ri.load(model_file)
    
    tests = [
        "hello",
        "what is 2 + 2",
        "thanks",
        "who made you",
        "bye",
        "make a website",
    ]
    
    print("Test Results:")
    print("-" * 50)
    for t in tests:
        response = ri.respond(t)
        print(f"  {t:30s} -> {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Intelligence - Spiking Neural Network")
    parser.add_argument("--train", help="Train from data file")
    parser.add_argument("--chat", help="Chat with saved model")
    parser.add_argument("--test", help="Test saved model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--output", default="ri_model.npz", help="Model output file")
    
    args = parser.parse_args()
    
    if args.train:
        train_mode(args.train, args.epochs, args.output)
    elif args.chat:
        chat_mode(args.chat)
    elif args.test:
        test_mode(args.test)
    else:
        print("Real Intelligence - Spiking Neural Network")
        print()
        print("Usage:")
        print("  python ri.py --train data.txt        Train model")
        print("  python ri.py --chat ri_model.npz     Chat with model")
        print("  python ri.py --test ri_model.npz     Test model")
        print()
        print("Data format (input -> response):")
        print("  hello -> Hi there!")
        print("  what is 2 + 2 -> 4")
        print("  bye -> Goodbye!")
