# Real Intelligence

A brain-inspired intelligence system using spiking neurons. No transformers, no backpropagation, no GPU required.

## What is this?

Real Intelligence (RI) is a neural network that works like a biological brain:

- **Spiking neurons** - Neurons fire or don't fire (like real neurons)
- **Pattern memory** - Stores spike patterns, matches new inputs to stored patterns
- **No prediction** - Doesn't predict next tokens like transformers
- **CPU-only** - Runs on any computer, no GPU needed

## Quick Start

```bash
# Train from data file
python ri.py --train data.txt

# Chat with trained model
python ri.py --chat ri_model.npz

# Run tests
python ri.py --test ri_model.npz
```

## Data Format

Training data uses `input -> response` format:

```
hello -> Hi there!
what is 2 + 2 -> 4
thanks -> You're welcome!
bye -> Goodbye!
```

## Python API

```python
from real_intelligence import RealIntelligence

# Create brain
ri = RealIntelligence(input_size=200, hidden_size=100)

# Train on pairs
pairs = [
    ("hello", "Hi there!"),
    ("what is 2 + 2", "4"),
    ("thanks", "You're welcome!"),
]
ri.train(pairs, epochs=3)

# Get response
print(ri.respond("hello"))  # Hi there!

# Creative mode (varied responses)
print(ri.respond_creative("hello"))  # May differ each time

# Save/load
ri.save("model.npz")
ri.load("model.npz")
```

## How It Works

1. **Encoding**: Words are converted to spike trains (binary timing patterns)
2. **Processing**: Spiking neurons process input through recurrent connections
3. **Memory**: Spike patterns are stored with their responses
4. **Matching**: New inputs are compared to stored patterns via dot product
5. **Output**: Closest matching pattern determines the response

## Architecture

```
Input Layer (200 neurons)
    ↓ rate-coded spikes
Hidden Layer (100 LIF neurons)
    ↓ recurrent connections
Pattern Memory
    ↓ cosine similarity
Output Response
```

## Parameters

- `input_size`: Vocabulary size (default: 200)
- `hidden_size`: Hidden neurons (default: 100)
- `creativity`: Response randomness (0.0-1.0)

## Performance

| Model | Training | Inference | RAM | Hardware |
|-------|----------|-----------|-----|----------|
| Transformer | Hours | 100ms | GBs | GPU required |
| RI (this) | Minutes | 1ms | MBs | CPU only |

## License

MIT
