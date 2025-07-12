import torch
import numpy as np

def generate_switching_sine_data(n_samples=1000, sequence_length=20):
    """Generate data that switches between different sine wave patterns"""
    t = np.linspace(0, 8*np.pi, n_samples)
    
    pattern1 = np.sin(t)
    pattern2 = np.sin(2*t)
    pattern3 = np.sin(t) * (0.5 + 0.5*np.sin(0.5*t))
    
    data = np.zeros_like(t)
    for i in range(len(t)):
        if i % (n_samples//3) < (n_samples//9):
            data[i] = pattern1[i]
        elif i % (n_samples//3) < 2*(n_samples//9):
            data[i] = pattern2[i]
        else:
            data[i] = pattern3[i]
    
    X = np.array([data[i:i+sequence_length] for i in range(len(data) - sequence_length)])
    y = np.array([data[i+sequence_length] for i in range(len(data) - sequence_length)])
    
    return torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1)

def generate_lorenz_data(n_samples=1000, sequence_length=20):
    """Generate data from Lorenz attractor"""
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return x_dot, y_dot, z_dot

    dt = 0.01
    x, y, z = 1, 1, 1
    data = []
    for i in range(n_samples):
        dx, dy, dz = lorenz(x, y, z)
        x = x + dx * dt
        y = y + dy * dt
        z = z + dz * dt
        data.append(x)
    
    X = np.array([data[i:i+sequence_length] for i in range(len(data) - sequence_length)])
    y = np.array([data[i+sequence_length] for i in range(len(data) - sequence_length)])
    
    return torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1)

def generate_mixed_frequency_data(n_samples=1000, sequence_length=20):
    """Generate data with multiple frequency components"""
    t = np.linspace(0, 8*np.pi, n_samples)
    
    data = (np.sin(t) + 
            0.5 * np.sin(3*t) + 
            0.25 * np.sin(7*t)) * (1 + 0.5 * np.sin(0.5*t))
    
    X = np.array([data[i:i+sequence_length] for i in range(len(data) - sequence_length)])
    y = np.array([data[i+sequence_length] for i in range(len(data) - sequence_length)])
    
    return torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1)

def generate_memory_data(n_samples=1000, sequence_length=20, pattern_length=5):
    """Generate data where future values depend on patterns from earlier in sequence"""
    data = []
    patterns = []
    
    for i in range(n_samples):
        if i % pattern_length == 0:
            pattern = np.random.choice([-1, 1], size=pattern_length)
            patterns.append(pattern)
        
        if i >= pattern_length:
            data.append(patterns[-2][i % pattern_length])
        else:
            data.append(0)
    
    X = np.array([data[i:i+sequence_length] for i in range(len(data) - sequence_length)])
    y = np.array([data[i+sequence_length] for i in range(len(data) - sequence_length)])
    
    return torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1)

def get_tiny_shakespeare_data(sequence_length=20):
    """Reads the first 10000 characters of the local Tiny Shakespeare dataset."""
    # This function assumes you have a 'data' directory with 'tinyshakespeare.txt' in it.
    with open("data/tinyshakespeare.txt", 'r', encoding='utf-8') as f:
        text = f.read()[:10000]
    
    # Create character vocabulary
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    
    # Encode the text and create sequences
    X, y = [], []
    encoded = [char_to_idx[c] for c in text]
    for i in range(len(encoded) - sequence_length):
        X.append(encoded[i:i+sequence_length])
        y.append(encoded[i+sequence_length])
    
    # CORRECTED: The metadata now includes the char_to_idx map
    metadata = {
        "vocab_size": len(chars),
        "char_to_idx": char_to_idx
    }
    
    return torch.LongTensor(X), torch.LongTensor(y), metadata
