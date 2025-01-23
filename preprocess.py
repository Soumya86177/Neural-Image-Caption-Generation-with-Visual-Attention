import os
import csv
import nltk
from collections import Counter
import string
import pickle

# For image preprocessing (if needed)
from PIL import Image
from torchvision import transforms

# Set the path to captions file
CAPTIONS_FILE = "captions.txt"

def load_captions(filename):
    """
    Load captions from the file and return a dictionary mapping image IDs to captions.
    Assumes each line in the file is formatted as: image_id.jpg,caption text
    Skips the header row.
    """
    if not os.path.exists(filename):
        print(f"Error: {filename} does not exist.")
        return {}
    
    captions_dict = {}
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader, None)  # Skip the header
        if header != ['image', 'caption']:
            print("Warning: The header does not match the expected format ['image', 'caption'].")
        
        for idx, row in enumerate(reader, start=2):  # Start at 2 considering header
            if len(row) < 2:
                print(f"Skipping line {idx} due to unexpected format: {','.join(row)}")
                continue
            image_id = row[0].strip()
            caption = ','.join(row[1:]).strip()  # Handles captions with commas
            if not image_id or not caption:
                print(f"Skipping line {idx} due to empty image ID or caption.")
                continue
            if image_id not in captions_dict:
                captions_dict[image_id] = []
            captions_dict[image_id].append(caption)
    
    return captions_dict

def preprocess_caption(caption):
    """
    Preprocess a single caption: tokenize, lowercase, remove punctuation, and add start/end tokens.
    """
    # Convert to lowercase
    caption = caption.lower()
    # Remove punctuation
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = nltk.tokenize.word_tokenize(caption, language='english')
    # Add start and end tokens
    tokens = ['<start>'] + tokens + ['<end>']
    return tokens

def build_vocab(captions, threshold=5):
    """
    Build a vocabulary with words that appear at least 'threshold' times.
    """
    # Flatten all tokens
    all_tokens = []
    for caps in captions.values():
        for cap in caps:
            tokens = preprocess_caption(cap)
            all_tokens.extend(tokens)
    
    # Count word frequencies
    word_counts = Counter(all_tokens)
    # Keep words with frequency >= threshold
    vocab = [word for word, count in word_counts.items() if count >= threshold]
    # Add special tokens
    vocab = ['<pad>', '<start>', '<end>', '<unk>'] + vocab
    # Create word to index mapping
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    # Create index to word mapping
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word
    print("Max index in word2idx:", max(word2idx.values()))
    print("Vocab size (len(word2idx)):", len(word2idx))


def save_vocab(word2idx, idx2word, filename='vocab.pkl'):
    """
    Save the vocabulary mappings to a file.
    """
    with open(filename, 'wb') as f:
        pickle.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)

def main():
    # Optionally set a custom NLTK data path
    nltk_data_path = "C:/Users/soumy/AppData/Roaming/nltk_data"
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)
    
    # Ensure NLTK resources are available
    nltk.download('punkt')       # Basic tokenizer data
    nltk.download('punkt_tab')   # Additional tokenization data required for newer NLTK
    
    # Load captions
    captions = load_captions(CAPTIONS_FILE)
    if not captions:
        print("No captions loaded. Exiting.")
        return
    print(f'Loaded captions for {len(captions)} unique images.')
    
    # Build vocabulary
    word2idx, idx2word = build_vocab(captions, threshold=5)
    print(f'Vocabulary size (including special tokens): {len(word2idx)}')
    
    # Save vocabulary
    save_vocab(word2idx, idx2word)
    print('Saved vocabulary to vocab.pkl')

if __name__ == '__main__':
    main()
