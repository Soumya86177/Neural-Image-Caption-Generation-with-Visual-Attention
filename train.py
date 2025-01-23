# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import EncoderCNN, DecoderRNN, FlickrDataset, collate_fn
import pickle
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

def main():
    print("Training script started.")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Hyperparameters
    embed_size = 256
    hidden_size = 512
    attention_dim = 256
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-3
    vocab_threshold = 5

    # Load vocabulary
    vocab_path = 'vocab.pkl'
    if not os.path.exists(vocab_path):
        print(f"Error: {vocab_path} does not exist. Please run preprocess.py first.")
        return

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    word2idx = vocab['word2idx']
    idx2word = vocab['idx2word']
    vocab_size = len(word2idx)
    print(f'Vocabulary Size: {vocab_size}')

    # ---------------------------------------------------------------------
    # DEBUG SNIPPET: Print the index of each special token
    special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
    for token in special_tokens:
        if token in word2idx:
            print(f"{token} index = {word2idx[token]}")
            # If you want to check if it's in range:
            if word2idx[token] >= vocab_size:
                print(f"WARNING: {token} index {word2idx[token]} is out of range [0..{vocab_size-1}]!")
        else:
            print(f"WARNING: {token} not found in word2idx!")
    # ---------------------------------------------------------------------

    unk_idx = word2idx['<unk>']
    print("UNK index:", unk_idx, "of", vocab_size)

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),                # Resize images to 224x224
        transforms.ToTensor(),                        # Convert images to tensors
        transforms.Normalize((0.485, 0.456, 0.406),   # Normalize with ImageNet means
                             (0.229, 0.224, 0.225))   # and standard deviations
    ])

    # Initialize dataset and dataloader
    captions_file = 'captions.txt'
    if not os.path.exists(captions_file):
        print(f"Error: {captions_file} does not exist.")
        return

    dataset = FlickrDataset(
        root='images',
        captions_file=captions_file,
        word2idx=word2idx,
        transform=transform
    )
    print(f'Dataset size: {len(dataset)}')

    # Note: Our 'collate_fn' may return None if the dataset finds a problematic caption.
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,          # Set num_workers=0 for Windows
        collate_fn=collate_fn   # Custom collate function
    )
    print('DataLoader initialized.')

    # Initialize encoder and decoder
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, attention_dim).to(device)
    print('Encoder and Decoder initialized.')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    print('Loss function and optimizer set.')

    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/image_captioning')
    print('TensorBoard writer initialized.')

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        for i, batch_data in enumerate(dataloader):
            # =========================================
            # KEY CHANGE: skip if batch_data is None
            # =========================================
            if batch_data is None:
                print(f"Skipping batch {i+1}/{len(dataloader)} because it's None.")
                continue

            # batch_data should be (images, captions, lengths) if it's not None
            images, captions, lengths = batch_data

            print(f'Processing batch {i+1}/{len(dataloader)}')

            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)

            # Forward pass
            features = encoder(images)
            outputs, targets, decode_lengths, alphas, sort_ind = decoder(features, captions, lengths)

            # Reshape outputs and targets for loss computation
            outputs = outputs.view(-1, vocab_size)
            targets = targets[:, :max(decode_lengths)].reshape(-1)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward and optimize
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Log the loss
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}')
                writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)

        # Save the model checkpoint
        torch.save(decoder.state_dict(), f'models/decoder-{epoch+1}.ckpt')
        torch.save(encoder.state_dict(), f'models/encoder-{epoch+1}.ckpt')
        print(f'Models saved to models/decoder-{epoch+1}.ckpt and models/encoder-{epoch+1}.ckpt')

    writer.close()
    print("Training script finished.")

if __name__ == '__main__':
    main()



# Explanation:

# Hyperparameters:

# embed_size: Size of the word embeddings.
# hidden_size: Number of hidden units in the decoder's LSTM.
# attention_dim: Dimension of the attention linear layers.
# num_epochs: Number of training epochs.
# batch_size: Number of samples per batch.
# learning_rate: Learning rate for the optimizer.
# Device Configuration:

# Uses GPU if available; otherwise, defaults to CPU.
# Data Loading:

# Applies image transformations (resize, tensor conversion, normalization).
# Initializes FlickrDataset and DataLoader with the custom collate_fn.
# Model Initialization:

# Initializes EncoderCNN and DecoderRNN and moves them to the configured device.
# Loss and Optimizer:

# Uses CrossEntropyLoss, ignoring the <pad> token.
# Optimizer updates only the decoder parameters and the linear and batch norm layers of the encoder.
# Training Loop:

# Iterates over epochs and batches.
# Performs forward pass to compute outputs.
# Computes loss and backpropagates gradients.
# Updates model parameters.
# Prints loss every 100 steps.
# Saves model checkpoints after each epoch.
# Note:

# Ensure that the FlickrDataset and collate_fn are correctly imported from model.py.

# Training progress visualization

# Training deep learning models can be time-consuming. To monitor progress effectively:

# Use TensorBoard:

# TensorBoard provides visualization tools for monitoring training metrics.
