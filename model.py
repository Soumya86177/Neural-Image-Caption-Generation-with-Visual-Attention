# model.py


# STEP 1

# EncoderCNN:Uses a pre-trained ResNet-50 model from torchvision.models.
# Removes the last two layers (average pooling and fully connected) to obtain convolutional feature maps.
# Freezes the ResNet parameters to prevent updating during training.
# Adds an adaptive average pooling layer to ensure consistent spatial dimensions (14x14).
# Transforms the ResNet output to the desired embed_size using a linear layer followed by batch normalization.
# The output shape is (batch_size, num_pixels, embed_size), where num_pixels = 14 * 14 = 196.

import torch
import torch.nn as nn
import torchvision.models as models
from preprocess import load_captions
import os
from preprocess import preprocess_caption




import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """
        Initialize the Encoder with a pre-trained ResNet model and a linear layer to transform features.

        :param embed_size: Size of the word embeddings (256)
        """
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # Updated weights parameter
        modules = list(resnet.children())[:-2]  # Remove avgpool and fc layers
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))  # Ensure consistent spatial dimensions
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)  # (2048 -> 256)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)  # BatchNorm1d for embed_size=256

    def forward(self, images):
        """
        Forward pass to extract image features.

        :param images: Input images, shape (batch_size, 3, 224, 224)
        :return: Encoded image features, shape (batch_size, num_pixels=196, embed_size=256)
        """
        with torch.no_grad():
            features = self.resnet(images)  # (batch_size, 2048, H, W)
        features = self.adaptive_pool(features)  # (batch_size, 2048, 14, 14)
        features = features.view(features.size(0), -1, features.size(1))  # (batch_size, 196, 2048)
        features = self.linear(features)  # (batch_size, 196, 256)
        features = features.permute(0, 2, 1)  # (batch_size, 256, 196)
        features = self.bn(features)        # (batch_size, 256, 196)
        features = features.permute(0, 2, 1)  # (batch_size, 196, 256)
        return features  # Output to be used by the decoder

#STEP 2

# Building the Attention Mechanism
# The attention mechanism computes attention weights over the encoder's feature maps based on the decoder's hidden state.
# model.py

# model.py

import torch
import torch.nn as nn
import torchvision.models as models

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        Initialize the Attention module.

        :param encoder_dim: Feature size of encoded images (256)
        :param decoder_dim: Size of decoder's RNN (512)
        :param attention_dim: Size of the attention linear layers (256)
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # 256 -> 256
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # 512 -> 256
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax over the spatial locations

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward pass for the attention mechanism.

        :param encoder_out: Encoded images, shape (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: Previous decoder hidden state, shape (batch_size, decoder_dim)
        :return: Context vector and attention weights
        """
        # Apply linear layers
        att1 = self.encoder_att(encoder_out)       # (batch_size, num_pixels, attention_dim=256)
        att2 = self.decoder_att(decoder_hidden)    # (batch_size, attention_dim=256)

        # Combine and apply non-linearity
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)

        # Normalize attention scores
        alpha = self.softmax(att)  # (batch_size, num_pixels)

        # Compute context vector as the weighted sum of encoder outputs
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim=256)

        return context, alpha


    
# Explanation:

# Attention:
# Combines encoder features and decoder hidden state to compute attention scores.
# Applies linear transformations to both encoder outputs and decoder hidden state.
# Uses ReLU activation followed by a linear layer to compute attention scores.
# Applies softmax to obtain attention weights (alpha).
# Computes the context vector as the weighted sum of encoder features based on alpha.
    

# STEP 3
#Building the Decoder (RNN with Attention)
#The decoder generates the caption word by word, using the context vector from the attention mechanism
    

# model.py

# model.py

# model.py

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim):
        """
        Initialize the Decoder with Attention.

        :param embed_size: Size of the word embeddings (256)
        :param hidden_size: Number of hidden units in the LSTM (512)
        :param vocab_size: Size of the vocabulary
        :param attention_dim: Dimension of the attention linear layers (256)
        """
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim=256, decoder_dim=hidden_size, attention_dim=attention_dim)
        self.lstm = nn.LSTMCell(embed_size + 256, hidden_size, bias=True)  # Input: embed_size + encoder_dim
        self.dropout = nn.Dropout(0.5)
        self.init_h = nn.Linear(256, hidden_size)  # Initialize hidden state from encoder
        self.init_c = nn.Linear(256, hidden_size)  # Initialize cell state from encoder
        self.f_beta = nn.Linear(hidden_size, 256)  # Gating scalar, transform hidden state
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_out, captions, lengths):
        """
        Forward pass for the Decoder.

        :param encoder_out: Encoded images, shape (batch_size, num_pixels=196, encoder_dim=256)
        :param captions: Ground truth captions, shape (batch_size, max_caption_length)
        :param lengths: Caption lengths, shape (batch_size)
        :return: Predictions, targets, decode_lengths, alphas, sort_ind
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)  # 256
        vocab_size = self.fc.out_features

        # Sort input data by decreasing lengths
        lengths, sort_ind = lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        captions = captions[sort_ind]

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, 196, 256)
        num_pixels = encoder_out.size(1)

        # Embedding
        embeddings = self.embed(captions)  # (batch_size, max_caption_length, embed_size=256)

        # Initialize LSTM state
        h = self.init_h(encoder_out.mean(dim=1))  # (batch_size, hidden_size=512)
        c = self.init_c(encoder_out.mean(dim=1))  # (batch_size, hidden_size=512)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        decode_lengths = [int(l.item()) - 1 for l in lengths]


        # Create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        # Iterate through each time step
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                 h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # (batch_size_t, 256)
            attention_weighted_encoding = gate * attention_weighted_encoding

            # Concatenate embedding and attention weighted encoding
            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1)  # (batch_size_t, 512)

            # Decode the concatenated vector
            h, c = self.lstm(lstm_input, (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, hidden_size=512)

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, captions[:, 1:], decode_lengths, alphas, sort_ind



# Explanation:

#DecoderRNN:
#Initializes the embedding layer, attention module, LSTMCell, and a fully connected layer to project hidden states to vocabulary scores.
#forward:
#Sorts the batch by caption lengths for efficient processing.
#Embeds the captions.
#Initializes LSTM hidden and cell states.
#Iterates through each time step up to the maximum caption length.
#At each step:
#Computes the context vector using the attention mechanism.
#Concatenates the current word embedding and context vector as input to the LSTM.
#Updates LSTM hidden and cell states.
#Generates predictions for the next word.
#Stores attention weights for visualization or analysis.
#init_hidden_state:
#Initializes the hidden and cell states to zeros.
#Note:

#This implementation uses LSTMCell for flexibility and manual control over each time step.
#The forward method is designed for training with teacher forcing, where the ground truth caption is provided.



#STEP 4
#Training the Model
#With the model components in place, we'll proceed to train the model. This involves creating a dataset loader, defining the loss function and optimizer, and implementing the training loop.
import torch.utils.data as data
from PIL import Image
import json

class FlickrDataset(data.Dataset):
    def __init__(self, root, captions_file, word2idx, transform=None):
        self.root = root
        self.captions_dict = load_captions(captions_file)
        self.image_ids = list(self.captions_dict.keys())
        self.word2idx = word2idx
        self.transform = transform
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.root, image_id)
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        # Get random caption
        captions = self.captions_dict[image_id]
        caption = captions[torch.randint(len(captions), (1,)).item()]
        
        tokens = preprocess_caption(caption)
        
        # Convert tokens to indices
        caption_idxs = []
        for token in tokens:
            idx = self.word2idx.get(token, self.word2idx['<unk>'])
            # If idx is out of range, skip this sample entirely
            if idx >= len(self.word2idx):
                print(f"Skipping caption due to out-of-range index {idx} for token '{token}': {caption}")
                return None  # We'll handle this in the collate_fn
            caption_idxs.append(idx)
        
        return image, torch.tensor(caption_idxs)


#Explanation:

#FlickrDataset:
#Initializes with the image directory, captions file, vocabulary mappings, and image transformations.
#getitem:
#Retrieves an image and a randomly selected caption.
#Applies transformations to the image.
#Tokenizes and converts the caption to numerical indices.
#Returns the image tensor and caption tensor.
    
#STEP 5
#Create Collate Function:
#Since captions have varying lengths, we'll pad them to the maximum length in a batch.
def collate_fn(data):
    # Filter out any 'None' items
    data = [d for d in data if d is not None]

    if len(data) == 0:
        # Return empty batch in the correct format
        return None, None, None

    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    max_length = max(lengths)
    padded_captions = torch.zeros(len(captions), max_length).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = cap[:end]
    
    return images, padded_captions, torch.tensor(lengths)



#Explanation:

#collate_fn:
#Sorts the data by caption length.
#Stacks images into a single tensor.
#Pads captions with zeros (<pad>) to match the maximum length in the batch.
#Returns images, padded captions, and caption lengths.