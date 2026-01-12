import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    """
    Encoder architecture based on ResNet50.
    """
    def __init__(self, embed_size, train_cnn=False):
        super().__init__()
        
        # Loading the pretrained ResNet50 model with default weights
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freezing or unfreezing layers based on the train_cnn flag
        for param in resnet.parameters():
            param.requires_grad = train_cnn
        
        # Removing the last fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Linear bridge layer to map features to embedding dimension
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # Extracting features from the image
        features = self.resnet(images)
        features = features.view(features.shape[0], -1)
        features = self.embed(features)
        return self.dropout(self.relu(features))


class DecoderLSTM(nn.Module):
    """
    Decoder architecture using LSTM with Image Injection.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        # Embedding layer for text
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer (Input size is doubled to accommodate concatenated Image + Word)
        self.lstm = nn.LSTM(embed_size * 2, hidden_size, num_layers, batch_first=True)
        
        # Output linear layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # Forward pass logic (required for loading state dict correctly)
        captions = captions[:, :-1]
        embeddings = self.dropout(self.embed(captions))
        
        # Expanding image features to match sequence length
        seq_len = embeddings.shape[1]
        features = features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenating Image + Word vectors
        combined_input = torch.cat((embeddings, features), dim=2)
        
        hiddens, _ = self.lstm(combined_input)
        outputs = self.linear(hiddens)
        return outputs