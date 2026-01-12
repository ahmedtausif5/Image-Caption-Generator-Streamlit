import torch
import tiktoken
from torchvision import transforms

def get_transforms():
    """
    Defining the standard ResNet image transformations.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])

def get_tokenizer():
    """
    Initializing the GPT-2 tokenizer (tiktoken).
    """
    return tiktoken.get_encoding("cl100k_base")

def generate_caption(image, encoder, decoder, tokenizer, max_len=50):
    """
    Running the inference loop to generate a caption.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setting models to evaluation mode
    encoder.eval()
    decoder.eval()
    
    # 1. Encoding the image
    with torch.no_grad():
        features = encoder(image)

    # 2. preparing start and end tokens
    start_token_id = tokenizer.encode("<|startoftext|>", allowed_special="all")[0]
    end_token_id = tokenizer.encode("<|endoftext|>", allowed_special="all")[0]
    
    # Initializing input with start token
    current_input = torch.tensor([[start_token_id]], device=device)
    states = None
    result_caption = []

    # 3. Generation loop
    with torch.no_grad():
        for _ in range(max_len):
            # Getting embeddings for the current word
            embeddings = decoder.embed(current_input)
            
            # Reshaping image features for concatenation
            features_reshaped = features.unsqueeze(1)
            
            # Concatenating Image + Word
            lstm_input = torch.cat((embeddings, features_reshaped), dim=2)
            
            # Passing through LSTM
            outputs, states = decoder.lstm(lstm_input, states)
            
            # Predicting next word
            prediction = decoder.linear(outputs.squeeze(1))
            predicted_id = prediction.argmax(1).item()
            
            # Checking for end token
            if predicted_id == end_token_id:
                break
            
            result_caption.append(predicted_id)
            current_input = torch.tensor([[predicted_id]], device=device)
    
    # 4. Decoding tokens to text and cleaning string
    caption = tokenizer.decode(result_caption)
    return caption.replace("<|startoftext|>", "").replace("|startoftext|>", "").strip()