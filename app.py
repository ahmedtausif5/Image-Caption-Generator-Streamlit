import streamlit as st
import torch
from PIL import Image
import os
import gdown

# Importing local modules
from model_architecture import CNNEncoder, DecoderLSTM
from utils import get_tokenizer, get_transforms, generate_caption

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_SIZE = 512
HIDDEN_SIZE = 512
VOCAB_SIZE = 100278

ENCODER_ID = "1C7BYc_e4-sIpAozWOH_B6NoqWETHSfWe" 
DECODER_ID = "1zjsim4j5BvFeHrP018XSrhFbAZlVnGjV"


@st.cache_resource
def download_files():
    # Only download if files are missing
    if not os.path.exists("encoder.pth"):
        url = f'https://drive.google.com/uc?id={ENCODER_ID}'
        gdown.download(url, "encoder.pth", quiet=False)
        
    if not os.path.exists("decoder.pth"):
        url = f'https://drive.google.com/uc?id={DECODER_ID}'
        gdown.download(url, "decoder.pth", quiet=False)

# --- Model Loading ---
@st.cache_resource
def load_models():
    # 1. Download weights if missing
    download_files()

    # 2. Create model instances
    encoder = CNNEncoder(EMBED_SIZE).to(DEVICE)
    decoder = DecoderLSTM(EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE).to(DEVICE)
    
    # 3. Load weights
    encoder_state = torch.load("encoder.pth", map_location=DEVICE)
    decoder_state = torch.load("decoder.pth", map_location=DEVICE)
    
    encoder.load_state_dict(encoder_state)
    decoder.load_state_dict(decoder_state)
    
    encoder.eval()
    decoder.eval()
    return encoder, decoder

# --- UI Layout ---
st.set_page_config(page_title="AI Caption Generator", page_icon="📸")

st.title("📸 AI Image Caption Generator")
st.markdown("Upload an image to see what the AI thinks is happening.")

# Initializing resources
tokenizer = get_tokenizer()
transforms = get_transforms()
encoder, decoder = load_models()

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and encoder is not None:
    try:
        # Opening and displaying the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if st.button('Generate Caption'):
            with st.spinner('Generating...'):
                # Transforming image to tensor
                img_tensor = transforms(image).unsqueeze(0).to(DEVICE)
                
                # Running inference
                caption = generate_caption(img_tensor, encoder, decoder, tokenizer)
            
            # Displaying result
            st.success("Caption Generated!")
            st.markdown(f"### {caption}")
            
    except Exception as e:
        st.error(f"Error processing image: {e}")