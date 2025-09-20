# Image Caption Generator

Automatically generates descriptive captions for images using a deep learning encoder-decoder architecture.

---

## Features

- **Encoder-Decoder Architecture:** Combines CNN and LSTM for image captioning.  
- **EncoderCNN:** Uses **pretrained ResNet-50 (ImageNet)** to extract rich feature vectors from images.  
- **DecoderRNN:** LSTM-based decoder generates captions word by word based on the image features.  
- **Workflow:**  
Image → EncoderCNN (ResNet-50) → Feature Vector → DecoderRNN (LSTM) → Caption

- **Tech Stack:** Python, PyTorch, Torchvision, LSTM, CNN, Deep Learning.  
- **Highlights:**  
  - Leverages pretrained CNN for efficient feature extraction.  
  - Generates coherent and contextually relevant captions.  
  - Supports flexible vocabulary and variable-length captions.  

---

## Usage

```python
from model import EncoderCNN, DecoderRNN

# Initialize models
encoder = EncoderCNN(embed_size=256)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=vocab_size)

# Generate caption for a preprocessed image tensor
features = encoder(image_tensor)
caption_indices = decoder.sample(features)
caption = decode_caption(caption_indices)
print("Generated Caption:", caption)
