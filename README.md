# Cellula Toxic Content Analyzer with Image Context

This repository contains a Streamlit web application that combines:
1. Text toxicity analysis using a pre-trained BiLSTM model
2. Image understanding using BLIP image captioning
3. Combined context analysis for better toxicity detection

The app can analyze text alone, or combine text with image context by generating image captions and analyzing the combined input.

## Project overview

- Purpose: classify text for toxic content using a BiLSTM classifier saved as an HDF5 model.
- Primary entrypoint: `main.py` — load the model and run inference or training (depending on how `main.py` is implemented).

## Files in this repository

- `bilstm_toxic_classifier_seed42.h5` — pretrained Keras model (BiLSTM) saved in HDF5 format
- `cellula_toxic_data.csv` — dataset used for training/analysis (CSV)
- `database.csv` — analysis history database (CSV, auto-created on first run)
- `imagecaption.py` — BLIP image captioning implementation
- `main.py` — Streamlit web application entry point
- `requirements.txt` — Python dependencies
- `tokenizer.pkl` — saved tokenizer for text preprocessing
- `label_encoder.pkl` — label encoder for toxicity classes

## Quick setup (Windows, PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

Required packages:
- `streamlit` — web interface
- `tensorflow` — for BiLSTM model
- `nltk` — text preprocessing
- `pillow` — image handling
- `pandas` — data management
- BLIP dependencies (see `requirements.txt`)

Notes:
- If you need GPU support, install GPU-enabled TensorFlow
- First run may download BLIP model weights

## Running the Streamlit app

This is a Streamlit web application that provides an interactive UI for toxicity analysis. To run:

```powershell
# Start the Streamlit server
streamlit run main.py
```

The app will open in your default web browser with:
- Text input field for messages
- Image upload option
- Real-time analysis results
- Analysis history viewer

### Features
1. **Text Analysis**
   - Enter any text to analyze
   - Text is preprocessed using NLTK (lemmatization, stopwords removal)
   - BiLSTM model predicts toxicity with confidence scores

2. **Image Understanding**
   - Upload images in JPG/JPEG/PNG formats
   - BLIP model generates natural language captions
   - Captions provide additional context for analysis

3. **Combined Analysis**
   - Text and image captions are preprocessed separately
   - Combined as: `preprocessed_text + " " + preprocessed_image_caption`
   - Single prediction based on full context

4. **Analysis History**
   - All analyses are saved to `database.csv`
   - View history with preprocessed inputs
   - Download complete analysis history

```python
# quick test (example - modify to match your main.py API)
from tensorflow.keras.models import load_model
model = load_model('bilstm_toxic_classifier_seed42.h5')
# Prepare text -> tokens -> model.predict([...])
```

## How to use the model for inference

1. Preprocess input text the same way training used it: tokenization, sequence padding, and any vocabulary mapping.
2. Load the model:

```python
from tensorflow.keras.models import load_model
model = load_model('bilstm_toxic_classifier_seed42.h5')
```

3. Call `model.predict(...)` on prepared input arrays and interpret outputs (sigmoid/probabilities or softmax depending on output layer).

## Dataset and privacy

- `cellula_toxic_data.csv` likely contains labeled examples. Inspect it before using it in production.
- Ensure you have the right to use any data included and follow privacy/consent rules when using or publishing results.




