import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

# NLTK preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Import BLIP image captioning
from imagecaption import generate_caption

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# -------------------------
# Load Model + Tokenizer + Encoder
# -------------------------
model = load_model("src/bilstm_toxic_classifier_seed42.h5", compile=False) 

with open("src/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("src/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Define max_length used in training
max_length = 75   # change if different

# -------------------------
# Preprocessing Function (Same as Training)
# -------------------------
def preprocess_text(text, use_lemmatization=True, remove_stopwords=True):
    """Clean and normalize text with the same preprocessing used in training"""
    if pd.isna(text) or text == "":
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove extra whitespace and newlines
    text = re.sub(r"\s+", " ", text)
    
    # Remove special characters but keep basic punctuation (.?!,)
    text = re.sub(r"[^\w\s\.\?\!,]", "", text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords (optional)
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    
    # Apply stemming or lemmatization
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    else:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens back
    text = " ".join(tokens).strip()
    
    return text

# -------------------------
# Combined Input Processing (Matches Training)
# -------------------------
def prepare_combined_input(user_text, image_caption):
    """
    Prepare combined input exactly like in training:
    'query_clean' + ' ' + 'image_desc_clean'
    """
    # Preprocess both inputs separately (same as training)
    text_clean = preprocess_text(user_text)
    image_clean = preprocess_text(image_caption)
    
    # Combine exactly like in training: text + space + image_desc
    combined_input = f"{text_clean} {image_clean}".strip()
    
    return combined_input, text_clean, image_clean

# -------------------------
# Prediction Function
# -------------------------
def predict_toxicity(combined_input):
    """Make prediction on the combined input"""
    original_text = combined_input
    text_clean = combined_input  # Already preprocessed
    
    seq = tokenizer.texts_to_sequences([text_clean])
    padded = pad_sequences(seq, maxlen=max_length)
    prediction = model.predict(padded, verbose=0)
    predicted_class = le.classes_[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    
    return original_text, text_clean, predicted_class, confidence

# -------------------------
# Database Management
# -------------------------
DB_FILE = "database.csv"

def save_to_db(original_text, cleaned_text, predicted_class, confidence, 
               user_text_clean="", image_caption_clean="", input_type="combined"):
    new_entry = pd.DataFrame([{
        "original_input": original_text,
        "cleaned_input": cleaned_text,
        "user_text_clean": user_text_clean,
        "image_caption_clean": image_caption_clean,
        "prediction": predicted_class,
        "confidence": confidence,
        "input_type": input_type,
        "timestamp": pd.Timestamp.now()
    }])
    if os.path.exists(DB_FILE):
        db = pd.read_csv(DB_FILE)
        db = pd.concat([db, new_entry], ignore_index=True)
    else:
        db = new_entry
    db.to_csv(DB_FILE, index=False)

# -------------------------
# Streamlit UI
# -------------------------
st.title("🚨 Toxicity Classifier ")

st.markdown("""
### How it works:
1. **Upload an image** and/or **enter text**
2. The app generates an image caption (if image provided)
3. Both inputs are preprocessed separately
4. Combined as: `preprocessed_text + ' ' + preprocessed_image_caption`
5. Single prediction based on combined context (matches training data format)
""")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Your Text Input")
    user_text = st.text_area(
        "Enter your text (optional):",
        height=120,
        placeholder="Type your message here...",
        key="text_input"
    )

with col2:
    st.subheader("🖼️ Your Image Input")
    uploaded_img = st.file_uploader(
        "Upload an image (optional):", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image to add visual context to your text",
        key="image_uploader"
    )
    if uploaded_img is not None:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

# Combined classification button
if st.button("🔍 Analyze Combined Input", type="primary", use_container_width=True):
    
    # Validate that at least one input is provided
    if not user_text.strip() and uploaded_img is None:
        st.error("❌ Please provide at least one input (text or image)")
        st.stop()
    
    with st.spinner("🔄 Processing your inputs..."):
        # Generate and preprocess image caption if image is provided
        image_caption = ""
        image_clean = ""
        if uploaded_img is not None:
            try:
                image_caption = generate_caption(image)
                st.success(f"📸 Generated Image Caption: **{image_caption}**")
            except Exception as e:
                st.error(f"❌ Error generating image caption: {str(e)}")
                image_caption = ""
        
        # Preprocess and combine inputs (EXACTLY like training)
        combined_input, text_clean, image_clean = prepare_combined_input(user_text, image_caption)
        
        # Display preprocessing details
        st.subheader("🔧 Preprocessing Steps")
        
        col1, col2 = st.columns(2)
        with col1:
            if user_text.strip():
                st.write("**Original Text:**", user_text)
                st.write("**Preprocessed Text:**", text_clean if text_clean else "[Empty]")
        
        with col2:
            if image_caption:
                st.write("**Original Image Caption:**", image_caption)
                st.write("**Preprocessed Caption:**", image_clean if image_clean else "[Empty]")
        
        # Show combined input
        st.subheader("🧩 Combined Input Sent to Model")
        st.info(f"**{combined_input}**")
        st.caption("Format: preprocessed_text + ' ' + preprocessed_image_caption")
        
        # Make prediction
        if combined_input.strip():
            original, cleaned, predicted_class, confidence = predict_toxicity(combined_input)
            
            # Display results
            st.subheader("📊 Analysis Results")
            
            # Color code based on prediction
            if predicted_class == "toxic":
                st.error(f"🚨 Prediction: **{predicted_class.upper()}** (confidence: {confidence:.2%})")
            else:
                st.success(f"✅ Prediction: **{predicted_class.upper()}** (confidence: {confidence:.2%})")
            
            # Save to database with detailed info
            save_to_db(original, cleaned, predicted_class, confidence, 
                      text_clean, image_clean, "combined")
            
        else:
            st.warning("No valid input to analyze after preprocessing.")

# Show database
st.markdown("---")
if st.checkbox("📂 Show Analysis History"):
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        
        # Display database with relevant columns
        display_cols = ['user_text_clean', 'image_caption_clean', 'cleaned_input', 'prediction', 'confidence', 'timestamp']
        available_cols = [col for col in display_cols if col in df.columns]
        
        st.dataframe(df[available_cols], use_container_width=True)
        
        # Statistics
        st.subheader("📈 Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analyses", len(df))
        with col2:
            toxic_count = len(df[df['prediction'] == 'toxic'])
            st.metric("Toxic Content", toxic_count)
        with col3:
            non_toxic_count = len(df[df['prediction'] == 'non-toxic'])
            st.metric("Non-Toxic Content", non_toxic_count)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Analysis History",
            data=csv,
            file_name="toxicity_analysis_history.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No analysis history yet. Perform your first analysis above!")
