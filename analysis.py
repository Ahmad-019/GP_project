import pandas as pd
import torch
import emoji
import re
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- PHASE 3: ADVANCED ANALYTICS AND AI MODELING ---

def run_sentiment_analysis(input_file, output_file):
    # Check if the cleaned dataset exists
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found. Please run cleaning.py first.")
        return

    # 1. Load Cleaned Dataset
    df = pd.read_csv(input_file)
    print(f"SYSTEM: Loading {len(df)} cleaned signals for AI processing...")

    # 2. Text Preprocessing (Standardizing for NLP)
    def preprocess_text(text):
        text = str(text)
        # Convert emojis to text descriptors to preserve emotional signals
        text = emoji.demojize(text, delimiters=(" ", " ")) 
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip() 
        # Reduce repeated characters (e.g., loooove -> loove)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text) 
        return text

    df['Processed_Text'] = df['Comment_Content'].apply(preprocess_text)

    # 3. Load Pre-trained Multilingual BERT Model
    # Selected for its high performance on both Arabic and English text
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval() # Set model to evaluation mode (Faster inference)

    def get_sentiment(text):
        # Tokenizing text and preparing tensors for PyTorch
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad(): # Disable gradient calculation for speed
            outputs = model(**inputs)
            
        # Convert logits to probabilities using Softmax
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        stars = probs.argmax() + 1 # Model predicts on a 1-5 star scale
        confidence = probs.max() # Extracting prediction reliability
        return stars, confidence

    print("STATUS: Initializing AI Inference... (This might take a minute)")
    
    # Applying AI prediction to each actionable signal
    results = df['Processed_Text'].apply(lambda x: pd.Series(get_sentiment(x)))
    df[['AI_Star_Rating', 'AI_Confidence']] = results

    # 4. Hybrid Sentiment Scoring (Data Enrichment)
    # Mapping 1-5 stars to a numeric range between -1 and +1 for BI Heatmaps
    rating_map = {1: -1, 2: -0.5, 3: 0, 4: 0.5, 5: 1}
    df['Sentiment_Score'] = df['AI_Star_Rating'].map(rating_map) * df['AI_Confidence']

    # 5. Export Final Enriched Dataset for Power BI
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print("AI ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*40)
    print(f"Total Processed     : {len(df)} records")
    print(f"Data Enrichment     : Sentiment Scores & Confidence added")
    print(f"Final Output File   : {output_file}")
    print("="*40)

if __name__ == "__main__":
    # Ensure filenames match your cleaned dataset
    INPUT = 'GP_Cleaned_Data.csv'
    OUTPUT = 'GP_Enriched_AI_Data.csv'
    
    run_sentiment_analysis(INPUT, OUTPUT)