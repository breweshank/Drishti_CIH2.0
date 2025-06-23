import os
import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

# Initialize nltk resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

### 1. Audio Feature Extraction from WAV File ###
def extract_audio_features(wav_file):
    try:
        print(f"Extracting audio features from {wav_file}...")
        y, sr = librosa.load(wav_file, sr=None)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # Extract Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[pitches > 0]
        pitch_mean = np.mean(pitches) if len(pitches) > 0 else 0
        
        # Extract Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
        
        # Combine features into a single feature vector
        audio_features = np.concatenate((mfccs_mean, [pitch_mean], spectral_contrast_mean))
        print("Audio features extracted successfully.")
        return audio_features
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")
        return None

### 2. Formant Feature Extraction using Praat from WAV and Formant CSV ###
def extract_formant_features(wav_file):
    try:
        print(f"Extracting formant features from {wav_file}...")
        sound = parselmouth.Sound(wav_file)
        formant_object = call(sound, "To Formant (burg)", 0.0, 5.0, 5500, 0.025, 50.0)
        
        formants = []
        for t in np.arange(0, sound.get_total_duration(), 0.01):
            f1 = call(formant_object, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formant_object, "Get value at time", 2, t, "Hertz", "Linear")
            f3 = call(formant_object, "Get value at time", 3, t, "Hertz", "Linear")
            formants.append([f1, f2, f3])
        
        formants = np.array(formants)
        formant_features = np.nanmean(formants, axis=0)  # Averaging formants over time
        print("Formant features extracted successfully.")
        return formant_features
    except Exception as e:
        print(f"Error extracting formant features from {wav_file}: {e}")
        return None

def load_formant_csv(csv_file):
    try:
        print(f"Loading formant features from {csv_file}...")
        formant_data = pd.read_csv(csv_file)
        formant_features_csv = formant_data.mean(axis=0).values
        print("Formant features loaded successfully.")
        return formant_features_csv
    except Exception as e:
        print(f"Error loading formant CSV {csv_file}: {e}")
        return None

### 3. Text Feature Extraction using TF-IDF ###
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

def extract_text_features(transcript_file):
    try:
        print(f"Extracting text features from {transcript_file}...")
        transcript = pd.read_csv(transcript_file)
        if "start_time	stop_time	speaker	value" in transcript.columns:
            transcript_text = transcript['start_time	stop_time	speaker	value'].astype(str).tolist()
        
        cleaned_transcripts = [preprocess_text(t) for t in transcript_text]
        
        # Vectorize text using TF-IDF
        vectorizer = TfidfVectorizer(max_features=100)
        text_features = vectorizer.fit_transform(cleaned_transcripts).toarray()
        
        print("Text features extracted successfully.")
        return np.mean(text_features, axis=0)
    except Exception as e:
        print(f"Error extracting text features from {transcript_file}: {e}")
        return None

### 4. Covarep Feature Extraction ###
def load_covarep_csv(covarep_csv_file):
    try:
        print(f"Loading Covarep features from {covarep_csv_file}...")
        covarep_data = pd.read_csv(covarep_csv_file)
        covarep_features = covarep_data.mean(axis=0).values  # Averaging Covarep features over all columns
        print("Covarep features loaded successfully.")
        return covarep_features
    except Exception as e:
        print(f"Error loading covarep CSV {covarep_csv_file}: {e}")
        return None

### 5. PHQ-8 Score Extraction ###
def extract_phq8_score(phq8_file, participant_id):
    print(f"Extracting PHQ-8 score for participant {participant_id}...")
    phq8_data = pd.read_csv(phq8_file)
    # Ensure participant IDs are strings
    phq8_data['Participant_ID'] = phq8_data['Participant_ID'].astype(str)
    participant_id = str(participant_id)
    phq8_score_row = phq8_data[phq8_data['Participant_ID'] == participant_id]

    if not phq8_score_row.empty:
        phq8_score = phq8_score_row['PHQ8_Score'].values[0]
        print(f"PHQ-8 score for participant {participant_id}: {phq8_score}")
        return phq8_score
    else:
        print(f"PHQ-8 score not found for participant {participant_id}")
        return None

### Scaling Features ###
def scale_features_individually(audio_features, formant_features_wav, formant_features_csv, covarep_features, text_features):
    # Combine all features into one vector
    combined_features = np.concatenate((
        audio_features, 
        formant_features_wav, 
        formant_features_csv, 
        covarep_features, 
        text_features
    ))
    print("Features combined into a single vector.")
    return combined_features

### 6. Process multiple datasets ###
def process_multiple_data(dataset_folder, phq8_file):
    print(f"Processing data from folder: {dataset_folder}...")
    processed_data = []
    participant_ids = []
    
    # Iterate through each participant's folder
    for participant_folder in os.listdir(dataset_folder):
        
        participant_path = os.path.join(dataset_folder, participant_folder)
        
        if os.path.isdir(participant_path):
            print(f"Processing participant: {participant_folder}")
            # Construct the paths for audio, formant, covarep, and transcript files
            participant_id = participant_folder
            wav_file = os.path.join(participant_path, f"{participant_folder}_AUDIO.wav")
            formant_csv = os.path.join(participant_path, f"{participant_folder}_FORMANT.csv")
            covarep_csv = os.path.join(participant_path, f"{participant_folder}_COVAREP.csv")
            transcript = os.path.join(participant_path, f"{participant_folder}_TRANSCRIPT.csv")
            
            # Extract features
            audio_features = extract_audio_features(wav_file)
            formant_features_wav = extract_formant_features(wav_file)
            formant_features_csv = load_formant_csv(formant_csv)
            covarep_features = load_covarep_csv(covarep_csv)
            text_features = extract_text_features(transcript)
            phq8_score = extract_phq8_score(phq8_file, participant_id)
            
            # Check if any feature extraction failed (is None)
            if (audio_features is None or 
                formant_features_wav is None or 
                formant_features_csv is None or 
                covarep_features is None or 
                text_features is None or 
                phq8_score is None):
                print(f"Skipping participant {participant_id} due to missing data")
                continue
            
            # Combine features without scaling
            combined_features = scale_features_individually(audio_features, formant_features_wav, formant_features_csv, covarep_features, text_features)
            
            processed_data.append((combined_features, phq8_score))
            participant_ids.append(participant_id)
    
    print(f"Processed data for {len(processed_data)} participants.")
    return processed_data, participant_ids

if __name__ == "__main__":
    # Folder path containing participant subfolders
    dataset_folder = r"D:\Khushi\Dataset\final_train"
    
    # PHQ-8 scores file path
    phq8_file = r"D:\Khushi\Dataset\final_train\train_split_Depression_AVEC2017.csv"
    
    # Call the function to process the data
    processed_data = process_multiple_data(dataset_folder, phq8_file)

    print("Data processing complete.")


# After processing the data
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("Processed data saved successfully.")
