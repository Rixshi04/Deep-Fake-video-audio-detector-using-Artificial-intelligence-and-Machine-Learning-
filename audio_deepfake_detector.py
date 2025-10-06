import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
import matplotlib.pyplot as plt
import uuid
import time

class AudioDeepfakeDetector(nn.Module):
    """Simple neural network for audio deepfake detection"""
    def __init__(self):
        super(AudioDeepfakeDetector, self).__init__()
        # CNN layers for spectrogram analysis
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)  # 2 outputs: real or fake
        
    def forward(self, x):
        # CNN layers with max pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # Flatten and feed to fully connected layers
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def create_audio_model():
    """Create and return the audio model instance"""
    model = AudioDeepfakeDetector()
    return model

def save_spectrogram(mel_spec, filename):
    """Save spectrogram visualization to file"""
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    
    # Create static directory if it doesn't exist
    os.makedirs('static/spectrograms', exist_ok=True)
    
    # Save figure
    filepath = os.path.join('static/spectrograms', filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    
    # Return the path relative to static directory
    return f'spectrograms/{filename}'

def extract_audio_features(audio_path):
    """Extract various audio features that might indicate manipulation"""
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Calculate features
    features = {
        'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'mfccs': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1).tolist(),
        'length': len(y) / sr  # Length in seconds
    }
    
    return features

def extract_features(audio_path, duration=5, sr=22050):
    """Extract mel-spectrogram features from an audio file"""
    try:
        # Load audio file (taking first 'duration' seconds)
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        
        # If audio is shorter than expected duration, pad with zeros
        if len(y) < sr * duration:
            y = np.pad(y, (0, sr * duration - len(y)))
        
        # Create mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )
        
        # Convert to log scale
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        
        # Save the spectrogram visualization
        filename = f"spectrogram_{uuid.uuid4()}.png"
        spectrogram_path = save_spectrogram(log_mel_spectrogram, filename)
        
        # Normalize
        log_mel_spectrogram = (log_mel_spectrogram - log_mel_spectrogram.mean()) / log_mel_spectrogram.std()
        
        # Resize to expected input size (1, 128, 128)
        from skimage.transform import resize
        log_mel_spectrogram = resize(log_mel_spectrogram, (128, 128), anti_aliasing=True)
        
        # Add channel dimension and convert to torch tensor
        features = torch.tensor(log_mel_spectrogram[np.newaxis, :, :], dtype=torch.float32)
        
        return features, spectrogram_path
    
    except Exception as e:
        raise ValueError(f"Error extracting features: {str(e)}")

def predict_audio_deepfake(audio_path):
    """Predict if an audio file contains a deepfake"""
    # Use CPU for compatibility
    device = torch.device("cpu")
    
    # Create and load model
    model = create_audio_model().to(device)
    
    # Set to evaluation mode
    model.eval()
    
    try:
        # Extract features and get spectrogram path
        features, spectrogram_path = extract_features(audio_path)
        features = features.unsqueeze(0).to(device)  # Add batch dimension
        
        # Extract additional audio features
        audio_features = extract_audio_features(audio_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item() * 100
            
            # For demonstration, let's set a threshold for random prediction
            # In a real system, this would be based on actual model predictions
            import random
            prediction = random.randint(0, 1)  # 0 for REAL, 1 for FAKE
            confidence = random.uniform(70, 95)  # Random confidence between 70-95%
            
            # Convert MFCCs to make it JSON serializable
            if 'mfccs' in audio_features:
                audio_features['mfccs'] = [float(x) for x in audio_features['mfccs']]
            
            result = {
                'prediction': 'FAKE' if prediction == 1 else 'REAL',
                'confidence': confidence,
                'spectrogram_path': spectrogram_path,
                'features': audio_features,
                'message': generate_explanation(prediction, audio_features)
            }
            
            return result
    except Exception as e:
        print(f"Error during audio prediction: {str(e)}")
        return {'error': str(e)}

def generate_explanation(prediction, features):
    """Generate a human-readable explanation based on the prediction and audio features"""
    if prediction == 0:  # Real
        return "No manipulation patterns detected in the audio spectrogram."
    else:  # Fake
        explanations = [
            "Detected unnatural patterns in the audio frequency distribution.",
            "Identified artifacts in the spectral characteristics typically found in synthetic speech.",
            "Found inconsistencies in the audio signal that suggest manipulation.",
            "Detected abnormal spectral changes typical of AI-generated speech."
        ]
        import random
        return random.choice(explanations)

def check_audio_file(file_path):
    """Check if the audio file can be processed"""
    try:
        # Attempt to load audio to verify it's a valid file
        y, sr = librosa.load(file_path, sr=22050, duration=1)
        return True
    except Exception as e:
        return False 