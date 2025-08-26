import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import time
import random
import sqlite3
import hashlib
import datetime
import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai

genai.configure(api_key="AIzaSyCUw-D_ayVTIereW_JYQPVtz08JGGPKydA")
st.set_page_config(page_title="Sensory Overload Navigator", layout="wide")

# Download NLTK resources
try:
    nltk.data.find('vader_lexicon')
except:
    nltk.download('vader_lexicon')
GEMINI_API_KEY = "AIzaSyCUw-D_ayVTIereW_JYQPVtz08JGGPKydA" 

try:
    genai.configure(api_key=GEMINI_API_KEY)
except:
    st.warning("Chat features will be limited.")

# Set page configuration
st.set_page_config(
    page_title="Sensory Overload Navigator",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling

st.markdown("""
<style>
    :root {
        --primary: #130220;
        --primary-light: #0a364e;
        --secondary: #1c023a;
        --accent: #ff6b6b;
        --text: #0e0c0c;
        --text-light: #666666;
        --background: #05021e;
        --card-bg: #0b0c32;
        --success: #00cc66;
        --warning: #ffa500;
        --danger: #ff4b4b;
    }
        .stApp {
            background-color: #2E003E;
        }
        .stMarkdown, .stText, .stButton, .stSelectbox, .stTextInput, .stDataFrame {
            color: white;
        }
        h1, h2, h3, h4, h5, h6, p, label {
            color: white !important;
        }
    .main-header {
        font-size: 3rem;
        color: var(--primary);
        text-align: center;
        color:#1f2937;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.8rem;
        color: var(--secondary);
        margin-bottom: 1.5rem;
        font-weight: 600;
        color:#1f2937;
        border-left: 4px solid var(--primary);
        padding-left: 1rem;
    }
    .section-header {
        font-size: 1.4rem;
        color: var(--secondary);
        color:#1f2937;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
    }
    .recommendation-box {
        background-color: var(--primary-light);
        padding: 1.5rem;
        border-radius: 12px;
        color:#1f2937;
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--primary);
        box-shadow: 0 4px 12px rgba(106, 13, 173, 0.1);
    }
    .metric-box {
        background: rgb(13, 10, 39);
        padding: 1.2rem;
        color:#1f2937;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(106, 13, 173, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-box:hover {
        transform: translateY(-5px);
    }
    .stButton>button {
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 5px 15px rgba(106, 13, 173, 0.3);
        transform: translateY(-2px);
    }
    .environment-tag {
        display: inline-block;
        background-color: rgba(106, 13, 173, 0.1);
        padding: 0.4rem 1rem;
        color: var(--primary);
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .login-container {
        max-width: 450px;
        margin: 2rem auto;
        padding: 2.5rem;
            background: rgb(13, 10, 39);
        color: var(--text);
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    .profile-section {
            background: rgb(13, 10, 39);
        padding: 2rem;
        color: var(--text);
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    .prediction-high {
        color: var(--danger);
        font-weight: 700;
    }
    .prediction-medium {
        color: var(--warning);
        font-weight: 700;
    }
    .prediction-low {
        color: var(--success);
        font-weight: 700;
    }
    .chat-container {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        height: 500px;
        overflow-y: auto;
        margin-bottom: 1rem;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    .user-message {
        background: rgb(13, 10, 39);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        text-align: right;
    }
    .bot-message {
        background: rgb(13, 10, 39);
        color: var(--text);
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        color:#cbd6e4;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .tab-container {
        background: var(--card-bg);
        border-radius: 12px;
        color:#1f2937;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    .feature-card {
        background: rgb(0, 1, 2);
        border-radius: 12px;
        padding: 1.5rem;
        color:#1f2937;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-left: 4px solid var(--primary);
    }
    /* Custom slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
    }
    /* Remove unnecessary white boxes */
    .stApp > header {
        background-color: transparent;
    }
    .stApp {
        margin: 0;
        padding: 0;
        background: #f8f9fa;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }


    /* ===== Global Background & Text ===== */
.stApp {
        background-color:  #0e0316!important;  /* Deep Purple */
    color: #E6E6FA !important;             /* Light Lavender */
}

/* Fix background for all containers */
.main, .block-container, .css-18e3th9, .css-1d391kg {
    background-color:  #0e0316!important;
    color: #E6E6FA !important;
}

/* ===== Text Styling ===== */
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #bcf9f8 !important;
}

/* ===== Input Fields ===== */
.stTextInput > div > div > input,
.stPassword > div > div > input,
textarea {
    background-color:  #0e0316!important;  /* Darker purple */
    color: #E6E6FA !important;
    border: 1px solid #6A0DAD !important;
    border-radius: 6px;
}

/* Placeholder text */
.stTextInput > div > div > input::placeholder,
textarea::placeholder {
    color: #BFAFD4 !important;  /* Muted lavender */
}

/* ===== Buttons ===== */
.stButton > button {
    background: transparent !important;
    color: #FFFFFF !important;
    border-radius: 8px;
    border: 1px solid #E6E6FA !important;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    color: #FFFFFF !important;
}

/* ===== Tabs / Radio Buttons ===== */
.stTabs [role="tablist"] button,
.stRadio > label {
    color: #E6E6FA !important;
}

.stTabs [role="tablist"] button[aria-selected="true"] {
    color: #FFFFFF !important;
}

/* ===== Sidebar ===== */
.css-1d391kg, .css-1lcbmhc, .stSidebar {
    background-color: #130220;
    color: #E6E6FA !important;
}

/* ===== DataFrames / Tables ===== */
.stDataFrame, .dataframe {
    background-color: #130220;
    color: #E6E6FA !important;
}

.stDataFrame th, .dataframe th {
    background-color: #130220;
    color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)
class EnhancedEnvironmentCNN(nn.Module):
    def __init__(self, num_classes=4):  # crowd, complexity, lighting, noise
        super(EnhancedEnvironmentCNN, self).__init__()
        
        # Feature extraction backbone (pretrained ResNet)
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove classification head
        
        # Additional layers for our specific task
        self.regression_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.regression_head(features)
        return output

# Advanced NLP Model for Text Processing
class AdvancedSensoryNLP:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
        
    def analyze_text(self, text):
        if not text or text.strip() == "":
            return np.zeros(8)
        
        # Sentiment analysis
        sentiment = self.sia.polarity_scores(text)
        
        # Text complexity features
        words = text.split()
        sentences = text.split('.')
        word_count = len(words)
        sentence_count = len([s for s in sentences if len(s.strip()) > 0])
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        unique_words = len(set(words))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        
        # Text features
        features = np.array([
            sentiment['neg'],
            sentiment['neu'],
            sentiment['pos'],
            sentiment['compound'],
            word_count / 100,  # Normalized word count
            avg_sentence_length / 10,  # Normalized sentence length
            lexical_diversity,
            len(text) / 500  # Normalized text length
        ])
        
        return features

class SensoryChatbot:
    def __init__(self):
        self.chat_history = []
        self.api_key_configured = True  # already configured at import

    def get_response(self, user_input: str) -> str:
        if not self.api_key_configured:
            return "Chat features are currently unavailable."

        try:
            # ✅ Use latest Gemini model
            model = genai.GenerativeModel("gemini-2.0-flash")

            # Add user input to history
            self.chat_history.append({"role": "user", "parts": [user_input]})

            # Generate response with conversation context
            response = model.generate_content(self.chat_history)

            # Save model reply into history
            if hasattr(response, "text"):
                reply = response.text
            else:
                reply = "Sorry, I couldn't generate a response."

            self.chat_history.append({"role": "model", "parts": [reply]})

            return reply

        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try again later."
st.set_page_config(page_title="Sensory Overload Navigator", layout="wide")

# Enhanced Ensemble Model for Sensory Prediction
class AdvancedSensoryPredictor:
    def __init__(self):
        self.cnn_model = None
        self.nlp_model = AdvancedSensoryNLP()
        self.ml_model = None
        self.scaler = StandardScaler()
        self.environment_types = ["Shopping Mall", "Supermarket", "Public Transport", 
                                 "Restaurant/Cafe", "Park", "Office", "Classroom", "Gym", 
                                 "Concert", "Library", "Airport", "Hospital", "Custom"]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.models_loaded = False
        
    def load_models(self):
        if self.models_loaded:
            return
            
        try:
            # Load or create CNN model
            self.cnn_model = EnhancedEnvironmentCNN()
            cnn_path = 'enhanced_environment_cnn.pth'
            if os.path.exists(cnn_path):
                self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=torch.device('cpu')))
            else:
                # Initialize with pretrained weights
                torch.save(self.cnn_model.state_dict(), cnn_path)
                
            self.cnn_model.eval()
            
            # Load or train ML model
            ml_path = 'advanced_sensory_ml_model.pkl'
            if os.path.exists(ml_path):
                self.ml_model = joblib.load(ml_path)
            else:
                self._train_ml_model()
            
            self.models_loaded = True
        except Exception as e:
            st.error(f"Error loading models: {e}")
    
    def _train_ml_model(self):
        # Generate more realistic synthetic training data
        np.random.seed(42)
        n_samples = 5000
        
        # Features: noise_level, crowd_density, lighting_conditions, visual_complexity, environment_type
        X = np.zeros((n_samples, 12))  # 5 base features + 4 from CNN + 3 from 
        env_params = {
            "Shopping Mall": {"noise": (70, 15), "crowd": (75, 20), "lighting_conditions": (80, 15), "visual": (85, 10)},
            "Supermarket": {"noise": (65, 10), "crowd": (60, 15), "lighting_conditions": (85, 8), "visual": (75, 12)},
            "Public Transport": {"noise": (75, 12), "crowd": (80, 15), "lighting_conditions": (70, 10), "visual": (70, 15)},
            "Restaurant/Cafe": {"noise": (65, 8), "crowd": (60, 10), "lighting_conditions": (75, 5), "visual": (70, 8)},
            "Park": {"noise": (50, 20), "crowd": (40, 25), "lighting_conditions": (70, 25), "visual": (60, 20)},
            "Office": {"noise": (60, 10), "crowd": (50, 15), "lighting_conditions": (75, 10), "visual": (65, 12)},
            "Classroom": {"noise": (65, 8), "crowd": (55, 10), "lighting_conditions": (80, 5), "visual": (70, 8)},
            "Gym": {"noise": (70, 10), "crowd": (60, 15), "lighting_conditions": (75, 10), "visual": (70, 12)},
            "Concert": {"noise": (90, 5), "crowd": (85, 10), "lighting_conditions": (60, 20), "visual": (80, 15)},
            "Library": {"noise": (45, 10), "crowd": (30, 15), "lighting_conditions": (70, 10), "visual": (60, 12)},
            "Airport": {"noise": (75, 10), "crowd": (70, 15), "lighting_conditions": (80, 8), "visual": (75, 10)},
            "Hospital": {"noise": (60, 8), "crowd": (50, 12), "lighting_conditions": (75, 5), "visual": (65, 8)},
            "Custom": {"noise": (65, 20), "crowd": (60, 25), "lighting_conditions": (70, 20), "visual": (65, 20)}
        }
        
        for i, (env, params) in enumerate(env_params.items()):
            start_idx = i * (n_samples // len(env_params))
            end_idx = (i + 1) * (n_samples // len(env_params))
            
            # Generate base features
            X[start_idx:end_idx, 0] = np.random.normal(params["noise"][0], params["noise"][1], end_idx - start_idx)
            X[start_idx:end_idx, 1] = np.random.normal(params["crowd"][0], params["crowd"][1], end_idx - start_idx)
            X[start_idx:end_idx, 2] = np.random.normal(params["lighting_conditions"][0], params["lighting_conditions"][1], end_idx - start_idx)
            X[start_idx:end_idx, 3] = np.random.normal(params["visual"][0], params["visual"][1], end_idx - start_idx)
            X[start_idx:end_idx, 4] = i  # environment type
            
            # Simulated CNN features (more realistic)
            X[start_idx:end_idx, 5] = np.clip(X[start_idx:end_idx, 1] / 100 + np.random.normal(0, 0.05, end_idx - start_idx), 0, 1)
            X[start_idx:end_idx, 6] = np.clip(X[start_idx:end_idx, 3] / 100 + np.random.normal(0, 0.05, end_idx - start_idx), 0, 1)
            X[start_idx:end_idx, 7] = np.clip(X[start_idx:end_idx, 2] / 100 + np.random.normal(0, 0.05, end_idx - start_idx), 0, 1)
            X[start_idx:end_idx, 8] = np.clip(X[start_idx:end_idx, 0] / 120 + np.random.normal(0, 0.05, end_idx - start_idx), 0, 1)
            
            # Simulated NLP features
            X[start_idx:end_idx, 9] = np.random.normal(0.3, 0.1, end_idx - start_idx)  # sentiment
            X[start_idx:end_idx, 10] = np.random.normal(0.5, 0.15, end_idx - start_idx)  # complexity
            X[start_idx:end_idx, 11] = np.random.normal(0.4, 0.1, end_idx - start_idx)  # length
        
        # Calculate target (sensory score) based on features with realistic weighting
        y = (X[:, 0] * 0.25 + X[:, 1] * 0.20 + X[:, 2] * 0.15 + X[:, 3] * 0.10 + 
             X[:, 5] * 0.08 + X[:, 6] * 0.07 + X[:, 7] * 0.05 + X[:, 8] * 0.05 +
             X[:, 9] * 0.02 + X[:, 10] * 0.02 + X[:, 11] * 0.01)
        
        y += np.random.normal(0, 0.8, n_samples)  # Add realistic noise
        y = np.clip(y, 0, 10)  # Ensure scores are between 0-10
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Train advanced model (Gradient Boosting for better performance)
        self.ml_model = GradientBoostingRegressor(
            n_estimators=150, 
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.ml_model.fit(X, y)
        
        # Save model and scaler
        joblib.dump(self.ml_model, 'advanced_sensory_ml_model.pkl')
        joblib.dump(self.scaler, 'feature_scaler.pkl')
        
    def analyze_image(self, image):
        if image is None:
            # Return intelligent default features based on common environments
            return np.array([0.6, 0.5, 0.7, 0.6])
            
        try:
            # Ensure models are loaded
            self.load_models()
            
            # Preprocess image
            image = image.convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            # Get CNN predictions
            with torch.no_grad():
                predictions = self.cnn_model(image_tensor)
                features = predictions.numpy()[0]
                
            # Apply sigmoid to get values between 0-1
            features = 1 / (1 + np.exp(-features))
            
            return np.clip(features, 0, 1)
            
        except Exception as e:
            st.error(f"Error analyzing image: {e}")
            # Return default features on error
            return np.array([0.5, 0.5, 0.5, 0.5])
    
    def predict(self, noise_level=None, crowd_density=None, lighting_conditions=None, 
                visual_complexity=None, environment_type="Custom", image=None, user_profile_text=""):
        # Ensure models are loaded
        self.load_models()
        
        # Handle missing values with intelligent defaults
        if noise_level is None:
            noise_level = 65  # Moderate default
        if crowd_density is None:
            crowd_density = 50  # Medium default
        if lighting_conditions is None:
            lighting_conditions = 60  # Moderate default
        if visual_complexity is None:
            visual_complexity = 50  # Medium default
        
        # Encode environment type
        try:
            env_encoded = self.environment_types.index(environment_type)
        except:
            env_encoded = len(self.environment_types) - 1  # Default to Custom
        
        # Analyze image if provided
        image_features = self.analyze_image(image)
        
        # Analyze user profile text
        nlp_features = self.nlp_model.analyze_text(user_profile_text)
        
        # Create feature array
        base_features = np.array([noise_level, crowd_density, lighting_conditions, 
                                 visual_complexity, env_encoded])
        
        # Combine all features
        all_features = np.concatenate([base_features, image_features, nlp_features])
        all_features = all_features.reshape(1, -1)
        
        # Scale features
        try:
            all_features = self.scaler.transform(all_features)
        except:
            pass  # If scaler not available, use raw features
        
        # Predict sensory score
        try:
            score = self.ml_model.predict(all_features)[0]
            score = max(0, min(10, score))  # Ensure score is between 0-10
        except:
            # Fallback calculation if model prediction fails
            score = (noise_level * 0.3 + crowd_density * 0.25 + lighting_conditions * 0.2 + 
                    visual_complexity * 0.15 + np.mean(image_features) * 0.1) / 10
        
        # Determine risk level
        if score < 3.5:
            risk = "Low"
            color = "green"
        elif score < 6.5:
            risk = "Medium"
            color = "orange"
        else:
            risk = "High"
            color = "red"
            
        return score, risk, color, image_features

# Initialize AI model and chatbot
predictor = AdvancedSensoryPredictor()
chatbot = SensoryChatbot()

# Database functions
def init_db():
    conn = sqlite3.connect('sensory_app.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  email TEXT,
                  full_name TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create user profiles table
    c.execute('''CREATE TABLE IF NOT EXISTS user_profiles
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  age INTEGER,
                  sensory_profile TEXT,
                  triggers TEXT,
                  coping_strategies TEXT,
                  preferences TEXT,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Create environment history table
    c.execute('''CREATE TABLE IF NOT EXISTS environment_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  environment_type TEXT,
                  sensory_score REAL,
                  overload_risk TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Create chat history table
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  message TEXT,
                  is_user BOOLEAN,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Insert sample data if tables are empty
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        # Create sample user
        hashed_password = hashlib.sha256("password123".encode()).hexdigest()
        c.execute("INSERT INTO users (username, password, email, full_name) VALUES (?, ?, ?, ?)",
                  ("demo_user", hashed_password, "demo@example.com", "Demo User"))
        
        # Create sample profile
        user_id = c.lastrowid
        c.execute("INSERT INTO user_profiles (user_id, age, sensory_profile, triggers, coping_strategies, preferences) VALUES (?, ?, ?, ?, ?, ?)",
                  (user_id, 28, "Auditory Sensitivity, Visual Sensitivity", 
                   "Loud noises, bright lights, crowded spaces", 
                   "Noise-cancelling headphones, breathing exercises", 
                   "Quiet spaces, advance notice of changes"))
        
        # Create sample environment history
        sample_data = [
            (user_id, "Shopping Mall", 7.2, "High", "2023-10-15 14:30:00"),
            (user_id, "Park", 3.1, "Low", "2023-10-14 11:15:00"),
            (user_id, "Supermarket", 5.8, "Medium", "2023-10-13 16:45:00"),
            (user_id, "Restaurant", 6.5, "Medium", "2023-10-12 19:20:00"),
            (user_id, "Office", 4.2, "Low", "2023-10-11 09:30:00")
        ]
        
        c.executemany("INSERT INTO environment_history (user_id, environment_type, sensory_score, overload_risk, timestamp) VALUES (?, ?, ?, ?, ?)",
                     sample_data)
    
    conn.commit()
    conn.close()

def create_user(username, password, email, full_name):
    conn = sqlite3.connect('sensory_app.db')
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password, email, full_name) VALUES (?, ?, ?, ?)",
                  (username, hashed_password, email, full_name))
        conn.commit()
        user_id = c.lastrowid
        # Create empty profile
        c.execute("INSERT INTO user_profiles (user_id) VALUES (?)", (user_id,))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def verify_user(username, password):
    conn = sqlite3.connect('sensory_app.db')
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT id, username, full_name FROM users WHERE username = ? AND password = ?",
              (username, hashed_password))
    result = c.fetchone()
    conn.close()
    return result

def update_profile(user_id, age, sensory_profile, triggers, coping_strategies, preferences):
    conn = sqlite3.connect('sensory_app.db')
    c = conn.cursor()
    c.execute('''UPDATE user_profiles 
                 SET age=?, sensory_profile=?, triggers=?, coping_strategies=?, preferences=?
                 WHERE user_id=?''',
              (age, sensory_profile, triggers, coping_strategies, preferences, user_id))
    conn.commit()
    conn.close()

def get_profile(user_id):
    conn = sqlite3.connect('sensory_app.db')
    c = conn.cursor()
    c.execute('''SELECT u.username, u.email, u.full_name, 
                        p.age, p.sensory_profile, p.triggers, p.coping_strategies, p.preferences
                 FROM users u
                 JOIN user_profiles p ON u.id = p.user_id
                 WHERE u.id = ?''', (user_id,))
    result = c.fetchone()
    conn.close()
    return result

def save_environment_history(user_id, environment_type, sensory_score, overload_risk):
    conn = sqlite3.connect('sensory_app.db')
    c = conn.cursor()
    c.execute('''INSERT INTO environment_history (user_id, environment_type, sensory_score, overload_risk)
                 VALUES (?, ?, ?, ?)''',
              (user_id, environment_type, sensory_score, overload_risk))
    conn.commit()
    conn.close()

def get_environment_history(user_id):
    conn = sqlite3.connect('sensory_app.db')
    c = conn.cursor()
    c.execute('''SELECT environment_type, sensory_score, overload_risk, timestamp
                 FROM environment_history
                 WHERE user_id = ?
                 ORDER BY timestamp DESC LIMIT 20''', (user_id,))
    result = c.fetchall()
    conn.close()
    return result

def save_chat_message(user_id, message, is_user):
    conn = sqlite3.connect('sensory_app.db')
    c = conn.cursor()
    c.execute('''INSERT INTO chat_history (user_id, message, is_user)
                 VALUES (?, ?, ?)''',
              (user_id, message, is_user))
    conn.commit()
    conn.close()

def get_chat_history(user_id, limit=20):
    conn = sqlite3.connect('sensory_app.db')
    c = conn.cursor()
    c.execute('''SELECT message, is_user, timestamp
                 FROM chat_history
                 WHERE user_id = ?
                 ORDER BY timestamp DESC LIMIT ?''', (user_id, limit))
    result = c.fetchall()
    conn.close()
    return result
# Initialize database
init_db()

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.full_name = None

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'environment_data' not in st.session_state:
    st.session_state.environment_data = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# Login/Register functions
def show_login_form():
    st.markdown('<h2 class="main-header"> Sensory Overload Navigator</h2>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 2rem; color: var(--text-light);"><p>An AI-powered tool to help neurodiverse individuals navigate sensory-rich environments with confidence.</p></div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            st.subheader("Login to Your Account")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login", use_container_width=True)
            
            if login_button:
                if username and password:
                    user = verify_user(username, password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user[0]
                        st.session_state.username = user[1]
                        st.session_state.full_name = user[2]
                        st.success("Logged in successfully!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")
    
    with tab2:
        with st.form("register_form"):
            st.subheader("Create a New Account")
            new_username = st.text_input("Choose a Username")
            new_password = st.text_input("Choose a Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            email = st.text_input("Email Address")
            full_name = st.text_input("Full Name")
            register_button = st.form_submit_button("Register", use_container_width=True)
            
            if register_button:
                if not new_username or not new_password or not confirm_password:
                    st.error("Please fill in all required fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    if create_user(new_username, new_password, email, full_name):
                        st.success("Account created successfully! Please log in.")
                    else:
                        st.error("Username already exists")

def show_main_app():
    # Sidebar for user info and navigation
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.full_name or st.session_state.username}!")
        
        menu = st.selectbox("Navigation", ["Environment Analysis", "Sensory Assistant", "Profile Settings", "History", "About"])
        
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.full_name = None
            st.session_state.chat_messages = []
            st.rerun()
    
    # Main content based on menu selection
    if menu == "Environment Analysis":
        show_environment_analysis()
    elif menu == "Sensory Assistant":
        show_sensory_assistant()
    elif menu == "Profile Settings":
        show_profile_settings()
    elif menu == "History":
        show_history()
    elif menu == "About":
        show_about()

def show_environment_analysis():
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Environment Analysis</h3>', unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-header">Environment Settings</h4>', unsafe_allow_html=True)
        
        environment_type = st.selectbox(
            "Select Environment Type",
            ["Shopping Mall", "Supermarket", "Public Transport", "Restaurant/Cafe", 
             "Park", "Office", "Classroom", "Gym", "Concert", "Library", "Airport", "Hospital", "Custom"]
        )
        
        # Environment parameters with better defaults
        noise_level = st.slider("Noise Level (dB)", 30, 120, 65, 
                               help="Estimated noise level in decibels")
        crowd_density = st.slider("Crowd Density", 0, 100, 40,
                                 help="How crowded the environment is")
        lighting_conditions = st.slider("Lighting Intensity", 0, 100, 60,
                                       help="Brightness and intensity of lighting")
        visual_complexity = st.slider("Visual Complexity", 0, 100, 50,
                                     help="Amount of visual information and patterns")
        
        # Upload image for analysis
        uploaded_image = st.file_uploader("Upload Environment Image (Optional)", 
                                         type=['jpg', 'jpeg', 'png'],
                                         help="Upload an image of the environment for AI analysis")
        
        image = None
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Environment Image", use_column_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close feature-card
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-header">Personal Context</h4>', unsafe_allow_html=True)
        
        # Get user profile for context
        profile = get_profile(st.session_state.user_id)
        if profile:
            st.info(f"Your sensory profile: {profile[4] or 'Not specified'}")
            
            # Display user's known triggers
            if profile[5]:
                st.warning(f"Your known triggers: {profile[5]}")
            
            # Display coping strategies
            if profile[6]:
                st.success(f"Your coping strategies: {profile[6]}")
        
        # Additional context input
        additional_context = st.text_area(
            "Additional Context (Optional)",
            height=100,
            help="Add any additional information about how you're feeling or specific concerns"
        )
        
        # Analyze button
        if st.button("Analyze Environment", use_container_width=True):
            with st.spinner("Analyzing environment..."):
                # Get prediction from model
                score, risk, color, image_features = predictor.predict(
                    noise_level=noise_level,
                    crowd_density=crowd_density,
                    lighting_conditions=lighting_conditions,
                    visual_complexity=visual_complexity,
                    environment_type=environment_type,
                    image=image,
                    user_profile_text=additional_context
                )
                
                # Store results in session state
                st.session_state.analysis_done = True
                st.session_state.environment_data = {
                    "environment_type": environment_type,
                    "score": score,
                    "risk": risk,
                    "color": color,
                    "image_features": image_features,
                    "timestamp": datetime.datetime.now()
                }
                
                # Save to history
                save_environment_history(
                    st.session_state.user_id,
                    environment_type,
                    score,
                    risk
                )
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close feature-card
    
    # Display results if analysis is done
    if st.session_state.analysis_done and st.session_state.environment_data:
        data = st.session_state.environment_data
        
        st.markdown("---")
        st.markdown('<h3 class="section-header">Analysis Results</h3>', unsafe_allow_html=True)
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f'''
            <div class="metric-box">
                <h3>{data['score']:.1f}/10</h3>
                <p>Sensory Score</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-box">
                <h3 style="color: {data['color']};">{data['risk']}</h3>
                <p>Overload Risk</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-box">
                <h3>{data['environment_type']}</h3>
                <p>Environment Type</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Recommendations based on risk level
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-header">Recommendations</h4>', unsafe_allow_html=True)
        
        if data['risk'] == "Low":
            st.success("""
            ✅ **This environment should be comfortable for you.**  
            - You can likely spend extended time here without issues
            - Minimal sensory challenges expected
            - Standard coping strategies should be sufficient
            """)
        elif data['risk'] == "Medium":
            st.warning("""
            ⚠️ **This environment may present some challenges.**  
            - Consider limiting your time here
            - Use your preferred coping strategies
            - Be mindful of your sensory limits
            - Take breaks if needed
            """)
        else:
            st.error("""
            ❌ **This environment may be highly challenging.**  
            - Strongly consider avoiding or limiting exposure
            - Ensure you have all necessary coping tools
            - Plan an exit strategy
            - Consider going with support if necessary
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close recommendation-box
        
        # Additional insights from image analysis if available
        if uploaded_image and len(data['image_features']) > 0:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown('<h4 class="section-header">Image Analysis Insights</h4>', unsafe_allow_html=True)
            
            # Create a small bar chart for image features
            features = ['Crowd', 'Complexity', 'Lighting', 'Noise']
            values = data['image_features'] * 100  # Convert to percentage
            
            fig = go.Figure(data=[
                go.Bar(x=features, y=values, marker_color=['#6a0dad', '#5a0cad', '#4a0bad', '#3a0aad'])
            ])
            fig.update_layout(
                title="Environment Features Detected",
                yaxis_title="Intensity (%)",
                yaxis_range=[0, 100],
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close feature-card
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close tab-container

def show_sensory_assistant():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .sub-header {
        color: #4f8bf9;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4f8bf9;
        margin-bottom: 1rem;
    }
    .chat-container {
        background-color: #011017;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        height: 400px;
        overflow-y: auto;
        border: 1px solid #e6e6e6;
    }
    .user-message {
        background-color: #4f8bf9;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 15px 15px 0 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        text-align: right;
    }
    .bot-message {
        background-color: #070111;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 15px 15px 15px 0;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .suggestion-button {
        background-color: #f0f2f6;
        border: 1px solid #4f8bf9;
        color: #4f8bf9;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        width: 100%;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
    }
    .suggestion-button:hover {
        background-color: #4f8bf9;
        color: white;
    }
    .stChatInput > div > div > input {
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Sensory Assistant</h3>', unsafe_allow_html=True)
    
    # Initialize chat messages if not exists
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat container
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat messages in reverse order (newest at bottom)
        for msg in st.session_state.chat_messages:
            if msg['is_user']:
                st.markdown(f'<div class="user-message">{msg["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">{msg["message"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input at the bottom
    user_input = st.chat_input("Ask about sensory environments or coping strategies...")
    
    if user_input:
        # Add user message to chat
        st.session_state.chat_messages.append({"message": user_input, "is_user": True})
        save_chat_message(st.session_state.user_id, user_input, True)
        
        # Get bot response
        with st.spinner("Thinking..."):
            chatbot = SensoryChatbot()
            response = chatbot.get_response(user_input)
            
            # Add bot response to chat
            st.session_state.chat_messages.append({"message": response, "is_user": False})
            save_chat_message(st.session_state.user_id, response, False)
            
            # Rerun to update the chat display
            st.rerun()
    
    # Suggested questions in a compact layout
    st.markdown("### Try asking:")
    
    suggestions = [
        "What are some coping strategies for loud environments?",
        "How can I prepare for a visit to a shopping mall?",
        "What are calming techniques for sensory overload?",
        "How to create a sensory-friendly workspace?"
    ]
    
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                st.session_state.chat_messages.append({
                    "message": suggestion, 
                    "is_user": True
                })
                save_chat_message(st.session_state.user_id, suggestion, True)
                
                chatbot = SensoryChatbot()
                response = chatbot.get_response(suggestion)
                st.session_state.chat_messages.append({"message": response, "is_user": False})
                save_chat_message(st.session_state.user_id, response, False)
                st.rerun()



def show_profile_settings():
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Profile Settings</h3>', unsafe_allow_html=True)
    
    # Get current profile
    profile = get_profile(st.session_state.user_id)
    
    if profile:
        with st.form("profile_form"):
            st.markdown('<div class="profile-section">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                username = st.text_input("Username", value=profile[0], disabled=True)
                full_name = st.text_input("Full Name", value=profile[2] or "")
                email = st.text_input("Email", value=profile[1] or "")
                age = st.number_input("Age", min_value=1, max_value=120, value=profile[3] or 25)
            
            with col2:
                sensory_profile = st.text_area(
                    "Sensory Profile", 
                    value=profile[4] or "",
                    placeholder="Describe your sensory sensitivities (e.g., auditory sensitivity, visual sensitivity, tactile issues)"
                )
                
                triggers = st.text_area(
                    "Known Triggers", 
                    value=profile[5] or "",
                    placeholder="List environments or stimuli that typically cause sensory overload"
                )
                
                coping_strategies = st.text_area(
                    "Coping Strategies", 
                    value=profile[6] or "",
                    placeholder="What strategies help you manage sensory challenges?"
                )
                
                preferences = st.text_area(
                    "Preferences", 
                    value=profile[7] or "",
                    placeholder="Any preferences or accommodations that help you"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close profile-section
            
            if st.form_submit_button("Update Profile", use_container_width=True):
                update_profile(
                    st.session_state.user_id,
                    age,
                    sensory_profile,
                    triggers,
                    coping_strategies,
                    preferences
                )
                st.success("Profile updated successfully!")
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close tab-container

def show_history():
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Environment History</h3>', unsafe_allow_html=True)
    
    # Get user's environment history
    history = get_environment_history(st.session_state.user_id)
    
    if history:
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(history, columns=["Environment", "Score", "Risk", "Timestamp"])
        
        # Convert Timestamp to datetime
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Fix the TypeError by ensuring we're working with numeric data
            if not df.empty and "Score" in df.columns:
                # Convert Score to numeric, handling any conversion errors
                df["Score"] = pd.to_numeric(df["Score"], errors='coerce')
                # Drop any NaN values that resulted from conversion errors
                df_clean = df.dropna(subset=["Score"])
                if not df_clean.empty:
                    avg_score = df_clean["Score"].mean()
                    risk_level = "Low" if avg_score < 3.5 else "Medium" if avg_score < 6.5 else "High"
                    color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
                    st.markdown(f'''
                    <div class="metric-box">
                        <h3>{avg_score:.1f}/10</h3>
                        <p>Average Score</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="metric-box">
                        <h3>N/A</h3>
                        <p>Average Score</p>
                    </div>
                    ''', unsafe_allow_html=True)
        
        with col2:
            if not df.empty and "Risk" in df.columns:
                risk_counts = df["Risk"].value_counts()
                most_common_risk = risk_counts.index[0] if not risk_counts.empty else "N/A"
                risk_color = "green" if most_common_risk == "Low" else "orange" if most_common_risk == "Medium" else "red"
                st.markdown(f'''
                <div class="metric-box">
                    <h3 style="color: {risk_color};">{most_common_risk}</h3>
                    <p>Most Common Risk</p>
                </div>
                ''', unsafe_allow_html=True)
        
        with col3:
            if not df.empty:
                total_assessments = len(df)
                st.markdown(f'''
                <div class="metric-box">
                    <h3>{total_assessments}</h3>
                    <p>Total Assessments</p>
                </div>
                ''', unsafe_allow_html=True)
        
        # Display history table
        st.markdown('<h4 class="section-header">Recent Assessments</h4>', unsafe_allow_html=True)
        
        # Format the DataFrame for display
        display_df = df.copy()
        display_df["Timestamp"] = display_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        
        # Add color coding for risk levels
        def color_risk(val):
            if val == "High":
                color = "red"
            elif val == "Medium":
                color = "orange"
            else:
                color = "green"
            return f"color: {color}; font-weight: bold;"
        
        styled_df = display_df.style.applymap(color_risk, subset=["Risk"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Display charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4 class="section-header">Score Trend</h4>', unsafe_allow_html=True)
            if not df.empty and "Score" in df.columns and "Timestamp" in df.columns:
                fig = px.line(df, x="Timestamp", y="Score", title="Sensory Score Over Time",
                             labels={"Score": "Sensory Score", "Timestamp": "Date"})
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h4 class="section-header">Risk Distribution</h4>', unsafe_allow_html=True)
            if not df.empty and "Risk" in df.columns:
                risk_counts = df["Risk"].value_counts()
                fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                            title="Risk Level Distribution",
                            color=risk_counts.index,
                            color_discrete_map={'Low':'green', 'Medium':'orange', 'High':'red'})
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No environment history yet. Start by analyzing some environments!")
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close tab-container

def show_about():
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">About Sensory Overload Navigator</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
    <p>The <strong>Sensory Overload Navigator</strong> is an AI-powered tool designed to help neurodiverse individuals 
    (including those with autism, ADHD, sensory processing disorder, and other conditions) navigate sensory-rich 
    environments with greater confidence and comfort.</p>
    
    <p>This application uses machine learning and computer vision to analyze environments and predict potential 
    sensory challenges, providing personalized recommendations to help you prepare for and manage different situations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h4 class="section-header">Key Features</h4>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
        <h5>🏢 Environment Analysis</h5>
        <p>Get AI-powered assessments of various environments based on noise levels, crowd density, 
        lighting conditions, and visual complexity.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <h5>📊 Personalized Predictions</h5>
        <p>Receive tailored predictions about sensory overload risk based on your unique profile 
        and sensitivities.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
        <h5>🤖 AI Assistant</h5>
        <p>Chat with our sensory assistant for advice, coping strategies, and preparation tips 
        for various environments.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <h5>📈 History & Trends</h5>
        <p>Track your environment assessments over time and identify patterns in your sensory experiences.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h4 class="section-header">How It Works</h4>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
    <ol>
    <li><strong>Create your profile</strong> - Tell us about your sensory sensitivities, triggers, and coping strategies</li>
    <li><strong>Analyze environments</strong> - Use sliders to describe an environment or upload an image for AI analysis</li>
    <li><strong>Get predictions</strong> - Receive a sensory score and risk assessment</li>
    <li><strong>Plan accordingly</strong> - Use the recommendations to prepare for the environment</li>
    <li><strong>Track your history</strong> - Monitor patterns and improve your understanding of sensory challenges</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
    <p><strong>Disclaimer:</strong> This tool is designed to assist with planning and preparation but does not replace 
    professional medical advice. Always consult with healthcare providers for personalized guidance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close tab-container

def main():
    if not st.session_state.logged_in:
        show_login_form()
    else:
        show_main_app()

if __name__ == "__main__":
    main()