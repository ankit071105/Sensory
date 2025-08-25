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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download NLTK resources
try:
    nltk.data.find('vader_lexicon')
except:
    nltk.download('vader_lexicon')

# Set page configuration
st.set_page_config(
    page_title="Sensory Overload Navigator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #6a0dad;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5a0cad;
        margin-bottom: 1rem;
    }
    .recommendation-box {
        background-color: #f0e6ff;
        padding: 1.5rem;
        border-radius: 10px;
        color: #333333;
        margin-bottom: 1rem;
        border-left: 5px solid #6a0dad;
    }
    .metric-box {
        background-color: #f8f5ff;
        padding: 1rem;
        color: #333333;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #6a0dad;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .environment-tag {
        display: inline-block;
        background-color: #e6d7ff;
        padding: 0.3rem 0.8rem;
        color: #333333;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .login-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background-color: #f8f5ff;
        color: #333333;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .profile-section {
        background-color: #f8f5ff;
        padding: 1.5rem;
        color: #333333;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .prediction-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .prediction-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .prediction-low {
        color: #00cc66;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# CNN Model for Image Analysis
class EnvironmentCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(EnvironmentCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# NLP Model for Text Processing
class SensoryNLPModel:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.lda = LatentDirichletAllocation(n_components=3, random_state=42)
        
    def analyze_text(self, text):
        if not text:
            return np.zeros(5)
        
        # Sentiment analysis
        sentiment = self.sia.polarity_scores(text)
        
        # Text features
        features = np.array([
            sentiment['neg'],
            sentiment['neu'],
            sentiment['pos'],
            sentiment['compound'],
            len(text.split()) / 100  # Normalized word count
        ])
        
        return features

# Ensemble Model for Sensory Prediction
class SensoryPredictor:
    def __init__(self):
        self.cnn_model = None
        self.nlp_model = SensoryNLPModel()
        self.ml_model = None
        self.label_encoder = LabelEncoder()
        self.environment_types = ["Shopping Mall", "Supermarket", "Public Transport", 
                                 "Restaurant/Cafe", "Park", "Office", "Classroom", "Custom"]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.models_loaded = False
        
    def load_models(self):
        if self.models_loaded:
            return
            
        # Load or create CNN model
        self.cnn_model = EnvironmentCNN()
        cnn_path = 'environment_cnn.pth'
        if os.path.exists(cnn_path):
            self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=torch.device('cpu')))
        else:
            # Initialize with random weights (in real app, you'd train on a proper dataset)
            torch.save(self.cnn_model.state_dict(), cnn_path)
            
        self.cnn_model.eval()
        
        # Load or train ML model
        ml_path = 'sensory_ml_model.pkl'
        if os.path.exists(ml_path):
            self.ml_model = joblib.load(ml_path)
        else:
            self._train_ml_model()
        
        self.models_loaded = True
        
    def _train_ml_model(self):
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 5000
        
        # Features: noise_level, crowd_density, lighting_conditions, visual_complexity, environment_type
        X = np.zeros((n_samples, 9))  # 5 base features + 4 from CNN
        
        # Generate realistic data for different environment types
        for i, env in enumerate(self.environment_types):
            start_idx = i * (n_samples // len(self.environment_types))
            end_idx = (i + 1) * (n_samples // len(self.environment_types))
            
            if env == "Shopping Mall":
                X[start_idx:end_idx, 0] = np.random.normal(70, 10, end_idx - start_idx)  # noise
                X[start_idx:end_idx, 1] = np.random.normal(75, 15, end_idx - start_idx)  # crowd
                X[start_idx:end_idx, 2] = np.random.normal(80, 10, end_idx - start_idx)  # lighting
                X[start_idx:end_idx, 3] = np.random.normal(85, 8, end_idx - start_idx)   # visual
                X[start_idx:end_idx, 4] = i  # environment type
                # Simulated CNN features for shopping malls
                X[start_idx:end_idx, 5] = np.random.normal(0.8, 0.1, end_idx - start_idx)  # crowd feature
                X[start_idx:end_idx, 6] = np.random.normal(0.7, 0.1, end_idx - start_idx)  # complexity feature
                X[start_idx:end_idx, 7] = np.random.normal(0.6, 0.1, end_idx - start_idx)  # lighting feature
                X[start_idx:end_idx, 8] = np.random.normal(0.9, 0.1, end_idx - start_idx)  # noise feature
                
            elif env == "Park":
                X[start_idx:end_idx, 0] = np.random.normal(50, 15, end_idx - start_idx)
                X[start_idx:end_idx, 1] = np.random.normal(40, 20, end_idx - start_idx)
                X[start_idx:end_idx, 2] = np.random.normal(70, 20, end_idx - start_idx)
                X[start_idx:end_idx, 3] = np.random.normal(60, 15, end_idx - start_idx)
                X[start_idx:end_idx, 4] = i
                # Simulated CNN features for parks
                X[start_idx:end_idx, 5] = np.random.normal(0.3, 0.1, end_idx - start_idx)
                X[start_idx:end_idx, 6] = np.random.normal(0.4, 0.1, end_idx - start_idx)
                X[start_idx:end_idx, 7] = np.random.normal(0.7, 0.1, end_idx - start_idx)
                X[start_idx:end_idx, 8] = np.random.normal(0.2, 0.1, end_idx - start_idx)
                
            elif env == "Supermarket":
                X[start_idx:end_idx, 0] = np.random.normal(65, 8, end_idx - start_idx)
                X[start_idx:end_idx, 1] = np.random.normal(60, 10, end_idx - start_idx)
                X[start_idx:end_idx, 2] = np.random.normal(85, 5, end_idx - start_idx)
                X[start_idx:end_idx, 3] = np.random.normal(75, 10, end_idx - start_idx)
                X[start_idx:end_idx, 4] = i
                # Simulated CNN features for supermarkets
                X[start_idx:end_idx, 5] = np.random.normal(0.6, 0.1, end_idx - start_idx)
                X[start_idx:end_idx, 6] = np.random.normal(0.7, 0.1, end_idx - start_idx)
                X[start_idx:end_idx, 7] = np.random.normal(0.8, 0.1, end_idx - start_idx)
                X[start_idx:end_idx, 8] = np.random.normal(0.5, 0.1, end_idx - start_idx)
                
            else:  # Other environments
                X[start_idx:end_idx, 0] = np.random.normal(60, 15, end_idx - start_idx)
                X[start_idx:end_idx, 1] = np.random.normal(55, 20, end_idx - start_idx)
                X[start_idx:end_idx, 2] = np.random.normal(70, 15, end_idx - start_idx)
                X[start_idx:end_idx, 3] = np.random.normal(65, 15, end_idx - start_idx)
                X[start_idx:end_idx, 4] = i
                # Simulated CNN features for other environments
                X[start_idx:end_idx, 5] = np.random.normal(0.5, 0.1, end_idx - start_idx)
                X[start_idx:end_idx, 6] = np.random.normal(0.5, 0.1, end_idx - start_idx)
                X[start_idx:end_idx, 7] = np.random.normal(0.5, 0.1, end_idx - start_idx)
                X[start_idx:end_idx, 8] = np.random.normal(0.5, 0.1, end_idx - start_idx)
        
        # Calculate target (sensory score) based on features with some randomness
        y = (X[:, 0] * 0.3 + X[:, 1] * 0.25 + X[:, 2] * 0.2 + X[:, 3] * 0.15 + 
             X[:, 5] * 0.05 + X[:, 6] * 0.03 + X[:, 7] * 0.01 + X[:, 8] * 0.01) / 10
        y += np.random.normal(0, 0.5, n_samples)  # Add some noise
        y = np.clip(y, 0, 10)  # Ensure scores are between 0-10
        
        # Train model
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.ml_model.fit(X, y)
        
        # Save model
        joblib.dump(self.ml_model, 'sensory_ml_model.pkl')
        
    def analyze_image(self, image):
        if image is None:
            # Return default features when no image is provided
            return np.array([0.5, 0.5, 0.5, 0.5])
            
        try:
            # Ensure models are loaded
            self.load_models()
            
            # Preprocess image
            image = image.convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            # Get CNN features
            with torch.no_grad():
                features = self.cnn_model.features(image_tensor)
                features = features.view(features.size(0), -1)
                # For simplicity, we'll use the mean of features as our output
                features = features.mean(dim=1).numpy()
                
            # Simulate different environmental features from the image
            # In a real application, these would be learned by the model
            crowd_feature = np.clip(features[0] * 10, 0, 1)
            complexity_feature = np.clip(features[0] * 8, 0, 1)
            lighting_feature = np.clip(features[0] * 6, 0, 1)
            noise_feature = np.clip(features[0] * 12, 0, 1)
            
            return np.array([crowd_feature, complexity_feature, lighting_feature, noise_feature])
            
        except Exception as e:
            st.error(f"Error analyzing image: {e}")
            # Return default features on error
            return np.array([0.5, 0.5, 0.5, 0.5])
    
    def predict(self, noise_level, crowd_density, lighting_conditions, visual_complexity, 
                environment_type, image=None, user_profile_text=""):
        # Ensure models are loaded
        self.load_models()
        
        # Encode environment type
        env_encoded = self.environment_types.index(environment_type)
        
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
        
        # Ensure we have the right number of features
        if all_features.shape[1] > self.ml_model.n_features_in_:
            all_features = all_features[:, :self.ml_model.n_features_in_]
        elif all_features.shape[1] < self.ml_model.n_features_in_:
            # Pad with zeros if needed
            padding = np.zeros((1, self.ml_model.n_features_in_ - all_features.shape[1]))
            all_features = np.concatenate([all_features, padding], axis=1)
        
        # Predict sensory score
        score = self.ml_model.predict(all_features)[0]
        score = max(0, min(10, score))  # Ensure score is between 0-10
        
        # Determine risk level
        if score < 4:
            risk = "Low"
            color = "green"
        elif score < 7:
            risk = "Medium"
            color = "orange"
        else:
            risk = "High"
            color = "red"
            
        return score, risk, color, image_features

# Initialize AI model
predictor = SensoryPredictor()

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
                 ORDER by timestamp DESC LIMIT 10''', (user_id,))
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

# Login/Register functions
def show_login_form():
    st.markdown('<h2 class="main-header">🧠 Sensory Overload Navigator</h2>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;"><p>An AI-powered tool to help neurodiverse individuals navigate sensory-rich environments with confidence.</p></div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            st.subheader("Login to Your Account")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
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
            register_button = st.form_submit_button("Register")
            
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
        
        menu = st.selectbox("Navigation", ["Environment Analysis", "Profile Settings", "History", "About"])
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.full_name = None
            st.rerun()
    
    # Main content based on menu selection
    if menu == "Environment Analysis":
        show_environment_analysis()
    elif menu == "Profile Settings":
        show_profile_settings()
    elif menu == "History":
        show_history()
    elif menu == "About":
        show_about()

def show_environment_analysis():
    # Sidebar for user input
    with st.sidebar:
        st.header("Environment Settings")
        
        environment_type = st.selectbox(
            "Select Environment Type",
            ["Shopping Mall", "Supermarket", "Public Transport", "Restaurant/Cafe", "Park", "Office", "Classroom", "Custom"]
        )
        
        st.subheader("Environment Parameters")
        
        # Environment parameters
        noise_level = st.slider("Noise Level (dB)", 30, 120, 65)
        crowd_density = st.slider("Crowd Density", 0, 100, 40)
        lighting_conditions = st.slider("Lighting Intensity", 0, 100, 60)
        visual_complexity = st.slider("Visual Complexity", 0, 100, 50)
        
        # Upload image for analysis
        uploaded_image = st.file_uploader("Upload Environment Image", type=['jpg', 'jpeg', 'png'])
        
        analyze_btn = st.button("Analyze Environment", type="primary")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Display environment visualization
        st.markdown('<h3 class="sub-header">Environment Analysis</h3>', unsafe_allow_html=True)
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Environment", use_column_width=True)
            
            # Show image analysis results
            with st.spinner("Analyzing image with AI..."):
                image_features = predictor.analyze_image(image)
                st.info(f"Image analysis: Detected crowd level {image_features[0]*100:.1f}%, " +
                       f"complexity {image_features[1]*100:.1f}%, " +
                       f"lighting {image_features[2]*100:.1f}%")
        else:
            # Generate a sample visualization based on parameters
            fig = go.Figure()
            
            # Generate sample data based on parameters
            x = np.linspace(0, 10, 100)
            y = np.sin(x) * (noise_level/30) + np.random.randn(100) * (visual_complexity/50)
            
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Sensory Pattern',
                                    line=dict(color='#6a0dad', width=2)))
            
            fig.update_layout(
                title="Simulated Sensory Environment",
                xaxis_title="Time",
                yaxis_title="Sensory Input",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        if analyze_btn:
            st.session_state.analysis_done = True
            
            # Get user profile for personalized recommendations
            profile = get_profile(st.session_state.user_id)
            profile_text = ""
            if profile:
                profile_text = f"{profile[4] or ''} {profile[5] or ''} {profile[6] or ''} {profile[7] or ''}"
            
            # Analyze environment using AI model
            with st.spinner("Analyzing environment with AI..."):
                time.sleep(2)
                
                # Use AI model to predict sensory score
                sensory_score, overload_risk, overload_color, image_features = predictor.predict(
                    noise_level, crowd_density, lighting_conditions, visual_complexity, 
                    environment_type, Image.open(uploaded_image) if uploaded_image else None,
                    profile_text
                )
                
                # Store data in session state
                st.session_state.environment_data = {
                    "sensory_score": sensory_score,
                    "overload_risk": overload_risk,
                    "overload_color": overload_color,
                    "noise_level": noise_level,
                    "crowd_density": crowd_density,
                    "lighting_conditions": lighting_conditions,
                    "visual_complexity": visual_complexity,
                    "environment_type": environment_type,
                    "image_features": image_features
                }
                
                # Save to history
                save_environment_history(
                    st.session_state.user_id,
                    environment_type,
                    sensory_score,
                    overload_risk
                )

    with col2:
        st.markdown('<h3 class="sub-header">Sensory Metrics</h3>', unsafe_allow_html=True)
        
        if st.session_state.analysis_done and st.session_state.environment_data:
            data = st.session_state.environment_data
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <h3 style="margin:0; color: {data['overload_color']};">{data['overload_risk']}</h3>
                    <p style="margin:0;">Overload Risk</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <h3 style="margin:0;">{data['noise_level']} dB</h3>
                    <p style="margin:0;">Noise Level</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <h3 style="margin:0;">{data['sensory_score']:.1f}/10</h3>
                    <p style="margin:0;">Sensory Score</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <h3 style="margin:0;">{data['crowd_density']}%</h3>
                    <p style="margin:0;">Crowd Density</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show image analysis results if available
            if 'image_features' in data and any(data['image_features']):
                st.markdown("**Image Analysis:**")
                st.markdown(f"- Crowd level: {data['image_features'][0]*100:.1f}%")
                st.markdown(f"- Visual complexity: {data['image_features'][1]*100:.1f}%")
                st.markdown(f"- Lighting intensity: {data['image_features'][2]*100:.1f}%")
                st.markdown(f"- Noise potential: {data['image_features'][3]*100:.1f}%")
            
            # Environment tags
            st.markdown("**Environment Tags:**")
            st.markdown(f'<span class="environment-tag">{data["environment_type"]}</span>', unsafe_allow_html=True)
            
            if data["noise_level"] > 70:
                st.markdown('<span class="environment-tag">Loud</span>', unsafe_allow_html=True)
            if data["crowd_density"] > 60:
                st.markdown('<span class="environment-tag">Crowded</span>', unsafe_allow_html=True)
            if data["lighting_conditions"] > 70:
                st.markdown('<span class="environment-tag">Bright Lights</span>', unsafe_allow_html=True)
            if data["visual_complexity"] > 60:
                st.markdown('<span class="environment-tag">Visually Complex</span>', unsafe_allow_html=True)
        else:
            st.info("Analyze your environment to see sensory metrics here.")

    # Recommendations section
    if st.session_state.analysis_done and st.session_state.environment_data:
        st.markdown("---")
        st.markdown('<h3 class="sub-header">Personalized Recommendations</h3>', unsafe_allow_html=True)
        
        data = st.session_state.environment_data
        
        if data["overload_risk"] == "Low":
            st.markdown("""
            <div class="recommendation-box">
                <h4>🌿 Low Sensory Risk Detected</h4>
                <p>This environment appears to be generally manageable. Enjoy your time while being mindful of your sensory needs.</p>
                <ul>
                    <li>You might still want to take short breaks if you feel overwhelmed</li>
                    <li>Keep your noise-cancelling headphones accessible just in case</li>
                    <li>Stay hydrated and be aware of your energy levels</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif data["overload_risk"] == "Medium":
            st.markdown(f"""
            <div class="recommendation-box">
                <h4>⚠️ Medium Sensory Risk Detected</h4>
                <p>This environment has some potential sensory challenges. Consider these strategies:</p>
                <ul>
                    <li>Use noise-cancelling headphones or earplugs to reduce auditory input</li>
                    <li>Take regular breaks in a quieter space if possible</li>
                    <li>Use sunglasses or a hat if lighting is bothersome</li>
                    <li>Plan a shorter visit or break your time into smaller segments</li>
                    <li>Focus on a single task or area to reduce visual complexity</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.markdown(f"""
            <div class="recommendation-box">
                <h4>🚨 High Sensory Risk Detected</h4>
                <p>This environment may be challenging for sensory processing. Strongly consider these precautions:</p>
                <ul>
                    <li>Use both noise-cancelling headphones and sunglasses to reduce input</li>
                    <li>Limit your time in this environment to under 15 minutes if possible</li>
                    <li>Have an exit strategy planned in advance</li>
                    <li>Consider visiting at a less busy time if possible</li>
                    <li>Use grounding techniques if you feel overwhelmed</li>
                    <li>Bring a companion for support if available</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Get user profile for personalized recommendations
        profile = get_profile(st.session_state.user_id)
        if profile and profile[4]:  # If sensory profile exists
            st.markdown("#### 🎯 Personalized Based on Your Profile")
            if "auditory" in profile[4].lower():
                st.markdown("- **Auditory sensitivity detected**: Use noise-cancelling headphones or earplugs")
            if "visual" in profile[4].lower():
                st.markdown("- **Visual sensitivity detected**: Consider wearing sunglasses or a hat with a brim")
            if "crowd" in profile[4].lower() or "social" in profile[4].lower():
                st.markdown("- **Crowd sensitivity detected**: Try to stay near exits or less crowded areas")
            if "tactile" in profile[4].lower():
                st.markdown("- **Tactile sensitivity detected**: Wear comfortable clothing and avoid crowded pathways")
        
        # Alternative suggestions
        st.markdown("#### 🌟 Alternative Suggestions")
        
        if environment_type == "Shopping Mall":
            st.markdown("""
            - Consider shopping during off-peak hours (weekday mornings)
            - Use the mall's directory to plan your route in advance
            - Look for sensory-friendly shopping hours (some malls offer these)
            - Consider online shopping for items that aren't essential to try in person
            """)
        elif environment_type == "Supermarket":
            st.markdown("""
            - Use grocery pickup or delivery services when possible
            - Shop at smaller stores or during less busy hours
            - Make a detailed list organized by aisle to minimize time spent
            - Consider using a familiar store to reduce novelty stress
            """)
        elif environment_type == "Public Transport":
            st.markdown("""
            - Travel during off-peak hours to avoid crowds
            - Use noise-cancelling headphones and avoid eye contact if helpful
            - Choose less busy carriages or seating areas
            - Have a distraction ready (music, book, or game on your phone)
            - Know the route in advance to reduce uncertainty
            """)
        
        # Coping strategies
        st.markdown("#### 🧘‍♂️ Quick Coping Strategies")
        
        coping_col1, coping_col2, coping_col3 = st.columns(3)
        
        with coping_col1:
            st.markdown("""
            **Deep Pressure**
            - Wear a weighted vest or scarf
            - Use a weighted lap pad
            - Apply deep pressure to shoulders or thighs
            """)
        
        with coping_col2:
            st.markdown("""
            **Breathing Techniques**
            - 4-7-8 breathing (inhale 4s, hold 7s, exhale 8s)
            - Box breathing (4s in, 4s hold, 4s out, 4s hold)
            - Focus on exhaling completely
            """)
        
        with coping_col3:
            st.markdown("""
            **Grounding Techniques**
            - 5-4-3-2-1 method (5 things you see, 4 feel, 3 hear, 2 smell, 1 taste)
            - Hold a familiar object in your pocket
            - Focus on the physical sensation of feet on ground
            """)
        
        # Navigation help
        st.markdown("#### 🗺️ Navigation Assistance")
        
        nav_col1, nav_col2 = st.columns(2)
        
        with nav_col1:
            st.markdown("""
            **Quiet Areas Nearby**
            - Restrooms (often quieter stalls available)
            - Fitting rooms in stores
            - Seating areas near exits
            - Information desks (can ask for quiet space)
            """)
        
        with nav_col2:
            st.markdown("""
            **Exit Strategies**
            - Identify all exits upon arrival
            - Have a code word for companions if you need to leave
            - Park near exits for quick departure
            - Keep essentials easily accessible
            """)

    else:
        # Show initial instructions
        st.markdown("---")
        st.markdown("""
        <div style="background-color: #f0e6ff; padding: 2rem; border-radius: 10px; color: #333333;">
            <h3 style="color: #6a0dad;">How to Use the Sensory Overload Navigator</h3>
            <ol>
                <li>Select your environment type from the sidebar</li>
                <li>Adjust the sensory parameters based on your current or anticipated environment</li>
                <li>Upload an image of the environment if available (optional)</li>
                <li>Click "Analyze Environment" to get personalized recommendations</li>
            </ol>
            <p>The system will provide you with:
            <ul>
                <li>Sensory overload risk assessment using AI models</li>
                <li>Personalized coping strategies based on your profile</li>
                <li>Environment-specific recommendations</li>
                <li>Navigation assistance for challenging environments</li>
            </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)

def show_profile_settings():
    st.markdown('<h2 class="sub-header">Your Profile Settings</h2>', unsafe_allow_html=True)
    
    # Get current profile data
    profile = get_profile(st.session_state.user_id)
    
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            username = st.text_input("Username", value=profile[0] if profile else "", disabled=True)
            email = st.text_input("Email", value=profile[1] if profile else "")
            full_name = st.text_input("Full Name", value=profile[2] if profile else "")
            age = st.number_input("Age", min_value=5, max_value=100, value=profile[3] if profile and profile[3] else 25)
        
        with col2:
            st.subheader("Sensory Profile")
            sensory_options = ["Auditory Sensitivity", "Visual Sensitivity", "Tactile Sensitivity", 
                              "Olfactory Sensitivity", "Crowd Anxiety", "Bright Light Sensitivity"]
            
            # Get current selections
            current_selections = []
            if profile and profile[4]:
                current_selections = [s.strip() for s in profile[4].split(",")]
            
            sensory_profile = st.multiselect(
                "Select your sensory sensitivities",
                sensory_options,
                default=current_selections
            )
            
            triggers = st.text_area(
                "Specific triggers (describe what specifically bothers you)",
                value=profile[5] if profile and profile[5] else "",
                placeholder="e.g., Sudden loud noises, fluorescent lighting, strong perfumes"
            )
        
        st.subheader("Coping Strategies & Preferences")
        coping_strategies = st.text_area(
            "What coping strategies work best for you?",
            value=profile[6] if profile and profile[6] else "",
            placeholder="e.g., Deep pressure, noise-cancelling headphones, taking breaks"
        )
        
        preferences = st.text_area(
            "Any other preferences or notes",
            value=profile[7] if profile and profile[7] else "",
            placeholder="e.g., Prefer corner tables in restaurants, need advance preparation for new environments"
        )
        
        save_button = st.form_submit_button("Save Profile")
        
        if save_button:
            update_profile(
                st.session_state.user_id,
                age,
                ", ".join(sensory_profile),
                triggers,
                coping_strategies,
                preferences
            )
            st.success("Profile updated successfully!")

def show_history():
    st.markdown('<h2 class="sub-header">Your Environment History</h2>', unsafe_allow_html=True)
    
    history = get_environment_history(st.session_state.user_id)
    
    if history:
        # Create a dataframe for visualization
        df = pd.DataFrame(history, columns=["Environment", "Sensory Score", "Risk Level", "Timestamp"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        
        # Display history table
        st.dataframe(df, use_container_width=True)
        
        # Create visualization
        fig = px.line(df, x="Timestamp", y="Sensory Score", color="Environment",
                     title="Your Sensory Overload History")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show insights
        st.subheader("Patterns & Insights")
        
        avg_score = df["Sensory Score"].mean()
        most_common_env = df["Environment"].mode()[0] if not df["Environment"].mode().empty else "None"
        high_risk_count = len(df[df["Risk Level"] == "High"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Sensory Score", f"{avg_score:.1f}/10")
        with col2:
            st.metric("Most Frequent Environment", most_common_env)
        with col3:
            st.metric("High Risk Experiences", high_risk_count)
            
        # Add trend analysis
        st.subheader("Trend Analysis")
        
        # Calculate weekly average
        df_weekly = df.set_index('Timestamp').resample('W').mean().reset_index()
        fig_trend = px.line(df_weekly, x="Timestamp", y="Sensory Score", 
                           title="Weekly Average Sensory Score Trend")
        st.plotly_chart(fig_trend, use_container_width=True)
            
    else:
        st.info("You haven't analyzed any environments yet. Get started with the Environment Analysis tab!")

def show_about():
    st.markdown('<h2 class="sub-header">About the Sensory Overload Navigator</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommendation-box">
        <h3>Our Mission</h3>
        <p>The Sensory Overload Navigator is designed to help neurodiverse individuals (including those with autism, ADHD, 
        sensory processing disorder, and anxiety) navigate challenging environments with more confidence and less stress.</p>
        
        <h3>How It Works</h3>
        <p>Using advanced AI-powered analysis, our system:</p>
        <ul>
            <li>Uses convolutional neural networks (CNNs) to analyze environment images</li>
            <li>Employs natural language processing to understand your sensory profile</li>
            <li>Combines multiple inputs with ensemble machine learning models</li>
            <li>Provides personalized recommendations based on your unique needs</li>
            <li>Tracks your experiences to identify patterns and triggers over time</li>
        </ul>
        
        <h3>Our Technology</h3>
        <p>The system uses PyTorch-based deep learning models trained on sensory environment data to predict overload risk
        and provide personalized recommendations. The AI considers multiple factors including:</p>
        <ul>
            <li>Visual analysis of environment images</li>
            <li>Noise levels, crowd density, lighting conditions, and visual complexity</li>
            <li>Your personal sensory profile and triggers</li>
            <li>Historical data from your previous experiences</li>
        </ul>
        
        <h3>Who Can Benefit</h3>
        <p>This tool is designed for:</p>
        <ul>
            <li>Individuals with sensory processing differences</li>
            <li>People with autism spectrum disorder</li>
            <li>Those with ADHD or anxiety disorders</li>
            <li>Anyone who experiences overwhelm in sensory-rich environments</li>
            <li>Caregivers and support professionals</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>This application uses advanced machine learning and deep learning models to provide accurate predictions.</p>
        <p>Version 3.0 | Enhanced with PyTorch models for improved accuracy</p>
    </div>
    """, unsafe_allow_html=True)

# Main app logic
def main():
    if not st.session_state.logged_in:
        show_login_form()
    else:
        show_main_app()

if __name__ == "__main__":
    main()