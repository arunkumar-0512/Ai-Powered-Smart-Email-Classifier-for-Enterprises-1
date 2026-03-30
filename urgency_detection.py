# -*- coding: utf-8 -*-
"""
Urgency Detection Module
Combines rule-based and ML-based approaches for comprehensive urgency classification
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class UrgencyDetector:
    """
    Multi-strategy urgency detection combining rule-based and ML approaches
    """
    
    def __init__(self):
        self.urgency_keywords = {
            'High': [
                'urgent', 'asap', 'immediately', 'critical', 'emergency',
                'not working', 'down', 'broken', 'error', 'failed',
                'urgent action', 'time sensitive', 'deadline', 'now',
                'right now', 'cannot', "can't", 'issue', 'problem',
                'alert', 'warning', '!!', 'help', 'please help',
                'stuck', 'crash', 'severe', 'outage', 'urgent issue'
            ],
            'Medium': [
                'soon', 'priority', 'important', 'important action',
                'pending', 'waiting', 'stuck', 'delay', 'slow',
                'performance', 'issue with', 'help please', 'need help',
                'question', 'inquiry', 'asap please', 'needed soon',
                'before', 'by', 'schedule'
            ],
            'Low': [
                'fyi', 'for your information', 'reminder', 'update',
                'notification', 'info', 'knowledge base', 'documentation',
                'general', 'feedback', 'suggestion', 'idea', 'nice to have',
                'when available', 'no rush'
            ]
        }
        
        self.ml_model = None
        self.vectorizer = None
        self.model_trained = False
        
    def extract_urgency_signals(self, text):
        """
        Extract urgency signals from text using linguistic patterns
        Returns a dictionary of detected signals
        """
        if pd.isna(text):
            text = ""
        
        text_lower = str(text).lower()
        signals = {}
        
        # Count urgency keywords by level
        for level, keywords in self.urgency_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            signals[f'{level}_keywords'] = count
        
        # Check for multiple punctuation (urgency indicator)
        signals['multiple_punctuation'] = len(re.findall(r'!{2,}|\?{2,}', text))
        
        # Check for all caps words (emphasis indicator)
        all_caps_words = len(re.findall(r'\b[A-Z]{3,}\b', text))
        signals['all_caps_words'] = all_caps_words
        
        # Check for common urgency patterns
        urgent_patterns = [
            r'urgent\s+(fix|help|issue|action|request)',
            r'(please|pls)\s+help',
            r'does\s+not\s+work',
            r'not\s+(working|functioning)',
            r'(error|issue|problem|bug)',
            r'deadline',
            r'time\s+sensitive'
        ]
        
        pattern_matches = sum(1 for pattern in urgent_patterns if re.search(pattern, text_lower))
        signals['urgency_patterns'] = pattern_matches
        
        # Check sentiment indicators (questions often indicate issues)
        signals['question_marks'] = text.count('?')
        signals['exclamation_marks'] = text.count('!')
        
        return signals
    
    def rule_based_urgency(self, text):
        """
        Rule-based urgency detection
        Returns urgency level and confidence score
        """
        if pd.isna(text):
            return 'Low', 0.3
        
        text_lower = str(text).lower()
        
        # Check high urgency keywords
        high_count = sum(1 for keyword in self.urgency_keywords['High'] if keyword in text_lower)
        
        # Check medium urgency keywords
        medium_count = sum(1 for keyword in self.urgency_keywords['Medium'] if keyword in text_lower)
        
        # Check low urgency keywords
        low_count = sum(1 for keyword in self.urgency_keywords['Low'] if keyword in text_lower)
        
        # Calculate confidence based on keyword matches
        total_matches = high_count + medium_count + low_count
        
        if high_count > 0:
            confidence = min(1.0, (high_count + 0.3 * medium_count) / max(total_matches, 1))
            return 'High', confidence
        elif medium_count > 0:
            confidence = min(1.0, (medium_count + 0.3 * high_count) / max(total_matches, 1))
            return 'Medium', confidence
        else:
            confidence = 0.3 if low_count == 0 else 0.6
            return 'Low', confidence
    
    def train_ml_model(self, texts, labels, test_size=0.2, random_state=42):
        """
        Train ML-based urgency classifier using multiple models
        """
        # Create pipeline with TF-IDF and Gradient Boosting
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        self.ml_model = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state
            ))
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        # Train the model
        self.ml_model.fit(X_train, y_train)
        self.model_trained = True
        
        # Evaluate
        y_pred = self.ml_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("=" * 60)
        print("ML Urgency Detector Training Results")
        print("=" * 60)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'test_data': X_test,
            'true_labels': y_test
        }
    
    def predict_urgency_ml(self, text):
        """
        Predict urgency using trained ML model
        Returns urgency level and confidence
        """
        if not self.model_trained or self.ml_model is None:
            return None, None
        
        try:
            prediction = self.ml_model.predict([text])[0]
            probabilities = self.ml_model.predict_proba([text])[0]
            confidence = max(probabilities)
            return prediction, confidence
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return None, None
    
    def hybrid_urgency_detection(self, text, ml_weight=0.6, rule_weight=0.4):
        """
        Hybrid approach: combine rule-based and ML predictions
        ml_weight: weight for ML prediction (0-1)
        rule_weight: weight for rule-based prediction (0-1)
        """
        # Get rule-based prediction
        rule_urgency, rule_confidence = self.rule_based_urgency(text)
        rule_score = {'High': 3, 'Medium': 2, 'Low': 1}[rule_urgency] * rule_confidence
        
        # Get ML prediction if model is trained
        if self.model_trained and self.ml_model is not None:
            ml_urgency, ml_confidence = self.predict_urgency_ml(text)
            ml_score = {'High': 3, 'Medium': 2, 'Low': 1}.get(ml_urgency, 1) * (ml_confidence or 0.5)
        else:
            ml_score = 0
            ml_confidence = 0
        
        # Combine scores
        combined_score = (rule_score * rule_weight + ml_score * ml_weight) / (rule_weight + ml_weight)
        
        # Determine final urgency level
        if combined_score >= 2.5:
            final_urgency = 'High'
        elif combined_score >= 1.7:
            final_urgency = 'Medium'
        else:
            final_urgency = 'Low'
        
        # Calculate overall confidence
        overall_confidence = (rule_confidence * rule_weight + (ml_confidence or 0.5) * ml_weight)
        
        return {
            'urgency': final_urgency,
            'confidence': min(1.0, overall_confidence),
            'rule_urgency': rule_urgency,
            'rule_confidence': rule_confidence,
            'ml_urgency': ml_urgency if self.model_trained else None,
            'ml_confidence': ml_confidence if self.model_trained else None,
            'combined_score': combined_score,
            'signals': self.extract_urgency_signals(text)
        }
    
    def save_model(self, filepath):
        """Save trained model to disk"""
        if self.ml_model is not None:
            with open(filepath, 'wb') as f:
                pickle.dump(self.ml_model, f)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from disk"""
        try:
            with open(filepath, 'rb') as f:
                self.ml_model = pickle.load(f)
            self.model_trained = True
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def batch_predictions(self, texts_df, text_column='text'):
        """
        Generate urgency predictions for a batch of emails
        """
        results = []
        
        for idx, row in texts_df.iterrows():
            text = row[text_column]
            prediction = self.hybrid_urgency_detection(text)
            
            results.append({
                'original_index': idx,
                'urgency': prediction['urgency'],
                'confidence': prediction['confidence'],
                'rule_urgency': prediction['rule_urgency'],
                'ml_urgency': prediction['ml_urgency'],
                'signals': prediction['signals']
            })
        
        return pd.DataFrame(results)


# Usage Example
if __name__ == "__main__":
    print("Urgency Detection Module Initialized")
    print("-" * 60)
    
    # Example usage
    detector = UrgencyDetector()
    
    # Test examples
    test_emails = [
        "URGENT! The system is down and we need immediate help!!!",
        "Hi, just wanted to let you know about the meeting reminder",
        "Please fix this ASAP - the app is not working properly",
        "FYI - here's the project update for this week"
    ]
    
    print("\nTesting Rule-Based Detection:")
    for email in test_emails:
        urgency, confidence = detector.rule_based_urgency(email)
        print(f"Text: {email[:50]}...")
        print(f"Urgency: {urgency}, Confidence: {confidence:.2f}\n")
