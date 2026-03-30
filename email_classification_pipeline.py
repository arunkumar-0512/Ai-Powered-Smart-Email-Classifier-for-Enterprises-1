# -*- coding: utf-8 -*-
"""
Complete Email Classification Pipeline
Integrates preprocessing, urgency detection, category classification, and visualization
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from pathlib import Path
from datetime import datetime
import json

from urgency_detection import UrgencyDetector

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class EmailClassificationPipeline:
    """
    Complete pipeline for email preprocessing, classification, and urgency detection
    """
    
    def __init__(self, data_file=None):
        """Initialize pipeline"""
        self.df = None
        self.preprocessed_df = None
        self.classified_df = None
        self.urgency_detector = UrgencyDetector()
        
        self.tfidf_vectorizer = None
        self.category_classifier = None
        self.urgency_classifier = None
        
        self.stop_words = set(stopwords.words("english"))
        
        if data_file:
            self.load_data(data_file)
    
    def load_data(self, filepath):
        """Load raw email data"""
        try:
            self.df = pd.read_csv(filepath)
            print(f"[OK] Loaded {len(self.df)} emails from {filepath}")
            print(f"Columns: {self.df.columns.tolist()}")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading data: {e}")
            return False
    
    def remove_signature(self, text):
        """Remove email signatures"""
        if pd.isna(text):
            return ""
        
        patterns = [
            r"regards,.*",
            r"thanks,.*",
            r"best regards,.*",
            r"sincerely,.*",
            r"kind regards,.*",
            r"cheers,.*",
            r"--.*"
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, "", str(text), flags=re.IGNORECASE | re.DOTALL)
        
        return text
    
    def normalize_text(self, text):
        """Normalize text (lowercase, remove special chars)"""
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Keep alphanumeric and spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        return " ".join(words)
    
    def preprocess_email(self, text):
        """Complete preprocessing pipeline"""
        text = self.remove_signature(text)
        text = self.normalize_text(text)
        text = self.remove_stopwords(text)
        return text
    
    def preprocess_all(self, text_column='text'):
        """Preprocess all emails in dataset"""
        print("\n" + "="*60)
        print("STAGE 1: DATA PREPROCESSING")
        print("="*60)
        
        if self.df is None:
            print("[ERROR] No data loaded")
            return False
        
        print(f"Preprocessing {len(self.df)} emails...")
        
        self.preprocessed_df = self.df.copy()
        
        # Find the text column (could be 'text', 'body', 'subject', etc.)
        if text_column not in self.preprocessed_df.columns:
            available_cols = [col for col in self.preprocessed_df.columns 
                            if col.lower() in ['text', 'body', 'content', 'message']]
            if available_cols:
                text_column = available_cols[0]
            else:
                print("[WARNING] No suitable text column found")
                return False
        
        self.preprocessed_df['cleaned_text'] = self.preprocessed_df[text_column].apply(
            self.preprocess_email
        )
        
        # Remove any rows with empty cleaned text
        self.preprocessed_df = self.preprocessed_df[self.preprocessed_df['cleaned_text'].str.len() > 0]
        
        print(f"[OK] Preprocessed {len(self.preprocessed_df)} emails")
        print(f"Sample cleaned text: {self.preprocessed_df['cleaned_text'].iloc[0][:100]}...")
        
        return True
    
    def add_urgency_detection(self, train_urgency_classifier=True):
        """Add urgency detection to pipeline"""
        print("\n" + "="*60)
        print("STAGE 2: URGENCY DETECTION")
        print("="*60)
        
        if self.preprocessed_df is None:
            print("[ERROR] Please preprocess data first")
            return False
        
        print("Extracting urgency signals and generating predictions...")
        
        results = []
        for idx, row in self.preprocessed_df.iterrows():
            text = row['cleaned_text']
            
            # Get hybrid prediction
            prediction = self.urgency_detector.hybrid_urgency_detection(text)
            
            results.append({
                'urgency': prediction['urgency'],
                'urgency_confidence': prediction['confidence'],
                'rule_urgency': prediction['rule_urgency'],
                'ml_urgency': prediction['ml_urgency'],
                'signals': str(prediction['signals'])
            })
        
        urgency_df = pd.DataFrame(results)
        self.preprocessed_df = pd.concat([self.preprocessed_df, urgency_df], axis=1)
        
        print(f"[OK] Urgency detection complete")
        urgency_counts = self.preprocessed_df['urgency'].value_counts()
        print(f"Urgency distribution:\n{urgency_counts}")
        
        # Train ML urgency classifier if requested
        if train_urgency_classifier and 'urgency' in self.df.columns:
            print("\nTraining ML-based urgency classifier...")
            
            # Get labeled data
            labeled_data = self.df[self.df['urgency'].notna()].copy()
            if len(labeled_data) > 0:
                labeled_data['cleaned_text'] = labeled_data.get('text', labeled_data.get('body', '')).apply(
                    self.preprocess_email
                )
                
                train_results = self.urgency_detector.train_ml_model(
                    texts=labeled_data['cleaned_text'].values,
                    labels=labeled_data['urgency'].values,
                    test_size=0.2
                )
        
        return True
    
    def add_category_classification(self):
        """Add category classification"""
        print("\n" + "="*60)
        print("STAGE 3: CATEGORY CLASSIFICATION (SVM)")
        print("="*60)
        
        if self.preprocessed_df is None:
            print("[ERROR] Please preprocess data first")
            return False
        
        # Ensure we have category labels
        if 'category' not in self.preprocessed_df.columns and 'category_id' in self.preprocessed_df.columns:
            # Map category_id to category if needed
            category_map = {
                0: 'forum',
                1: 'promotions',
                2: 'social_media',
                3: 'spam',
                4: 'updates',
                5: 'verify_code'
            }
            self.preprocessed_df['category'] = self.preprocessed_df['category_id'].map(category_map)
        
        print(f"Training SVM classifier with {len(self.preprocessed_df)} emails...")
        
        X = self.preprocessed_df['cleaned_text']
        y = self.preprocessed_df['category']
        
        # Train TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        X_tfidf = self.tfidf_vectorizer.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} emails")
        print(f"Test set: {X_test.shape[0]} emails")
        
        # Train SVM classifier
        self.category_classifier = LinearSVC(random_state=42, max_iter=2000)
        self.category_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.category_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n[OK] SVM Training Complete")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Add predictions to full dataset
        print("\nGenerating predictions on full dataset...")
        predictions = self.category_classifier.predict(X_tfidf)
        self.preprocessed_df['predicted_category'] = predictions
        
        # Calculate prediction confidence (distance to decision boundary)
        decision_distances = self.category_classifier.decision_function(X_tfidf)
        # Convert to confidence scores (0-1)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0.5, 1.0))
        confidences = scaler.fit_transform(np.abs(decision_distances).max(axis=1).reshape(-1, 1))
        self.preprocessed_df['category_confidence'] = confidences.flatten()
        
        return True
    
    def create_final_dataset(self):
        """Create final classified dataset"""
        print("\n" + "="*60)
        print("STAGE 4: CREATING FINAL CLASSIFIED DATASET")
        print("="*60)
        
        if self.preprocessed_df is None:
            print("[ERROR] Please complete preprocessing first")
            return False
        
        self.classified_df = self.preprocessed_df.copy()
        
        # Select key columns for output
        key_columns = []
        
        # Add ID if available
        if 'id' in self.classified_df.columns:
            key_columns.append('id')
        
        # Add original content
        text_cols = [col for col in self.classified_df.columns 
                    if col in ['subject', 'body', 'text', 'message']]
        key_columns.extend(text_cols[:1])  # Keep only one text column
        
        # Add classification results
        key_columns.extend(['category', 'urgency', 'urgency_confidence'])
        
        if 'predicted_category' in self.classified_df.columns:
            key_columns.append('predicted_category')
            key_columns.append('category_confidence')
        
        if 'split' in self.classified_df.columns:
            key_columns.append('split')
        
        # Keep only available columns
        key_columns = [col for col in key_columns if col in self.classified_df.columns]
        
        self.classified_df = self.classified_df[key_columns]
        
        print(f"[OK] Final dataset created with {len(self.classified_df)} emails")
        print(f"Columns: {self.classified_df.columns.tolist()}")
        
        return True
    
    def save_results(self, output_dir='./output'):
        """Save all results and models"""
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save classified dataset
        if self.classified_df is not None:
            output_file = output_path / f"classified_emails_{timestamp}.csv"
            self.classified_df.to_csv(output_file, index=False)
            print(f"[OK] Saved classified emails to {output_file}")
        
        # Save models
        if self.category_classifier is not None:
            model_file = output_path / f"category_classifier_{timestamp}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(self.category_classifier, f)
            print(f"[OK] Saved category classifier to {model_file}")
        
        if self.tfidf_vectorizer is not None:
            vectorizer_file = output_path / f"tfidf_vectorizer_{timestamp}.pkl"
            with open(vectorizer_file, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            print(f"[OK] Saved TF-IDF vectorizer to {vectorizer_file}")
        
        # Save urgency detector
        urgency_file = output_path / f"urgency_detector_{timestamp}.pkl"
        self.urgency_detector.save_model(str(urgency_file))
        
        # Save statistics
        if self.classified_df is not None:
            stats = {
                'total_emails': len(self.classified_df),
                'categories': self.classified_df['category'].value_counts().to_dict() if 'category' in self.classified_df.columns else {},
                'urgency_levels': self.classified_df['urgency'].value_counts().to_dict() if 'urgency' in self.classified_df.columns else {},
                'timestamp': timestamp
            }
            
            stats_file = output_path / f"statistics_{timestamp}.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"✅ Saved statistics to {stats_file}")
    
    def print_summary_statistics(self):
        """Print comprehensive summary statistics"""
        print("\n" + "="*60)
        print("FINAL SUMMARY STATISTICS")
        print("="*60)
        
        if self.classified_df is None:
            print("❌ No classified data available")
            return
        
        print(f"\n📊 Dataset Overview:")
        print(f"  Total Emails: {len(self.classified_df):,}")
        print(f"  Columns: {len(self.classified_df.columns)}")
        
        if 'category' in self.classified_df.columns:
            print(f"\n📂 Email Categories:")
            for cat, count in self.classified_df['category'].value_counts().items():
                pct = count / len(self.classified_df) * 100
                print(f"  - {cat}: {count:,} ({pct:.1f}%)")
        
        if 'urgency' in self.classified_df.columns:
            print(f"\n⚠️ Urgency Levels:")
            for level in ['High', 'Medium', 'Low']:
                count = len(self.classified_df[self.classified_df['urgency'] == level])
                pct = count / len(self.classified_df) * 100 if len(self.classified_df) > 0 else 0
                print(f"  - {level}: {count:,} ({pct:.1f}%)")
        
        if 'urgency_confidence' in self.classified_df.columns:
            avg_conf = self.classified_df['urgency_confidence'].mean()
            print(f"\n📈 Average Urgency Confidence: {avg_conf:.2%}")
        
        if 'category_confidence' in self.classified_df.columns:
            avg_conf = self.classified_df['category_confidence'].mean()
            print(f"📈 Average Category Confidence: {avg_conf:.2%}")


def run_complete_pipeline(data_file, output_dir='./output', train_urgency_ml=True):
    """Run complete email classification pipeline"""
    print("\n" + "="*70)
    print("EMAIL CLASSIFICATION PIPELINE")
    print("="*70)
    
    # Initialize pipeline
    pipeline = EmailClassificationPipeline(data_file)
    
    if pipeline.df is None:
        return None
    
    # Execute pipeline stages
    if not pipeline.preprocess_all():
        return None
    
    if not pipeline.add_urgency_detection(train_urgency_classifier=train_urgency_ml):
        return None
    
    if not pipeline.add_category_classification():
        return None
    
    if not pipeline.create_final_dataset():
        return None
    
    # Save results
    pipeline.save_results(output_dir)
    
    # Print summary
    pipeline.print_summary_statistics()
    
    return pipeline


# Usage Example
if __name__ == "__main__":
    print("Email Classification Pipeline v1.0")
    
    # Run on sample data
    data_file = "email_dataset_full.csv"
    
    if Path(data_file).exists():
        pipeline = run_complete_pipeline(
            data_file=data_file,
            output_dir="./classification_output"
        )
        
        if pipeline:
            print("\n[OK] Pipeline execution complete!")
            print("[INFO] To view dashboard, run: streamlit run dashboard.py")
    else:
        print(f"[ERROR] Data file not found: {data_file}")
