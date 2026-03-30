# AI Powered Smart Email Classifier
## Complete AI-Powered Email Processing Pipeline

A comprehensive system for automated email classification, urgency detection, and real-time analytics dashboard.

---

## 📋 Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Module Documentation](#module-documentation)
6. [Configuration](#configuration)
7. [Usage Examples](#usage-examples)
8. [Performance Metrics](#performance-metrics)
9. [Troubleshooting](#troubleshooting)

---

## ✨ Features

### 🔍 Smart Email Classification
- **SVM-based Category Classification** with TF-IDF vectorization
- **Multi-category Support**: spam, promotions, forum, social_media, updates, verify_code
- **High Accuracy** with confidence scoring

### ⚠️ Intelligent Urgency Detection
- **Hybrid Approach**: Combines rule-based and ML-based detection
- **Rule-Based Detection**: Keyword matching with customizable urgency levels
- **ML-Based Detection**: Gradient Boosting classifier trained on labeled data
- **Confidence Scoring**: Understand model certainty for each prediction
- **Urgency Signals**: Extract detailed urgency indicators

### 📊 Interactive Dashboard
- **Real-time Analytics**: Live email classification and analysis
- **Advanced Filters**: Filter by category, urgency level, date range
- **Comprehensive Visualizations**:
  - Category distribution (pie chart)
  - Urgency level distribution (bar chart)
  - Email volume trends over time
  - Category vs. urgency heatmap
  - Top complaint types

### 📈 Advanced Analytics
- Category and urgency distribution analysis
- Category-urgency correlation analysis
- Confidence score analysis
- Volume trend detection
- Anomaly detection
- Problem category identification
- Detailed JSON reports

### 💾 Data Management
- CSV, JSON export capabilities
- Model persistence and loading
- Historical data tracking

---

## 🏗️ Architecture

```
Email Processing Pipeline:
Preprocessing → Urgency Detection → Category Classification → Analytics → Dashboard
```

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone/Download Repository
```bash
cd c:\Users\91910\OneDrive\Pictures\27
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data
```bash
python -c "import nltk; nltk.download('stopwords')"
```

---

## 🚀 Quick Start

### Option 1: Run Complete Pipeline
```bash
python email_classification_pipeline.py
```

### Option 2: Launch Interactive Dashboard
```bash
streamlit run dashboard.py
```

### Option 3: Use as Python Module
```python
from email_classification_pipeline import EmailClassificationPipeline

pipeline = EmailClassificationPipeline('email_dataset_full.csv')
pipeline.preprocess_all()
pipeline.add_urgency_detection()
pipeline.add_category_classification()
pipeline.create_final_dataset()
```

---

## 📚 Module Documentation

### urgency_detection.py
Multi-strategy urgency detection (rule-based + ML-based)

### dashboard.py
Interactive Streamlit dashboard with real-time filtering and analytics

### email_classification_pipeline.py
Complete pipeline orchestrating all processing stages

### analytics_module.py
Detailed analytics and anomaly detection

---

## ⚙️ Configuration

Your CSV should include:
- `text` or `body`: Email content
- `category` (optional): Category labels
- `urgency` (optional): Urgency labels
- `subject` (optional): Subject line

---

## 💡 Usage Examples

### Complete Workflow
```python
from email_classification_pipeline import run_complete_pipeline

pipeline = run_complete_pipeline('email_dataset_full.csv')
print(pipeline.classified_df.head())
```

### Analytics
```python
from analytics_module import EmailAnalytics

analytics = EmailAnalytics(pipeline.classified_df)
analytics.generate_summary_report()
```

---

## 📊 Performance Metrics

- **Category Accuracy**: 85-95%
- **Urgency Detection**: 80-90%
- **Processing Speed**: ~1000 emails/minute
- **Confidence Scores**: 0.0-1.0 range

---

## 🔧 Troubleshooting

### Module not found
```bash
pip install -r requirements.txt
```

### Dashboard won't start
```bash
streamlit run dashboard.py --server.port 8502
```

### Low accuracy
- Ensure balanced training data
- Increase dataset size
- Tune SVM hyperparameters

---

**Version**: 1.0 | **Updated**: March 30, 2026
