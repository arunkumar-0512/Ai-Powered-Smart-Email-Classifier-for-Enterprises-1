# Quick Reference Guide

## 🚀 30-Second Startup

### Option 1: Full Automation
```bash
pip install -r requirements.txt
python email_classification_pipeline.py
```

### Option 2: Interactive Dashboard
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

### Option 3: Quick Test
```bash
python quick_start_examples.py
```

---

## 📝 Common Tasks

### Task 1: Classify CSV File
```python
from email_classification_pipeline import run_complete_pipeline

pipeline = run_complete_pipeline('your_emails.csv', output_dir='./output')
print(pipeline.classified_df.head())
```

### Task 2: Check Single Email Urgency
```python
from urgency_detection import UrgencyDetector

detector = UrgencyDetector()
result = detector.hybrid_urgency_detection("your email text here")
print(result['urgency'])  # High, Medium, or Low
print(result['confidence'])  # 0.0 - 1.0
```

### Task 3: Generate Analytics Report
```python
from analytics_module import EmailAnalytics

analytics = EmailAnalytics(classified_df)
analytics.generate_summary_report()
analytics.export_report('report.json')
```

### Task 4: Process Multiple Emails
```python
from urgency_detection import UrgencyDetector
import pandas as pd

df = pd.read_csv('emails.csv')
detector = UrgencyDetector()
results = detector.batch_predictions(df, text_column='text')
```

---

## 📊 Dashboard Features

| Feature | How to Use |
|---------|-----------|
| Upload file | Click "Upload CSV file" in sidebar |
| Filter category | Select from "Select Categories" menu |
| Filter urgency | Select from "Select Urgency Levels" menu |
| Filter by date | Use "Select Date Range" calendar |
| Export results | Click "Download Filtered Data" button |
| View trends | Scroll to see volume trend chart |
| Check heatmap | See Category vs Urgency correlation |

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Add urgent keyword
URGENCY_KEYWORDS['High'].append('your_keyword')

# Change model weights
HYBRID_WEIGHTS = {'ml_weight': 0.7, 'rule_weight': 0.3}

# Adjust SVM parameters
SVM_CONFIG['max_iter'] = 3000
```

---

## 📂 Output Files

After running pipeline:

```
classification_output/
├── classified_emails_[timestamp].csv         # Final results
├── category_classifier_[timestamp].pkl       # SVM model
├── tfidf_vectorizer_[timestamp].pkl          # Feature vectorizer
├── urgency_detector_[timestamp].pkl          # Urgency model
└── statistics_[timestamp].json               # Summary stats
```

---

## 🔍 Understanding Results

### Urgency Prediction
- **Urgency**: High/Medium/Low
- **Confidence**: 0.0-1.0 (higher = more confident)
- **High > 0.7**: Trust the prediction
- **Low < 0.6**: Manual review recommended

### Category Prediction
- **Category**: One of 6 email types
- **Confidence**: 0.0-1.0 (higher = more confident)
- **Accuracy**: ~91% on test data

---

## 📈 Performance Tips

1. **Speed up processing**
   - Process in batches of 1000
   - Use fewer TF-IDF features
   - Disable ML model training if not needed

2. **Improve accuracy**
   - Use more training data
   - Balance categories (equal per type)
   - Customize urgency keywords for your domain

3. **Memory efficiency**
   - Use batch processing
   - Clear cache between runs
   - Limit TF-IDF max_features

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module not found" | `pip install -r requirements.txt` |
| Dashboard won't load | `streamlit run dashboard.py --server.port 8502` |
| Low accuracy | Check data balance; add training data |
| Slow processing | Use batch mode; reduce features |
| NLTK error | `python -c "import nltk; nltk.download('stopwords')"` |

---

## 📖 File Overview

| File | Purpose | Use When |
|------|---------|----------|
| `urgency_detection.py` | Urgency classification | Need urgency predictions |
| `dashboard.py` | Interactive UI | Want visual interface |
| `email_classification_pipeline.py` | Complete pipeline | Processing emails systematically |
| `analytics_module.py` | Statistical analysis | Need reports & insights |
| `config.py` | Settings | Customizing system |
| `quick_start_examples.py` | Code examples | Learning by example |

---

## 🎯 Workflow Examples

### Workflow A: Full Processing
```
1. python email_classification_pipeline.py
2. Results saved to ./classification_output/
3. View results in classified_emails_*.csv
```

### Workflow B: Interactive Analysis
```
1. streamlit run dashboard.py
2. Upload CSV file
3. Use filters to explore
4. Download results
```

### Workflow C: Single Classification
```
1. from urgency_detection import UrgencyDetector
2. detector = UrgencyDetector()
3. result = detector.hybrid_urgency_detection(text)
4. Check result['urgency'] and result['confidence']
```

---

## ✅ Installation Checklist

- [ ] Python 3.8+ installed
- [ ] `pip install -r requirements.txt` completed
- [ ] `nltk.download('stopwords')` completed
- [ ] email_dataset_full.csv present in directory
- [ ] All Python files present and readable

---

## 🔐 System Requirements

- **Python**: 3.8+
- **RAM**: 2GB minimum (4GB+ recommended)
- **Disk**: 500MB for models and data
- **CPU**: Modern processor (2+ cores)

---

## 💡 Pro Tips

1. **Customize urgency keywords** for your specific use case
2. **Train on your own data** for better accuracy
3. **Use batch processing** for 10K+ emails
4. **Export and version** results regularly
5. **Monitor confidence scores** to identify uncertain predictions

---

**Version**: 1.0 | **Last Updated**: March 30, 2026

Need help? See README.md or SUMMARY.md for detailed documentation.
