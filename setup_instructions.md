# Installation & Setup Instructions

## 🎯 Get Started in 5 Minutes

### Step 1: Install Python Packages (1 minute)
```bash
cd c:\Users\91910\OneDrive\Pictures\27
pip install -r requirements.txt
```

### Step 2: Download NLTK Data (1 minute)
```bash
python -c "import nltk; nltk.download('stopwords')"
```

### Step 3: Run Pipeline (2 minutes)
```bash
python email_classification_pipeline.py
```

### Step 4: View Results (1 minute)
- Check `classification_output/` folder
- Open CSV file with Excel or Python

---

## 📦 What You Just Installed

### Core Libraries
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning
- **nltk**: Natural language processing
- **streamlit**: Interactive dashboards
- **plotly**: Interactive visualizations

### What's Working Now
✅ Email preprocessing
✅ Urgency detection (ML + Rules)
✅ Category classification (SVM)
✅ Dashboard (Streamlit)
✅ Analytics & reporting
✅ Model persistence

---

## 🚀 Run Any of These Options

### Option A: Full Automated Pipeline
```bash
python email_classification_pipeline.py
```
**What it does**: Preprocesses, classifies, detects urgency, saves results
**Time**: ~5-10 seconds for 1000 emails
**Output**: CSV file with classifications

### Option B: Interactive Dashboard
```bash
streamlit run dashboard.py
```
**What it does**: Opens web interface with visualizations
**Access**: http://localhost:8501
**Features**: Filters, charts, exports

### Option C: Quick Examples
```bash
python quick_start_examples.py
```
**What it does**: Shows 7 different usage examples
**Time**: ~2 seconds
**Output**: Console demonstrations

---

## 📊 Expected Output

### After Running Pipeline:
```
classification_output/
├── classified_emails_20260330_151234.csv
├── category_classifier_20260330_151234.pkl
├── tfidf_vectorizer_20260330_151234.pkl
├── urgency_detector_20260330_151234.pkl
└── statistics_20260330_151234.json
```

### CSV Contains Columns:
- `id`: Email ID
- `subject`: Email subject
- `text`: Email content
- `category`: Predicted category
- `urgency`: Predicted urgency level
- `urgency_confidence`: Confidence (0.0-1.0)
- `predicted_category`: SVM prediction
- `category_confidence`: Category confidence score

### Statistics JSON Contains:
- Total emails processed
- Category distribution
- Urgency distribution
- Processing timestamps

---

## ✅ Verification

### Check Installation
```bash
# Should return module names without errors
python -c "import pandas, numpy, sklearn, nltk, streamlit, plotly; print('✅ All packages installed!')"
```

### Test Urgency Detection
```python
python -c "
from urgency_detection import UrgencyDetector
detector = UrgencyDetector()
result = detector.hybrid_urgency_detection('URGENT: system down')
print(f'Urgency: {result[\"urgency\"]}')
print(f'Confidence: {result[\"confidence\"]:.2%}')
"
```

### Test Pipeline
```bash
# Run on sample data
python email_classification_pipeline.py
```

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
# Solution: Install all packages
pip install -r requirements.txt --upgrade
```

### Issue: "NLTK stopwords not found"
```bash
# Solution: Download NLTK data
python -c "import nltk; nltk.download('stopwords', download_dir='/usr/share/nltk_data')"
```

### Issue: "Port already in use" (Dashboard)
```bash
# Solution: Use different port
streamlit run dashboard.py --server.port 8502
```

### Issue: "Memory error" (Large dataset)
```python
# Solution: Process in batches
batch_size = 5000
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    # process batch
```

### Issue: "Low accuracy"
```
Solutions:
1. Check if data is balanced (similar count per category)
2. Ensure training data is clean
3. Check if stopwords are properly removed
4. Try adjusting SVM parameters in config.py
5. Collect more training data
```

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Detailed documentation |
| `SUMMARY.md` | Project overview |
| `QUICKSTART.md` | Quick reference |
| `setup_instructions.md` | This file |

---

## 💻 System Requirements

### Minimum
- Python 3.8+
- 2GB RAM
- 500MB disk space
- Windows/Mac/Linux

### Recommended
- Python 3.10+
- 4GB+ RAM
- 1GB disk space
- Modern CPU (2+ cores)

---

## 🎓 Learning Path

### Day 1: Setup
1. Install packages
2. Run pipeline
3. Check results

### Day 2: Explore
1. Run quick examples
2. View dashboard
3. Understand outputs

### Day 3: Customize
1. Modify config.py
2. Train on your data
3. Deploy system

### Day 4+: Production
1. Monitor accuracy
2. Fine-tune models
3. Scale processing

---

## 📞 Next Steps

### Immediate
- [ ] Install requirements
- [ ] Download NLTK data
- [ ] Run pipeline
- [ ] Verify outputs

### Short Term
- [ ] Explore dashboard
- [ ] Review documentation
- [ ] Run examples
- [ ] Customize config

### Medium Term
- [ ] Prepare own data
- [ ] Train models
- [ ] Evaluate accuracy
- [ ] Deploy system

### Long Term
- [ ] Monitor performance
- [ ] Collect feedback
- [ ] Improve models
- [ ] Scale infrastructure

---

## 🆘 Getting Help

1. **Check QUICKSTART.md** - Quick answers
2. **Check README.md** - Detailed docs
3. **Check SUMMARY.md** - Project overview
4. **Look at quick_start_examples.py** - Code examples
5. **Review config.py** - Customization options

---

## 🎉 You're Ready!

Your email classification system is ready to use. Choose one:

### Quick Test (30 seconds)
```bash
python quick_start_examples.py
```

### Full Demo (2 minutes)
```bash
python email_classification_pipeline.py
```

### Interactive Demo (5 minutes)
```bash
streamlit run dashboard.py
```

---

## 📈 What You Can Do Now

✅ Classify emails by category (6 types)
✅ Detect urgency levels (High/Medium/Low)
✅ Get confidence scores
✅ Visualize distributions
✅ Filter and search data
✅ Export results
✅ Generate reports
✅ Train on custom data
✅ Save and load models
✅ Build custom workflows

---

**Installation Complete! 🎉**

You now have a production-ready email classification and urgency detection system.

Start with: `python email_classification_pipeline.py`
