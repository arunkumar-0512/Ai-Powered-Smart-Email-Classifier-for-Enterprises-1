# -*- coding: utf-8 -*-
"""
Quick Start Examples
Demonstrates various ways to use the email classification system
"""

import pandas as pd
from email_classification_pipeline import EmailClassificationPipeline, run_complete_pipeline
from urgency_detection import UrgencyDetector
from analytics_module import EmailAnalytics
from pathlib import Path

print("="*70)
print("EMAIL CLASSIFICATION SYSTEM - QUICK START EXAMPLES")
print("="*70)

# ============================================================================
# EXAMPLE 1: Complete Pipeline Execution
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 1: Complete Pipeline Execution")
print("="*70)

def example_complete_pipeline():
    """
    Run the complete email classification pipeline
    This includes preprocessing, urgency detection, category classification, and analytics
    """
    print("\nRunning complete pipeline...")
    
    # Check if data file exists
    data_file = "email_dataset_full.csv"
    if not Path(data_file).exists():
        print(f"⚠️ Data file '{data_file}' not found")
        print("Please ensure the CSV file is in the current directory")
        return
    
    # Run complete pipeline
    pipeline = run_complete_pipeline(
        data_file=data_file,
        output_dir="./classification_output",
        train_urgency_ml=True
    )
    
    if pipeline:
        # Show sample results
        print("\nSample classified emails:")
        print(pipeline.classified_df.head(10))
        
        # Show statistics
        print("\nClassification Statistics:")
        print(f"Total emails: {len(pipeline.classified_df)}")
        if 'category' in pipeline.classified_df.columns:
            print(f"Categories: {pipeline.classified_df['category'].nunique()}")
        if 'urgency' in pipeline.classified_df.columns:
            print(f"Urgency distribution:\n{pipeline.classified_df['urgency'].value_counts()}")

# ============================================================================
# EXAMPLE 2: Urgency Detection Only
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 2: Urgency Detection Only")
print("="*70)

def example_urgency_detection():
    """
    Demonstrate urgency detection on individual emails
    """
    print("\nTesting urgency detection on sample emails:")
    
    detector = UrgencyDetector()
    
    test_emails = [
        "URGENT! The system is completely down. Need immediate assistance!!!",
        "Hi, just a friendly reminder about the meeting tomorrow",
        "Please fix this bug soon - it's affecting performance",
        "FYI - here's the status update for this week",
        "ERROR: Application not working properly. ASAP fix needed!!!"
    ]
    
    for i, email in enumerate(test_emails, 1):
        # Rule-based prediction
        rule_urgency, rule_conf = detector.rule_based_urgency(email)
        
        # Hybrid prediction (combines rule-based and ML)
        hybrid_result = detector.hybrid_urgency_detection(email)
        
        print(f"\n{i}. Email: {email[:50]}...")
        print(f"   Rule-Based: {rule_urgency} ({rule_conf:.2%})")
        print(f"   Hybrid: {hybrid_result['urgency']} ({hybrid_result['confidence']:.2%})")
        print(f"   Signals: {hybrid_result['signals']}")

# ============================================================================
# EXAMPLE 3: Analytics on Classified Data
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 3: Analytics on Classified Data")
print("="*70)

def example_analytics():
    """
    Generate analytics on classified email data
    """
    print("\nLoading classified data for analysis...")
    
    # Load your classified data (if available)
    classified_file = "email_dataset_full.csv"
    
    if Path(classified_file).exists():
        df = pd.read_csv(classified_file)
        
        # Initialize analytics
        analytics = EmailAnalytics(df)
        
        # Generate summary report
        print("\nGenerating analytics report...")
        analytics.generate_summary_report()
        
        # Export detailed report
        report_file = "analytics_report.json"
        analytics.export_report(report_file)
        print(f"\n✅ Report exported to {report_file}")
    else:
        print(f"⚠️ Classified data file not found")

# ============================================================================
# EXAMPLE 4: Batch Processing
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 4: Batch Processing")
print("="*70)

def example_batch_processing():
    """
    Process a batch of emails through the urgency detector
    """
    print("\nProcessing batch of emails...")
    
    # Create sample data
    sample_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'text': [
            "URGENT: System failure - need immediate help!!!",
            "Reminder: Team meeting tomorrow at 2 PM",
            "Please check this when you get a chance",
            "Question: How to reset password?",
            "CRITICAL BUG FOUND - Fix ASAP before production release"
        ]
    })
    
    print(f"\nOriginal data ({len(sample_df)} emails):")
    print(sample_df[['id', 'text']])
    
    # Process with urgency detector
    detector = UrgencyDetector()
    results = detector.batch_predictions(sample_df, text_column='text')
    
    # Combine results with original data
    processed_df = pd.concat([sample_df, results], axis=1)
    
    print(f"\nProcessed data with urgency predictions:")
    print(processed_df[['id', 'urgency', 'confidence']])
    
    return processed_df

# ============================================================================
# EXAMPLE 5: Pipeline with Custom Preprocessing
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 5: Pipeline with Custom Preprocessing")
print("="*70)

def example_custom_pipeline():
    """
    Create and customize the classification pipeline
    """
    print("\nSetting up custom pipeline...")
    
    data_file = "email_dataset_full.csv"
    
    if not Path(data_file).exists():
        print(f"⚠️ Data file not found: {data_file}")
        return
    
    # Initialize pipeline
    pipeline = EmailClassificationPipeline(data_file)
    
    if pipeline.df is None:
        print("Failed to load data")
        return
    
    print(f"Loaded {len(pipeline.df)} emails")
    print(f"Columns: {pipeline.df.columns.tolist()}")
    
    # Step 1: Preprocess
    print("\nStep 1: Preprocessing...")
    if pipeline.preprocess_all():
        print(f"✅ Preprocessed {len(pipeline.preprocessed_df)} emails")
    
    # Step 2: Urgency detection
    print("\nStep 2: Urgency Detection...")
    if pipeline.add_urgency_detection():
        print("✅ Urgency detection complete")
    
    # Step 3: Category classification
    print("\nStep 3: Category Classification...")
    if pipeline.add_category_classification():
        print("✅ Category classification complete")
    
    # Step 4: Create final dataset
    print("\nStep 4: Creating final dataset...")
    if pipeline.create_final_dataset():
        print("✅ Final dataset created")
        print(f"\nFinal dataset shape: {pipeline.classified_df.shape}")
        print(f"Columns: {pipeline.classified_df.columns.tolist()}")

# ============================================================================
# EXAMPLE 6: Single Email Classification
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 6: Single Email Classification")
print("="*70)

def example_single_email():
    """
    Classify a single email through the complete pipeline
    """
    print("\nClassifying a single email...")
    
    # Create sample email
    email = """
    Subject: URGENT - Website Down!!!
    
    Hi Team,
    
    The production website is completely down and users cannot access it.
    This is a critical issue and needs immediate attention.
    
    Please fix this ASAP!
    
    Error logs show: Connection timeout to database
    
    Thanks,
    John
    """
    
    # Initialize detector
    detector = UrgencyDetector()
    
    # Detect urgency
    result = detector.hybrid_urgency_detection(email)
    
    print("\nEmail Content:")
    print(email[:200] + "...")
    
    print("\nDetection Results:")
    print(f"  Urgency Level: {result['urgency']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Rule-Based: {result['rule_urgency']} ({result['rule_confidence']:.2%})")
    print(f"  Signals:")
    for key, value in result['signals'].items():
        if value > 0:
            print(f"    - {key}: {value}")

# ============================================================================
# EXAMPLE 7: Dashboard Quick Start
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 7: Dashboard Quick Start")
print("="*70)

def example_dashboard_info():
    """
    Provide information about launching the dashboard
    """
    print("\nTo launch the interactive dashboard:")
    print("\n1. Install Streamlit (already in requirements.txt):")
    print("   pip install streamlit")
    
    print("\n2. Run the dashboard:")
    print("   streamlit run dashboard.py")
    
    print("\n3. Dashboard will open at: http://localhost:8501")
    
    print("\nDashboard Features:")
    print("  - Upload CSV files")
    print("  - Real-time filtering")
    print("  - Interactive visualizations")
    print("  - Export data (CSV/JSON)")
    print("  - Analytics and trends")
    print("  - Email viewing table")

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    
    # Uncomment the example you want to run:
    
    # Example 1: Complete pipeline
    # example_complete_pipeline()
    
    # Example 2: Urgency detection
    example_urgency_detection()
    
    # Example 3: Analytics
    # example_analytics()
    
    # Example 4: Batch processing
    # batch_results = example_batch_processing()
    
    # Example 5: Custom pipeline
    # example_custom_pipeline()
    
    # Example 6: Single email
    # example_single_email()
    
    # Example 7: Dashboard info
    # example_dashboard_info()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
    print("\nNext steps:")
    print("1. Modify examples as needed for your use case")
    print("2. Run: python email_classification_pipeline.py")
    print("3. View dashboard: streamlit run dashboard.py")
    print("="*70)
