# -*- coding: utf-8 -*-
"""
Configuration Module
Centralized configuration for email classification system
"""

# ============================================================================
# URGENCY DETECTION CONFIGURATION
# ============================================================================

URGENCY_KEYWORDS = {
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

# Urgency prediction thresholds
URGENCY_THRESHOLDS = {
    'high_minimum': 2.5,  # Minimum composite score for High urgency
    'medium_minimum': 1.7,  # Minimum composite score for Medium urgency
}

# ML model weights for hybrid approach
HYBRID_WEIGHTS = {
    'ml_weight': 0.6,       # Weight for ML prediction (0-1)
    'rule_weight': 0.4,     # Weight for rule-based prediction (0-1)
}

# ============================================================================
# CATEGORY CLASSIFICATION CONFIGURATION
# ============================================================================

# Supported email categories
EMAIL_CATEGORIES = {
    'spam': 3,
    'promotions': 1,
    'forum': 0,
    'social_media': 2,
    'updates': 4,
    'verify_code': 5
}

# SVM Model configuration
SVM_CONFIG = {
    'random_state': 42,
    'max_iter': 2000,
    'probability': False,
}

# TF-IDF Vectorizer configuration
TFIDF_CONFIG = {
    'max_features': 5000,
    'min_df': 2,
    'max_df': 0.8,
    'ngram_range': (1, 2),
    'stop_words': 'english',
    'lowercase': True,
    'sublinear_tf': True,
}

# Train-test split
TRAIN_TEST_SPLIT = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True,
}

# ============================================================================
# ML URGENCY MODEL CONFIGURATION
# ============================================================================

# Gradient Boosting classifier for urgency
URGENCY_ML_CONFIG = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'random_state': 42,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
}

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

PREPROCESSING_CONFIG = {
    'remove_urls': True,
    'remove_emails': True,
    'remove_numbers': False,
    'remove_punctuation': True,
    'lowercase': True,
    'remove_extra_spaces': True,
    'stemming_enabled': False,
    'lemmatization_enabled': False,
}

# Email signature patterns to remove
SIGNATURE_PATTERNS = [
    r"regards,.*",
    r"thanks,.*",
    r"best regards,.*",
    r"sincerely,.*",
    r"kind regards,.*",
    r"cheers,.*",
    r"--.*",
    r"sent from.*",
    r"^[\s]*$",  # Empty lines
]

# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================

DASHBOARD_CONFIG = {
    'page_title': 'Email Classification Dashboard',
    'page_icon': '📧',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'max_rows_display': 100,
    'default_rows': 20,
}

# Chart colors
CATEGORY_COLORS = {
    'spam': '#EF553B',
    'promotions': '#EF553B',
    'forum': '#00CC96',
    'social_media': '#636EFA',
    'updates': '#AB63FA',
    'verify_code': '#FFA15A'
}

URGENCY_COLORS = {
    'High': '#EF553B',
    'Medium': '#FFA15A',
    'Low': '#00CC96',
}

# ============================================================================
# ANALYTICS CONFIGURATION
# ============================================================================

ANALYTICS_CONFIG = {
    'anomaly_missing_threshold': 0.05,  # Flag if > 5% missing
    'anomaly_imbalance_threshold': 0.9,  # Flag if one class > 90%
    'outlier_std_threshold': 3,  # Outliers beyond 3 std deviations
    'volume_trend_ma_window': 7,  # 7-day moving average
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

OUTPUT_CONFIG = {
    'output_directory': './classification_output',
    'create_directory': True,
    'timestamp_format': '%Y%m%d_%H%M%S',
    'save_models': True,
    'save_statistics': True,
    'save_report': True,
}

# CSV export columns (final output)
CSV_EXPORT_COLUMNS = [
    'id',
    'subject',
    'text',
    'category',
    'urgency',
    'urgency_confidence',
    'predicted_category',
    'category_confidence',
    'split'
]

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'enabled': True,
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'classification.log',
}

# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

PERFORMANCE_CONFIG = {
    'batch_size': 1000,  # Process emails in batches
    'num_workers': 4,  # Number of parallel workers
    'cache_enabled': True,
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_urgency_label(score):
    """Get urgency label based on composite score"""
    if score >= URGENCY_THRESHOLDS['high_minimum']:
        return 'High'
    elif score >= URGENCY_THRESHOLDS['medium_minimum']:
        return 'Medium'
    else:
        return 'Low'


def get_category_id(category_name):
    """Get category ID from category name"""
    return EMAIL_CATEGORIES.get(category_name.lower(), None)


def get_category_name(category_id):
    """Get category name from category ID"""
    for name, id in EMAIL_CATEGORIES.items():
        if id == category_id:
            return name
    return None


def get_urgency_color(urgency_level):
    """Get color for urgency level"""
    return URGENCY_COLORS.get(urgency_level, '#808080')


def get_category_color(category):
    """Get color for category"""
    return CATEGORY_COLORS.get(category.lower(), '#808080')


def validate_config():
    """Validate configuration consistency"""
    errors = []
    
    # Check weights sum to 1
    total_weight = HYBRID_WEIGHTS['ml_weight'] + HYBRID_WEIGHTS['rule_weight']
    if total_weight != 1.0:
        errors.append(f"Hybrid weights don't sum to 1.0: {total_weight}")
    
    # Check thresholds are logical
    if URGENCY_THRESHOLDS['high_minimum'] <= URGENCY_THRESHOLDS['medium_minimum']:
        errors.append("High urgency threshold should be > Medium urgency threshold")
    
    # Check TF-IDF config
    if TFIDF_CONFIG['min_df'] >= TFIDF_CONFIG['max_df']:
        errors.append("TF-IDF min_df should be < max_df")
    
    if errors:
        print("Configuration Validation Errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


# ============================================================================
# Print configuration when imported
# ============================================================================

if __name__ == "__main__":
    print("Email Classification System Configuration")
    print("=" * 60)
    
    print("\nUrgency Keywords:")
    for level, keywords in URGENCY_KEYWORDS.items():
        print(f"  {level}: {len(keywords)} keywords")
    
    print("\nCategories:")
    for cat, id in EMAIL_CATEGORIES.items():
        print(f"  {cat}: {id}")
    
    print("\nHybrid Model Weights:")
    print(f"  ML-based: {HYBRID_WEIGHTS['ml_weight']}")
    print(f"  Rule-based: {HYBRID_WEIGHTS['rule_weight']}")
    
    print("\nValidating configuration...")
    if validate_config():
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration has errors!")
