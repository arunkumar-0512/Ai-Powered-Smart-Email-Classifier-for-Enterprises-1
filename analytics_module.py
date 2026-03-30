# -*- coding: utf-8 -*-
"""
Advanced Analytics & Reporting Module
Detailed analytics, anomaly detection, and reporting capabilities
"""

import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class EmailAnalytics:
    """
    Comprehensive analytics for classified emails
    """
    
    def __init__(self, classified_df):
        """Initialize analytics with classified dataframe"""
        self.df = classified_df.copy()
        self.report = {}
        
        # Ensure date column exists
        if 'date' not in self.df.columns:
            np.random.seed(42)
            dates = [datetime.now() - timedelta(days=int(x)) 
                    for x in np.random.uniform(0, 365, len(self.df))]
            self.df['date'] = dates
        else:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
    
    def analyze_category_distribution(self):
        """Analyze email distribution across categories"""
        if 'category' not in self.df.columns:
            return {}
        
        analysis = {}
        category_counts = self.df['category'].value_counts()
        
        analysis['total_categories'] = len(category_counts)
        analysis['category_distribution'] = category_counts.to_dict()
        analysis['category_percentages'] = (category_counts / len(self.df) * 100).to_dict()
        analysis['most_common'] = category_counts.index[0] if len(category_counts) > 0 else None
        analysis['least_common'] = category_counts.index[-1] if len(category_counts) > 0 else None
        
        return analysis
    
    def analyze_urgency_distribution(self):
        """Analyze urgency level distribution"""
        if 'urgency' not in self.df.columns:
            return {}
        
        analysis = {}
        urgency_counts = self.df['urgency'].value_counts()
        
        analysis['high_urgency'] = len(self.df[self.df['urgency'] == 'High'])
        analysis['medium_urgency'] = len(self.df[self.df['urgency'] == 'Medium'])
        analysis['low_urgency'] = len(self.df[self.df['urgency'] == 'Low'])
        
        analysis['high_urgency_pct'] = analysis['high_urgency'] / len(self.df) * 100
        analysis['medium_urgency_pct'] = analysis['medium_urgency'] / len(self.df) * 100
        analysis['low_urgency_pct'] = analysis['low_urgency'] / len(self.df) * 100
        
        return analysis
    
    def analyze_category_urgency_correlation(self):
        """Analyze correlation between category and urgency"""
        if 'category' not in self.df.columns or 'urgency' not in self.df.columns:
            return {}
        
        analysis = {}
        
        # Create crosstab
        crosstab = pd.crosstab(self.df['category'], self.df['urgency'])
        
        for category in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == category]
            urgency_dist = cat_data['urgency'].value_counts()
            
            analysis[category] = {
                'total': len(cat_data),
                'high_pct': urgency_dist.get('High', 0) / len(cat_data) * 100,
                'medium_pct': urgency_dist.get('Medium', 0) / len(cat_data) * 100,
                'low_pct': urgency_dist.get('Low', 0) / len(cat_data) * 100,
            }
        
        return analysis
    
    def analyze_confidences(self):
        """Analyze confidence scores"""
        analysis = {}
        
        if 'urgency_confidence' in self.df.columns:
            confidences = self.df['urgency_confidence'].dropna()
            analysis['urgency_confidence'] = {
                'mean': confidences.mean(),
                'median': confidences.median(),
                'std': confidences.std(),
                'min': confidences.min(),
                'max': confidences.max(),
                'high_confidence_count': len(confidences[confidences >= 0.8])
            }
        
        if 'category_confidence' in self.df.columns:
            confidences = self.df['category_confidence'].dropna()
            analysis['category_confidence'] = {
                'mean': confidences.mean(),
                'median': confidences.median(),
                'std': confidences.std(),
                'min': confidences.min(),
                'max': confidences.max(),
                'high_confidence_count': len(confidences[confidences >= 0.8])
            }
        
        return analysis
    
    def analyze_volume_trends(self, period='daily'):
        """Analyze email volume trends over time"""
        if 'date' not in self.df.columns:
            return {}
        
        analysis = {}
        
        if period == 'daily':
            daily_counts = self.df.groupby(self.df['date'].dt.date).size()
            analysis['total_days'] = len(daily_counts)
            analysis['average_daily_volume'] = daily_counts.mean()
            analysis['max_daily_volume'] = daily_counts.max()
            analysis['min_daily_volume'] = daily_counts.min()
            analysis['std_daily_volume'] = daily_counts.std()
        
        elif period == 'weekly':
            weekly_counts = self.df.groupby(self.df['date'].dt.isocalendar().week).size()
            analysis['total_weeks'] = len(weekly_counts)
            analysis['average_weekly_volume'] = weekly_counts.mean()
            analysis['max_weekly_volume'] = weekly_counts.max()
            analysis['min_weekly_volume'] = weekly_counts.min()
        
        return analysis
    
    def detect_anomalies(self):
        """Detect anomalies in the data"""
        anomalies = {
            'missing_values': {},
            'outliers': {},
            'unusual_patterns': []
        }
        
        # Check for missing values
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                pct = missing_count / len(self.df) * 100
                if pct > 5:  # Flag if > 5% missing
                    anomalies['missing_values'][col] = {
                        'count': missing_count,
                        'percentage': pct
                    }
        
        # Check for outliers in numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                if len(outliers) > 0:
                    anomalies['outliers'][col] = len(outliers)
        
        # Check for unusual patterns
        if 'urgency' in self.df.columns:
            # All same urgency level
            unique_urgency = self.df['urgency'].nunique()
            if unique_urgency == 1:
                anomalies['unusual_patterns'].append(
                    f"All emails have {self.df['urgency'].iloc[0]} urgency"
                )
        
        if 'category' in self.df.columns:
            # Highly imbalanced categories
            category_dist = self.df['category'].value_counts()
            max_pct = category_dist.iloc[0] / len(self.df)
            if max_pct > 0.9:
                anomalies['unusual_patterns'].append(
                    f"Highly imbalanced dataset: {category_dist.index[0]} is {max_pct*100:.1f}%"
                )
        
        return anomalies
    
    def identify_problem_categories(self):
        """Identify categories with most problems/complaints"""
        if 'category' not in self.df.columns:
            return {}
        
        problems = {}
        
        for category in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == category]
            
            high_urgency_count = len(cat_data[cat_data['urgency'] == 'High']) if 'urgency' in self.df.columns else 0
            high_urgency_pct = high_urgency_count / len(cat_data) * 100 if len(cat_data) > 0 else 0
            
            avg_confidence = cat_data['urgency_confidence'].mean() if 'urgency_confidence' in self.df.columns else 1.0
            
            problems[category] = {
                'high_urgency_count': high_urgency_count,
                'high_urgency_percentage': high_urgency_pct,
                'total_count': len(cat_data),
                'average_confidence': avg_confidence
            }
        
        return problems
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*70)
        print("EMAIL ANALYTICS SUMMARY REPORT")
        print("="*70)
        
        # Basic statistics
        print(f"\n📊 BASIC STATISTICS")
        print(f"  Total Emails: {len(self.df):,}")
        print(f"  Total Columns: {len(self.df.columns)}")
        
        # Category analysis
        cat_analysis = self.analyze_category_distribution()
        if cat_analysis:
            print(f"\n📂 CATEGORY ANALYSIS")
            print(f"  Total Categories: {cat_analysis.get('total_categories', 'N/A')}")
            if 'most_common' in cat_analysis:
                print(f"  Most Common: {cat_analysis['most_common']}")
            if 'least_common' in cat_analysis:
                print(f"  Least Common: {cat_analysis['least_common']}")
        
        # Urgency analysis
        urgency_analysis = self.analyze_urgency_distribution()
        if urgency_analysis:
            print(f"\n⚠️ URGENCY ANALYSIS")
            print(f"  High: {urgency_analysis.get('high_urgency', 0):,} ({urgency_analysis.get('high_urgency_pct', 0):.1f}%)")
            print(f"  Medium: {urgency_analysis.get('medium_urgency', 0):,} ({urgency_analysis.get('medium_urgency_pct', 0):.1f}%)")
            print(f"  Low: {urgency_analysis.get('low_urgency', 0):,} ({urgency_analysis.get('low_urgency_pct', 0):.1f}%)")
        
        # Confidence analysis
        conf_analysis = self.analyze_confidences()
        if conf_analysis:
            print(f"\n📈 CONFIDENCE ANALYSIS")
            if 'urgency_confidence' in conf_analysis:
                uc = conf_analysis['urgency_confidence']
                print(f"  Urgency Confidence - Mean: {uc['mean']:.2%}, Median: {uc['median']:.2%}, Std: {uc['std']:.2%}")
            if 'category_confidence' in conf_analysis:
                cc = conf_analysis['category_confidence']
                print(f"  Category Confidence - Mean: {cc['mean']:.2%}, Median: {cc['median']:.2%}, Std: {cc['std']:.2%}")
        
        # Volume trends
        volume_analysis = self.analyze_volume_trends()
        if volume_analysis:
            print(f"\n📊 VOLUME TRENDS (Daily)")
            print(f"  Average Daily: {volume_analysis.get('average_daily_volume', 0):.1f}")
            print(f"  Max Daily: {volume_analysis.get('max_daily_volume', 0):.0f}")
            print(f"  Min Daily: {volume_analysis.get('min_daily_volume', 0):.0f}")
        
        # Category-Urgency correlation
        corr_analysis = self.analyze_category_urgency_correlation()
        if corr_analysis:
            print(f"\n🔗 CATEGORY-URGENCY CORRELATION")
            print(f"  Highest Risk Category (High Urgency %):")
            max_risk_cat = max(corr_analysis.items(), key=lambda x: x[1].get('high_pct', 0))
            print(f"    {max_risk_cat[0]}: {max_risk_cat[1]['high_pct']:.1f}%")
        
        # Anomalies
        anomalies = self.detect_anomalies()
        if anomalies['missing_values'] or anomalies['outliers'] or anomalies['unusual_patterns']:
            print(f"\n⚠️ ANOMALIES DETECTED")
            if anomalies['missing_values']:
                print(f"  Missing Values:")
                for col, info in anomalies['missing_values'].items():
                    print(f"    - {col}: {info['count']} ({info['percentage']:.1f}%)")
            if anomalies['outliers']:
                print(f"  Outliers:")
                for col, count in anomalies['outliers'].items():
                    print(f"    - {col}: {count} outliers")
            if anomalies['unusual_patterns']:
                print(f"  Unusual Patterns:")
                for pattern in anomalies['unusual_patterns']:
                    print(f"    - {pattern}")
    
    def export_report(self, filepath):
        """Export detailed report as JSON"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_emails': len(self.df),
            'category_analysis': self.analyze_category_distribution(),
            'urgency_analysis': self.analyze_urgency_distribution(),
            'category_urgency_correlation': self.analyze_category_urgency_correlation(),
            'confidence_analysis': self.analyze_confidences(),
            'volume_trends': self.analyze_volume_trends(),
            'anomalies': self.detect_anomalies(),
            'problem_categories': self.identify_problem_categories()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Report exported to {filepath}")
        return report


# Usage Example
if __name__ == "__main__":
    # Example with sample data
    sample_data = {
        'category': np.random.choice(['spam', 'promotions', 'forum', 'social_media', 'updates', 'verify_code'], 500),
        'urgency': np.random.choice(['High', 'Medium', 'Low'], 500, p=[0.15, 0.35, 0.5]),
        'urgency_confidence': np.random.uniform(0.5, 1.0, 500),
        'category_confidence': np.random.uniform(0.6, 1.0, 500),
        'date': [datetime.now() - timedelta(days=int(x)) for x in np.random.uniform(0, 365, 500)]
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    analytics = EmailAnalytics(sample_df)
    analytics.generate_summary_report()
    analytics.export_report('analytics_report.json')
