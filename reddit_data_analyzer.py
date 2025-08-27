#!/usr/bin/env python3
"""
Reddit Data Analyzer
Analyze collected Reddit data with engagement scoring and filtering
"""

import pandas as pd
import numpy as np
import logging
import json
import sys
from datetime import datetime
import os
import glob
import re

# Sentiment analysis with VADER
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    logging.info("VADER sentiment analyzer available")
except ImportError:
    VADER_AVAILABLE = False
    logging.error("VADER not installed. Run: pip install vaderSentiment")
    raise ImportError("VADER is required for sentiment analysis")

# Topic modeling with Scikit-learn NMF
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    SKLEARN_AVAILABLE = True
    logging.info("Scikit-learn topic modeling available")
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not installed. Run: pip install scikit-learn")
    logging.warning("Topic modeling will be disabled")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_analysis.log'),
        logging.StreamHandler()
    ],
    force=True  # Override any existing logging config
)

class RedditDataAnalyzer:
    def __init__(self):
        """Initialize Reddit data analyzer with VADER sentiment analysis"""
        logging.info("Reddit Data Analyzer initialized")
        
        # Initialize VADER sentiment analyzer
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            logging.info("VADER sentiment analyzer loaded")
        else:
            self.sentiment_analyzer = None
            logging.warning("Sentiment analysis disabled - VADER not available")
    
    def load_data(self, file_path):
        """
        Load Reddit data from CSV or JSON file
        
        Args:
            file_path (str): Path to data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            if file_path.endswith('.csv'):
                # Read CSV with proper handling for complex JSON data
                df = pd.read_csv(file_path, escapechar='\\', quoting=1)  # quoting=1 is QUOTE_ALL
                logging.info(f"Loaded {len(df)} posts from CSV: {file_path}")
                
                # Deserialize JSON-encoded comments back to lists
                if 'comments' in df.columns:
                    def deserialize_comments(comments_json):
                        try:
                            if pd.isna(comments_json) or comments_json == '' or comments_json == '[]':
                                return []
                            if isinstance(comments_json, str):
                                # Handle escaped quotes in CSV
                                comments_json = comments_json.replace('""', '"')
                                parsed = json.loads(comments_json)
                                return parsed if isinstance(parsed, list) else []
                            return comments_json if isinstance(comments_json, list) else []
                        except (json.JSONDecodeError, TypeError) as e:
                            # Log deserialization failures for debugging
                            if isinstance(comments_json, str) and len(comments_json) > 10:
                                logging.warning(f"Failed to deserialize comments (length {len(comments_json)}): {str(e)[:100]}")
                            return []
                    
                    initial_empty = df['comments'].apply(lambda x: x == '[]' or pd.isna(x) or x == '').sum()
                    df['comments'] = df['comments'].apply(deserialize_comments)
                    final_empty = df['comments'].apply(lambda x: len(x) == 0).sum()
                    
                    if initial_empty != final_empty:
                        logging.info(f"Comments deserialized: {len(df) - final_empty}/{len(df)} posts have comments")
                    else:
                        logging.warning(f"Comment deserialization may have failed: {final_empty}/{len(df)} posts have no comments")
                    
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                logging.info(f"Loaded {len(df)} posts from JSON: {file_path}")
            else:
                raise ValueError("File must be CSV or JSON format")
            
            # Convert datetime columns if they exist
            datetime_columns = ['created_datetime', 'collection_timestamp']
            for col in datetime_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except (ValueError, TypeError):
                        logging.warning(f"Could not convert {col} to datetime, keeping as string")
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {e}")
            return pd.DataFrame()
    
    def calculate_engagement_score(self, df, weights=None):
        """
        Calculate engagement score for each post
        
        Args:
            df (pd.DataFrame): DataFrame with Reddit posts
            weights (dict): Custom weights for scoring components
            
        Returns:
            pd.DataFrame: DataFrame with engagement scores added
        """
        if weights is None:
            weights = {
                'score': 0.3,           # Reddit score (upvotes - downvotes)
                'upvote_ratio': 0.2,    # Upvote ratio
                'comments': 0.25,       # Number of comments
                'comments_quality': 0.15,  # Quality of comments (if available)
                'recency': 0.1          # Recency factor
            }
        
        df_scored = df.copy()
        
        # Normalize score (0-100 scale)
        if 'score' in df.columns:
            score_max = df['score'].max() if df['score'].max() > 0 else 1
            df_scored['score_normalized'] = (df['score'] / score_max * 100).clip(0, 100)
        else:
            df_scored['score_normalized'] = 0
        
        # Upvote ratio (already 0-1, convert to 0-100)
        if 'upvote_ratio' in df.columns:
            df_scored['upvote_ratio_normalized'] = (df['upvote_ratio'] * 100).clip(0, 100)
        else:
            df_scored['upvote_ratio_normalized'] = 50  # Neutral
        
        # Normalize comment count
        if 'num_comments' in df.columns:
            comments_max = df['num_comments'].max() if df['num_comments'].max() > 0 else 1
            df_scored['comments_normalized'] = (df['num_comments'] / comments_max * 100).clip(0, 100)
        else:
            df_scored['comments_normalized'] = 0
        
        # Comment quality score (based on collected comments if available)
        if 'comments_collected' in df.columns and 'comments' in df.columns:
            df_scored['comment_quality_score'] = df.apply(self._calculate_comment_quality, axis=1)
        else:
            df_scored['comment_quality_score'] = 0
        
        # Recency score (newer posts get higher score)
        if 'hours_ago' in df.columns:
            # Posts within last 24h get max points, decreasing over time
            df_scored['recency_score'] = np.maximum(0, 100 - (df['hours_ago'] / 24 * 50)).clip(0, 100)
        elif 'days_ago' in df.columns:
            df_scored['recency_score'] = np.maximum(0, 100 - (df['days_ago'] * 50 / 14)).clip(0, 100)
        else:
            df_scored['recency_score'] = 50  # Neutral
        
        # Calculate weighted engagement score
        df_scored['engagement_score'] = (
            df_scored['score_normalized'] * weights['score'] +
            df_scored['upvote_ratio_normalized'] * weights['upvote_ratio'] +
            df_scored['comments_normalized'] * weights['comments'] +
            df_scored['comment_quality_score'] * weights['comments_quality'] +
            df_scored['recency_score'] * weights['recency']
        ).round(2)
        
        logging.info(f"Calculated engagement scores for {len(df_scored)} posts")
        logging.info(f"Score range: {df_scored['engagement_score'].min():.2f} - {df_scored['engagement_score'].max():.2f}")
        logging.info(f"Average score: {df_scored['engagement_score'].mean():.2f}")
        
        return df_scored
    
    def _calculate_comment_quality(self, row):
        """Calculate comment quality score based on comment data"""
        if not row.get('comments') or row['comments_collected'] == 0:
            return 0
        
        try:
            comments = row['comments']
            if isinstance(comments, str):
                return 0  # Comments stored as string, can't analyze
            
            # Calculate average comment score
            comment_scores = [c.get('score', 0) for c in comments if isinstance(c, dict)]
            if comment_scores:
                avg_score = np.mean(comment_scores)
                max_score = max(comment_scores)
                # Normalize to 0-100 scale
                quality_score = min(100, (avg_score + max_score) * 5)
                return max(0, quality_score)
            
        except Exception:
            pass
        
        return 0
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using VADER
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores and classification
        """
        if not self.sentiment_analyzer or not text or pd.isna(text):
            return {
                'compound': 0.0,
                'pos': 0.0, 
                'neu': 1.0,
                'neg': 0.0,
                'classification': 'neutral',
                'priority_score': 50.0
            }
        
        # Clean text for better analysis
        cleaned_text = re.sub(r'http\S+|www.\S+', '', str(text))  # Remove URLs
        cleaned_text = re.sub(r'@\w+|#\w+', '', cleaned_text)     # Remove mentions/hashtags
        cleaned_text = re.sub(r'[^\w\s.,!?]', ' ', cleaned_text)  # Keep basic punctuation
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Normalize whitespace
        
        if not cleaned_text:
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0, 
                'neg': 0.0,
                'classification': 'neutral',
                'priority_score': 50.0
            }
        
        # Get VADER scores
        scores = self.sentiment_analyzer.polarity_scores(cleaned_text)
        
        # Classify sentiment based on compound score
        compound = scores['compound']
        if compound >= 0.05:
            classification = 'positive'
        elif compound <= -0.05:
            classification = 'negative'
        else:
            classification = 'neutral'
        
        # Calculate priority score (0-100, higher = more positive/constructive)
        # Positive sentiment gets higher priority
        if compound > 0:
            priority_score = min(100, 50 + (compound * 50) + (scores['pos'] * 25))
        else:
            # Negative sentiment gets lower priority, but not too harsh
            priority_score = max(10, 50 + (compound * 30))
        
        return {
            'compound': scores['compound'],
            'pos': scores['pos'],
            'neu': scores['neu'], 
            'neg': scores['neg'],
            'classification': classification,
            'priority_score': round(priority_score, 2)
        }
    
    def add_sentiment_analysis(self, df):
        """Add sentiment analysis to posts"""
        if not self.sentiment_analyzer:
            logging.warning("Sentiment analysis skipped - VADER not available")
            return df
        
        df_with_sentiment = df.copy()
        logging.info(f"Analyzing sentiment for {len(df)} posts...")
        
        # Analyze post titles
        logging.info("Analyzing post titles...")
        title_sentiments = []
        for i, title in enumerate(df_with_sentiment['title']):
            if i % 100 == 0 and i > 0:
                logging.info(f"  Processed {i}/{len(df_with_sentiment)} titles")
            title_sentiments.append(self.analyze_sentiment(title))
        
        # Add title sentiment columns
        df_with_sentiment['title_sentiment_compound'] = [s['compound'] for s in title_sentiments]
        df_with_sentiment['title_sentiment_pos'] = [s['pos'] for s in title_sentiments]
        df_with_sentiment['title_sentiment_neu'] = [s['neu'] for s in title_sentiments]
        df_with_sentiment['title_sentiment_neg'] = [s['neg'] for s in title_sentiments]
        df_with_sentiment['title_sentiment_class'] = [s['classification'] for s in title_sentiments]
        df_with_sentiment['title_sentiment_priority'] = [s['priority_score'] for s in title_sentiments]
        
        # Analyze post content if available
        if 'selftext' in df_with_sentiment.columns:
            logging.info("Analyzing post content...")
            content_sentiments = []
            for i, content in enumerate(df_with_sentiment['selftext']):
                if i % 100 == 0 and i > 0:
                    logging.info(f"  Processed {i}/{len(df_with_sentiment)} posts")
                
                # Use title sentiment if no content
                if pd.isna(content) or not content or len(str(content).strip()) < 10:
                    content_sentiments.append(title_sentiments[i])
                else:
                    content_sentiments.append(self.analyze_sentiment(content))
            
            df_with_sentiment['content_sentiment_class'] = [s['classification'] for s in content_sentiments]
            df_with_sentiment['content_sentiment_priority'] = [s['priority_score'] for s in content_sentiments]
        
        # Analyze comments sentiment (average of top comments)
        if 'comments' in df_with_sentiment.columns:
            logging.info("Analyzing comments sentiment...")
            df_with_sentiment['comments_sentiment_avg'] = df_with_sentiment.apply(
                self._analyze_comments_sentiment, axis=1
            )
        else:
            df_with_sentiment['comments_sentiment_avg'] = 50.0  # Neutral default
        
        # Calculate overall sentiment score (weighted average)
        title_weight = 0.4
        content_weight = 0.35 if 'selftext' in df_with_sentiment.columns else 0
        comments_weight = 0.25 if 'comments' in df_with_sentiment.columns else 0
        
        # Adjust weights if components missing
        if content_weight == 0:
            title_weight = 0.6
            comments_weight = 0.4 if comments_weight > 0 else 0
        if comments_weight == 0:
            title_weight = 0.6
            content_weight = 0.4
        
        # Calculate weighted overall sentiment
        overall_sentiment = (
            df_with_sentiment['title_sentiment_priority'] * title_weight +
            (df_with_sentiment.get('content_sentiment_priority', df_with_sentiment['title_sentiment_priority']) * content_weight) +
            df_with_sentiment['comments_sentiment_avg'] * comments_weight
        )
        
        df_with_sentiment['overall_sentiment_score'] = overall_sentiment.round(2)
        
        # Overall classification
        df_with_sentiment['overall_sentiment_class'] = df_with_sentiment['overall_sentiment_score'].apply(
            lambda x: 'positive' if x >= 65 else 'negative' if x <= 35 else 'neutral'
        )
        
        # Log results
        sentiment_counts = df_with_sentiment['overall_sentiment_class'].value_counts()
        logging.info(f"Sentiment analysis complete!")
        logging.info(f"Results: {sentiment_counts.to_dict()}")
        
        return df_with_sentiment
    
    def _analyze_comments_sentiment(self, row):
        """Analyze sentiment of comments for a post"""
        if not row.get('comments') or row['comments_collected'] == 0:
            return 50.0  # Neutral
        
        try:
            comments = row['comments'] 
            if isinstance(comments, str):
                return 50.0
            
            comment_scores = []
            for comment in comments[:5]:  # Top 5 comments
                if isinstance(comment, dict) and 'body' in comment:
                    body = comment['body']
                    if len(body) > 10:  # Skip very short comments
                        sentiment = self.analyze_sentiment(body)
                        comment_scores.append(sentiment['priority_score'])
            
            return np.mean(comment_scores) if comment_scores else 50.0
            
        except Exception:
            return 50.0  # Default neutral
    
    def perform_topic_modeling(self, df, n_topics=8, min_posts=5):
        """
        Perform topic modeling on post content using NMF
        
        Args:
            df (pd.DataFrame): DataFrame with posts
            n_topics (int): Number of topics to extract
            min_posts (int): Minimum posts required to run topic modeling
            
        Returns:
            pd.DataFrame: DataFrame with topic assignments and topic info
        """
        if not SKLEARN_AVAILABLE:
            logging.warning("Topic modeling skipped - Scikit-learn not available")
            return df
        
        if len(df) < min_posts:
            logging.warning(f"Too few posts ({len(df)}) for topic modeling (minimum: {min_posts})")
            return df
        
        df_topics = df.copy()
        
        # Combine title and content for topic modeling
        logging.info(f"Preparing text for topic modeling on {len(df)} posts...")
        documents = []
        doc_indices = []
        
        for idx, row in df.iterrows():
            # Combine title and selftext
            title = str(row.get('title', '')) if pd.notna(row.get('title', '')) else ''
            content = str(row.get('selftext', '')) if pd.notna(row.get('selftext', '')) else ''
            
            # Create document text
            doc_text = f"{title} {content}".strip()
            
            if len(doc_text) > 20:  # Skip very short posts
                documents.append(doc_text)
                doc_indices.append(idx)
        
        if len(documents) < min_posts:
            logging.warning(f"Too few valid documents ({len(documents)}) for topic modeling")
            return df
        
        logging.info(f"Running NMF topic modeling on {len(documents)} documents...")
        
        # Custom stop words for data engineering domain
        custom_stop_words = set(ENGLISH_STOP_WORDS) | {
            'data', 'use', 'using', 'used', 'like', 'would', 'could', 'should',
            'getting', 'need', 'want', 'trying', 'looking', 'help', 'thanks',
            'anyone', 'someone', 'people', 'way', 'ways', 'good', 'best',
            'question', 'questions', 'know', 'think', 'time', 'work', 'working',
            'im', 'ive', 'dont', 'didnt', 'cant', 'wont', 'isnt', 'arent'
        }
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=list(custom_stop_words),
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,           # Ignore terms in less than 2 documents
            max_df=0.8,         # Ignore terms in more than 80% of documents
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]+\b'  # Alphanumeric tokens
        )
        
        try:
            # Fit vectorizer and transform documents
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Perform NMF topic modeling
            nmf_model = NMF(
                n_components=n_topics,
                random_state=42,
                max_iter=200,
                init='nndsvd'
            )
            
            # Fit NMF and get topic assignments
            doc_topic_matrix = nmf_model.fit_transform(tfidf_matrix)
            
            # Get feature names for topic interpretation
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract and name topics
            topic_info = self._extract_topic_info(nmf_model, feature_names, n_topics)
            
            # Assign topics to documents
            topic_assignments = []
            topic_probabilities = []
            
            for i, doc_idx in enumerate(doc_indices):
                # Get dominant topic for this document
                topic_scores = doc_topic_matrix[i]
                dominant_topic = np.argmax(topic_scores)
                max_probability = topic_scores[dominant_topic]
                
                # Only assign topic if probability is above threshold
                if max_probability > 0.1:
                    topic_assignments.append(dominant_topic)
                    topic_probabilities.append(max_probability)
                else:
                    topic_assignments.append(-1)  # No clear topic
                    topic_probabilities.append(0.0)
            
            # Add topic information to dataframe
            df_topics['topic_id'] = -1
            df_topics['topic_probability'] = 0.0
            df_topics['topic_name'] = 'uncategorized'
            
            for i, doc_idx in enumerate(doc_indices):
                df_topics.loc[doc_idx, 'topic_id'] = topic_assignments[i]
                df_topics.loc[doc_idx, 'topic_probability'] = round(topic_probabilities[i], 3)
                if topic_assignments[i] >= 0:
                    df_topics.loc[doc_idx, 'topic_name'] = topic_info[topic_assignments[i]]['name']
            
            # Log topic modeling results
            logging.info(f"Topic modeling complete!")
            topic_distribution = df_topics['topic_name'].value_counts()
            logging.info(f"Topic distribution: {topic_distribution.to_dict()}")
            
            # Store topic information for reporting
            self.topic_info = topic_info
            
            return df_topics
            
        except Exception as e:
            logging.error(f"Topic modeling failed: {e}")
            return df
    
    def _extract_topic_info(self, nmf_model, feature_names, n_topics):
        """Extract interpretable topic information"""
        topic_info = {}
        
        # ETL/Data Pipeline topic keywords for naming
        topic_keywords = {
            'etl': ['etl', 'pipeline', 'extract', 'transform', 'load', 'data pipeline', 'data integration'],
            'bigquery': ['bigquery', 'gcp', 'google cloud', 'sql', 'warehouse', 'analytics'],
            'tools': ['tool', 'tools', 'airbyte', 'fivetran', 'dbt', 'airflow', 'dagster'],
            'automation': ['automation', 'automated', 'schedule', 'workflow', 'orchestration'],
            'databases': ['database', 'postgres', 'mysql', 'snowflake', 'redshift', 'mongodb'],
            'api': ['api', 'rest', 'endpoint', 'integration', 'connector', 'webhook'],
            'analytics': ['analytics', 'analysis', 'reporting', 'dashboard', 'visualization', 'bi'],
            'performance': ['performance', 'optimization', 'slow', 'speed', 'latency', 'scale']
        }
        
        for topic_idx in range(n_topics):
            # Get top words for this topic
            topic_words = []
            word_scores = nmf_model.components_[topic_idx]
            top_word_indices = word_scores.argsort()[-15:][::-1]  # Top 15 words
            
            for word_idx in top_word_indices:
                word = feature_names[word_idx]
                score = word_scores[word_idx]
                topic_words.append((word, round(score, 3)))
            
            # Generate topic name based on top words
            top_words = [word for word, _ in topic_words[:5]]
            topic_name = self._generate_topic_name(top_words, topic_keywords)
            
            topic_info[topic_idx] = {
                'name': topic_name,
                'top_words': topic_words[:10],
                'word_list': [word for word, _ in topic_words[:10]]
            }
        
        return topic_info
    
    def _generate_topic_name(self, top_words, topic_keywords):
        """Generate interpretable topic name"""
        top_words_lower = [word.lower() for word in top_words]
        
        # Check for matches with predefined categories
        for category, keywords in topic_keywords.items():
            matches = sum(1 for keyword in keywords if any(kw in ' '.join(top_words_lower) for kw in keyword.split()))
            if matches > 0:
                return f"{category}_{matches}"
        
        # Fallback: use top 2 most distinctive words
        return f"{top_words[0]}_{top_words[1]}" if len(top_words) >= 2 else top_words[0]
    
    def filter_by_relevance(self, df):
        """
        PRIMARY FILTER: Filter posts by business relevance using topics and keywords
        This runs FIRST before all other filtering to focus on relevant discussions
        
        Returns:
            pd.DataFrame: Only posts relevant to data integration, warehousing, ETL
        """
        logging.info(f"Applying PRIMARY relevance filter to {len(df)} posts...")
        
        # Define relevant topics (if topic modeling was performed)
        relevant_topics = [
            'etl', 'bigquery', 'tools', 'automation', 'databases', 
            'api', 'analytics', 'performance', 'snowflake', 'databricks'
        ]
        
        # Define source systems keywords
        source_keywords = [
            # Advertising platforms
            'bing ads', 'twitter ads', 'tiktok ads', 'snapchat ads', 'facebook ads', 
            'linkedin ads', 'pinterest ads', 'google ads', 'campaign manager', 'display video 360',
            'adroll', 'taboola', 'outbrain', 'criteo', 'apple search ads', 'quora ads',
            'rakuten advertising', 'spotify ads', 'reddit ads', 'youtube ads',
            
            # Marketing & CRM
            'hubspot', 'salesforce', 'mailchimp', 'activecampaign', 'klaviyo', 
            'brevo', 'pipedrive', 'zoho crm', 'zendesk', 'intercom', 'freshdesk',
            'notion', 'slack', 'trello', 'monday.com', 'clickup', 'jira',
            
            # Analytics & Data
            'google analytics', 'adobe analytics', 'amplitude', 'mixpanel', 'datadog',
            'google sheets', 'airtable', 'metabase', 'surveymonkey', 'typeform',
            
            # E-commerce & Business
            'shopify', 'bigcommerce', 'woocommerce', 'magento', 'prestashop',
            'stripe', 'braintree', 'square', 'quickbooks', 'xero', 'netsuite',
            'zuora', 'chargebee', 'maxio'
        ]
        
        # Define destination systems keywords  
        destination_keywords = [
            'bigquery', 'snowflake', 'redshift', 'databricks', 'looker studio',
            'power bi', 'powerbi', 'tableau', 'google sheets', 'excel',
            'amazon s3', 'azure blob', 'postgresql', 'mysql', 'sql server',
            'python', 'r language', 'jupyter', 'notebooks'
        ]
        
        # Define implementation/integration keywords
        implementation_keywords = [
            'etl', 'elt', 'pipeline', 'integration', 'connector', 'sync',
            'data warehouse', 'data lake', 'architecture', 'optimization',
            'implementation', 'migration', 'setup', 'configuration',
            'api integration', 'data transformation', 'data modeling',
            'performance tuning', 'best practices', 'troubleshooting',
            'data quality', 'monitoring', 'automation', 'orchestration',
            'real-time', 'batch processing', 'incremental load'
        ]
        
        # Combine all relevant keywords
        all_keywords = source_keywords + destination_keywords + implementation_keywords
        
        df_relevant = df.copy()
        relevance_scores = []
        
        for idx, row in df.iterrows():
            score = 0
            
            # Check topic relevance (if topic modeling was done)
            if 'topic_name' in row and pd.notna(row['topic_name']):
                topic_name = str(row['topic_name']).lower()
                if any(topic in topic_name for topic in relevant_topics):
                    score += 30  # High score for relevant topics
            
            # Combine title and content for keyword matching
            title = str(row.get('title', '')).lower() if pd.notna(row.get('title', '')) else ''
            content = str(row.get('selftext', '')).lower() if pd.notna(row.get('selftext', '')) else ''
            text = f"{title} {content}"
            
            # Score based on keyword matches
            keyword_matches = sum(1 for keyword in all_keywords if keyword in text)
            score += keyword_matches * 10  # 10 points per keyword match
            
            # Bonus for multiple keyword categories
            source_matches = sum(1 for keyword in source_keywords if keyword in text)
            dest_matches = sum(1 for keyword in destination_keywords if keyword in text)
            impl_matches = sum(1 for keyword in implementation_keywords if keyword in text)
            
            if source_matches > 0 and dest_matches > 0:
                score += 25  # Bonus for source + destination discussion
            if impl_matches > 0 and (source_matches > 0 or dest_matches > 0):
                score += 20  # Bonus for implementation + system discussion
            
            # Check for specific business use cases in title/content
            business_patterns = [
                'connect.*to.*', 'integrate.*with.*', 'sync.*data', 
                'etl.*pipeline', 'data.*warehouse', 'implementation.*challenge',
                'architecture.*question', 'optimization.*problem', 'migration.*',
                'best.*practice', 'real.*time.*data', 'batch.*processing'
            ]
            
            pattern_matches = sum(1 for pattern in business_patterns 
                                if re.search(pattern, text, re.IGNORECASE))
            score += pattern_matches * 15
            
            # Check comments for relevance (if available)
            if row.get('comments') and isinstance(row['comments'], list):
                comment_text = ' '.join([
                    c.get('body', '').lower() for c in row['comments'][:5] 
                    if isinstance(c, dict) and c.get('body')
                ])
                comment_matches = sum(1 for keyword in all_keywords if keyword in comment_text)
                score += comment_matches * 5  # Lower weight for comment matches
            
            relevance_scores.append(score)
        
        # Add relevance score to dataframe
        df_relevant['relevance_score'] = relevance_scores
        
        # Filter: Keep posts with relevance score >= 50 (much stricter)
        relevance_threshold = 50
        df_filtered = df_relevant[df_relevant['relevance_score'] >= relevance_threshold].copy()
        
        # Sort by relevance score (highest first)
        df_filtered = df_filtered.sort_values('relevance_score', ascending=False)
        
        filtered_count = len(df_filtered)
        original_count = len(df)
        
        logging.info(f"PRIMARY RELEVANCE FILTER:")
        logging.info(f"  Original posts: {original_count}")
        logging.info(f"  Relevant posts: {filtered_count} ({filtered_count/original_count*100:.1f}%)")
        logging.info(f"  Filtered out: {original_count - filtered_count} non-relevant posts")
        
        if filtered_count > 0:
            avg_relevance = df_filtered['relevance_score'].mean()
            max_relevance = df_filtered['relevance_score'].max()
            logging.info(f"  Average relevance score: {avg_relevance:.1f}")
            logging.info(f"  Max relevance score: {max_relevance}")
            
        
        return df_filtered
    
    def filter_by_thresholds(self, df, thresholds):
        """
        Filter posts by engagement thresholds
        
        Args:
            df (pd.DataFrame): DataFrame with engagement scores
            thresholds (dict): Threshold criteria
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        df_filtered = df.copy()
        original_count = len(df_filtered)
        
        # Apply thresholds
        if 'min_engagement_score' in thresholds:
            df_filtered = df_filtered[df_filtered['engagement_score'] >= thresholds['min_engagement_score']]
            logging.info(f"Min engagement score ({thresholds['min_engagement_score']}): {len(df_filtered)} posts remaining")
        
        if 'min_score' in thresholds:
            df_filtered = df_filtered[df_filtered['score'] >= thresholds['min_score']]
            logging.info(f"Min Reddit score ({thresholds['min_score']}): {len(df_filtered)} posts remaining")
        
        if 'min_comments' in thresholds:
            df_filtered = df_filtered[df_filtered['num_comments'] >= thresholds['min_comments']]
            logging.info(f"Min comments ({thresholds['min_comments']}): {len(df_filtered)} posts remaining")
        
        if 'min_upvote_ratio' in thresholds:
            df_filtered = df_filtered[df_filtered['upvote_ratio'] >= thresholds['min_upvote_ratio']]
            logging.info(f"Min upvote ratio ({thresholds['min_upvote_ratio']}): {len(df_filtered)} posts remaining")
        
        if 'max_hours_ago' in thresholds and 'hours_ago' in df.columns:
            df_filtered = df_filtered[df_filtered['hours_ago'] <= thresholds['max_hours_ago']]
            logging.info(f"Max hours ago ({thresholds['max_hours_ago']}): {len(df_filtered)} posts remaining")
        
        if 'max_days_ago' in thresholds and 'days_ago' in df.columns:
            df_filtered = df_filtered[df_filtered['days_ago'] <= thresholds['max_days_ago']]
            logging.info(f"Max days ago ({thresholds['max_days_ago']}): {len(df_filtered)} posts remaining")
        
        if 'subreddits' in thresholds:
            df_filtered = df_filtered[df_filtered['subreddit'].isin(thresholds['subreddits'])]
            logging.info(f"Subreddit filter: {len(df_filtered)} posts remaining")
        
        # Filter out posts mentioning specific companies/terms in comments
        if 'exclude_keywords' in thresholds and thresholds['exclude_keywords']:
            if 'comments' in df_filtered.columns:
                initial_count = len(df_filtered)
                excluded_keywords = thresholds['exclude_keywords']
                df_filtered = df_filtered[~df_filtered.apply(lambda row: self._has_keywords_in_comments(row, excluded_keywords), axis=1)]
                excluded_count = initial_count - len(df_filtered)
                if excluded_count > 0:
                    logging.info(f"Excluded {excluded_count} posts mentioning excluded keywords in comments: {len(df_filtered)} posts remaining")
        
        filtered_count = len(df_filtered)
        logging.info(f"Filtering complete: {filtered_count}/{original_count} posts passed thresholds ({filtered_count/original_count*100:.1f}%)")
        
        return df_filtered
    
    def _has_keywords_in_comments(self, row, exclude_keywords):
        """Check if a post has specific keywords mentions in comments"""
        if not row.get('comments') or row['comments_collected'] == 0:
            return False
        
        try:
            comments = row['comments']
            if isinstance(comments, str):
                return False  # Can't check string data
            
            # Convert keywords to lowercase for case-insensitive matching
            keywords_lower = [keyword.lower() for keyword in exclude_keywords]
            
            for comment in comments:
                if isinstance(comment, dict) and 'body' in comment:
                    comment_text = comment['body'].lower()
                    if any(keyword in comment_text for keyword in keywords_lower):
                        return True
            
            return False
            
        except Exception:
            return False  # Safe fallback
    
    def apply_final_sorting(self, df):
        """
        Apply final sorting to prioritize best content for output
        
        Returns:
            pd.DataFrame: Sorted dataframe with best content first
        """
        if df.empty:
            return df
        
        df_sorted = df.copy()
        
        # Create composite sorting criteria
        sort_columns = []
        sort_ascending = []
        
        # Primary: Relevance score (if available)
        if 'relevance_score' in df_sorted.columns:
            sort_columns.append('relevance_score')
            sort_ascending.append(False)  # Highest first
        
        # Secondary: Sentiment class (positive first)
        if 'overall_sentiment_class' in df_sorted.columns:
            # Create numerical sentiment for sorting: positive=3, neutral=2, negative=1
            sentiment_order = {'positive': 3, 'neutral': 2, 'negative': 1}
            df_sorted['_sentiment_order'] = df_sorted['overall_sentiment_class'].map(sentiment_order).fillna(2)
            sort_columns.append('_sentiment_order')
            sort_ascending.append(False)  # Positive first
        
        # Tertiary: Engagement score
        if 'engagement_score' in df_sorted.columns:
            sort_columns.append('engagement_score')
            sort_ascending.append(False)  # Highest first
        
        # Quaternary: Reddit score
        if 'score' in df_sorted.columns:
            sort_columns.append('score')
            sort_ascending.append(False)  # Highest first
        
        # Final: Recency (most recent first)
        if 'hours_ago' in df_sorted.columns:
            sort_columns.append('hours_ago')
            sort_ascending.append(True)  # Lowest hours_ago = most recent
        elif 'days_ago' in df_sorted.columns:
            sort_columns.append('days_ago')
            sort_ascending.append(True)  # Lowest days_ago = most recent
        
        # Apply sorting
        if sort_columns:
            df_sorted = df_sorted.sort_values(sort_columns, ascending=sort_ascending)
            
            # Remove temporary sorting column
            if '_sentiment_order' in df_sorted.columns:
                df_sorted = df_sorted.drop('_sentiment_order', axis=1)
            
            logging.info(f"📊 Final sorting applied with criteria: {sort_columns}")
            logging.info(f"   Best content prioritized: relevance → sentiment → engagement → recency")
        
        return df_sorted
    
    def generate_analysis_report(self, df):
        """Generate comprehensive analysis report"""
        logging.info("\n" + "="*50)
        logging.info("REDDIT DATA ANALYSIS REPORT")
        logging.info("="*50)
        
        # Basic stats
        logging.info(f"\n--- DATASET OVERVIEW ---")
        logging.info(f"Total posts: {len(df)}")
        logging.info(f"Unique subreddits: {df['subreddit'].nunique()}")
        logging.info(f"Date range: {df['created_datetime'].min()} to {df['created_datetime'].max()}")
        
        # Engagement score analysis
        if 'engagement_score' in df.columns:
            logging.info(f"\n--- ENGAGEMENT SCORE ANALYSIS ---")
            logging.info(f"Average engagement score: {df['engagement_score'].mean():.2f}")
            logging.info(f"Median engagement score: {df['engagement_score'].median():.2f}")
            logging.info(f"Score range: {df['engagement_score'].min():.2f} - {df['engagement_score'].max():.2f}")
            logging.info(f"High engagement posts (>70): {len(df[df['engagement_score'] > 70])}")
            logging.info(f"Medium engagement posts (30-70): {len(df[(df['engagement_score'] >= 30) & (df['engagement_score'] <= 70)])}")
            logging.info(f"Low engagement posts (<30): {len(df[df['engagement_score'] < 30])}")
        
        # Top performing posts
        logging.info(f"\n--- TOP 10 POSTS BY ENGAGEMENT ---")
        if 'engagement_score' in df.columns:
            top_posts = df.nlargest(10, 'engagement_score')[['title', 'subreddit', 'score', 'num_comments', 'engagement_score']]
        else:
            top_posts = df.nlargest(10, 'score')[['title', 'subreddit', 'score', 'num_comments']]
        
        for idx, post in top_posts.iterrows():
            title_short = post['title'][:60] + "..." if len(post['title']) > 60 else post['title']
            score_text = f"Engagement: {post['engagement_score']:.1f}, " if 'engagement_score' in post else ""
            logging.info(f"r/{post['subreddit']}: {title_short}")
            logging.info(f"  {score_text}Score: {post['score']}, Comments: {post['num_comments']}")
        
        # Subreddit analysis
        logging.info(f"\n--- TOP SUBREDDITS ---")
        subreddit_stats = df.groupby('subreddit').agg({
            'score': ['count', 'mean'],
            'num_comments': 'mean',
            'engagement_score': 'mean' if 'engagement_score' in df.columns else 'score'
        }).round(2)
        
        subreddit_stats.columns = ['post_count', 'avg_score', 'avg_comments', 'avg_engagement']
        top_subreddits = subreddit_stats.sort_values('avg_engagement', ascending=False).head(10)
        
        for subreddit, stats in top_subreddits.iterrows():
            logging.info(f"r/{subreddit}: {stats['post_count']} posts, "
                        f"Avg Score: {stats['avg_score']:.1f}, "
                        f"Avg Comments: {stats['avg_comments']:.1f}, "
                        f"Avg Engagement: {stats['avg_engagement']:.1f}")
    
    def save_analysis_results(self, df, filename_prefix, save_json=False):
        """Save analyzed data to CSV and optionally JSON in analysis folder"""
        # Create analysis folder if it doesn't exist
        output_folder = "analysis_results"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logging.info(f"Created output folder: {output_folder}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_filename = os.path.join(output_folder, f"{filename_prefix}_analyzed_{timestamp}.csv")
        df.to_csv(csv_filename, index=False)
        logging.info(f"Saved analyzed data to: {csv_filename}")
        
        # Save JSON (optional)
        json_filename = None
        if save_json:
            json_filename = os.path.join(output_folder, f"{filename_prefix}_analyzed_{timestamp}.json")
            df_json = df.copy()
            
            # Convert datetime columns to strings
            for col in df_json.columns:
                if df_json[col].dtype == 'datetime64[ns]':
                    df_json[col] = df_json[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            df_json.to_json(json_filename, indent=2, orient='records')
            logging.info(f"Saved analyzed data to: {json_filename}")
        
        return csv_filename, json_filename
    
    def _harmonize_schema(self, df):
        """
        Harmonize schema across different collection types
        Ensures all DataFrames have consistent core columns for safe combining
        """
        # Define the core schema that all datasets should have
        core_columns = {
            # Required Reddit fields
            'id': '',
            'title': '',
            'author': '',
            'score': 0,
            'upvote_ratio': 0.0,
            'num_comments': 0,
            'created_utc': 0.0,
            'created_datetime': '',
            'hours_ago': 0.0,
            'days_ago': 0.0,
            'url': '',
            'permalink': '',
            'selftext': '',
            'is_self': False,
            'domain': '',
            'subreddit': '',
            'flair_text': '',
            'is_video': False,
            'stickied': False,
            'over_18': False,
            'spoiler': False,
            'locked': False,
            'collection_timestamp': '',
            'comments': [],
            'comments_collected': 0,
            
            # Collection-specific fields (will be NaN/empty for files that don't have them)
            'search_query': '',
            'search_sort': '',
            'search_time_filter': '',
            'search_subreddit': '',
            'found_in_post': False,
            'found_in_comments': False,
            'matched_keywords': [],
            'broad_search_term': '',
            'search_location': '',
            
            # Daily collection specific fields  
            'tier': '',
            'target_posts': 0,
            'collection_method': ''
        }
        
        # Add missing columns with default values
        for col, default_val in core_columns.items():
            if col not in df.columns:
                # Create column with appropriate type based on default value
                if isinstance(default_val, list):
                    df[col] = [default_val.copy() for _ in range(len(df))]
                else:
                    df[col] = default_val
                logging.debug(f"Added missing column '{col}' with default value")
        
        # Convert data types for consistency
        try:
            # Numeric columns
            df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0).astype(int)
            df['num_comments'] = pd.to_numeric(df['num_comments'], errors='coerce').fillna(0).astype(int)
            df['upvote_ratio'] = pd.to_numeric(df['upvote_ratio'], errors='coerce').fillna(0.0)
            df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce').fillna(0.0)
            df['hours_ago'] = pd.to_numeric(df['hours_ago'], errors='coerce').fillna(0.0)
            df['days_ago'] = pd.to_numeric(df['days_ago'], errors='coerce').fillna(0.0)
            df['comments_collected'] = pd.to_numeric(df['comments_collected'], errors='coerce').fillna(0).astype(int)
            
            # Boolean columns  
            bool_columns = ['is_self', 'is_video', 'stickied', 'over_18', 'spoiler', 'locked', 'found_in_post', 'found_in_comments']
            for col in bool_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(False).astype(bool)
            
            # String columns - fill NaN with empty string
            string_columns = ['title', 'author', 'url', 'permalink', 'selftext', 'domain', 
                            'subreddit', 'flair_text', 'collection_timestamp', 'search_query',
                            'search_sort', 'search_time_filter', 'search_subreddit', 'broad_search_term', 
                            'search_location', 'tier', 'collection_method']
            for col in string_columns:
                if col in df.columns:
                    df[col] = df[col].fillna('').astype(str)
            
            # List columns - ensure they're properly formatted
            list_columns = ['comments', 'matched_keywords'] 
            for col in list_columns:
                if col in df.columns:
                    # Handle missing/empty columns
                    if df[col].empty or df[col].isna().all():
                        df[col] = [[] for _ in range(len(df))]
                    else:
                        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
            
        except Exception as e:
            logging.warning(f"Schema harmonization type conversion failed: {e}")
        
        logging.debug(f"Schema harmonized: {len(df.columns)} columns, {len(df)} rows")
        return df

def get_analysis_settings_for_file(file_path):
    """
    Get optimal analysis settings - now uses consistent balanced approach
    Returns tuple of (engagement_weights, quality_thresholds)
    
    Note: With two-tier filtering, thresholds are less critical since relevant posts auto-pass
    """
    # Use balanced weights suitable for all collection types
    custom_weights = {
        'score': 0.3,
        'upvote_ratio': 0.25,
        'comments': 0.25,
        'comments_quality': 0.15,
        'recency': 0.05
    }
    
    # Use consistent balanced thresholds for all files
    # Two-tier filtering ensures relevant posts (≥60 relevance) bypass these anyway
    quality_thresholds = {
        'min_engagement_score': 30,  # Balanced threshold
        'min_score': 3,               # Lowered to be more inclusive
        'min_comments': 1,            # Lowered to be more inclusive
        'min_upvote_ratio': 0.6,      # Balanced
        'max_days_ago': 14,           # More inclusive time window
    }
    
    logging.info(f"📊 Using consistent analysis settings for: {os.path.basename(file_path)}")
    logging.info(f"   Two-tier filtering: High relevance posts (≥50) auto-pass")
    
    return custom_weights, quality_thresholds

def analyze_reddit_data(file_path, engagement_weights=None, filter_thresholds=None, save_unfiltered=False):
    """
    Main analysis function
    
    Args:
        file_path (str): Path to data file
        engagement_weights (dict): Custom engagement score weights
        filter_thresholds (dict): Filtering thresholds
        save_unfiltered (bool): Save unfiltered results for debugging (default: False)
    """
    try:
        analyzer = RedditDataAnalyzer()
        
        # Load data
        df = analyzer.load_data(file_path)
        if df.empty:
            logging.error("No data loaded")
            return
        
        # Add sentiment analysis
        df_with_sentiment = analyzer.add_sentiment_analysis(df)
        
        # Perform topic modeling
        df_with_topics = analyzer.perform_topic_modeling(df_with_sentiment, n_topics=8)
        
        # APPLY PRIMARY RELEVANCE FILTER FIRST
        df_relevant = analyzer.filter_by_relevance(df_with_topics)
        
        if df_relevant.empty:
            logging.warning("No relevant posts found after relevance filtering")
            return None
        
        # Calculate engagement scores (only on relevant posts)
        df_scored = analyzer.calculate_engagement_score(df_relevant, engagement_weights)
        
        # Apply two-tier filtering system: preserve highly relevant posts
        if filter_thresholds:
            # Tier 1: High relevance posts (50+) - pass with minimal filtering  
            high_relevance = df_scored[df_scored['relevance_score'] >= 50].copy()
            
            # Tier 2: Medium relevance posts (20-49) - apply normal filtering
            medium_relevance = df_scored[df_scored['relevance_score'] < 50].copy()
            medium_filtered = analyzer.filter_by_thresholds(medium_relevance, filter_thresholds) if len(medium_relevance) > 0 else pd.DataFrame()
            
            # Combine both tiers
            if len(high_relevance) > 0 and len(medium_filtered) > 0:
                df_filtered = pd.concat([high_relevance, medium_filtered], ignore_index=True)
            elif len(high_relevance) > 0:
                df_filtered = high_relevance
            elif len(medium_filtered) > 0:
                df_filtered = medium_filtered
            else:
                df_filtered = pd.DataFrame()
            
            # Log the two-tier filtering results
            if len(high_relevance) > 0:
                logging.info(f"🎯 High relevance posts (≥50): {len(high_relevance)} posts auto-passed filtering")
            if len(medium_filtered) > 0:
                logging.info(f"📊 Medium relevance posts (<50): {len(medium_filtered)}/{len(medium_relevance)} posts passed filtering")
        else:
            df_filtered = df_scored
        
        # Apply final sorting to prioritize best content
        df_final = analyzer.apply_final_sorting(df_filtered)
        
        # Generate analysis report
        analyzer.generate_analysis_report(df_final)
        
        # Save results
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Save unfiltered results with all scores for debugging (optional)
        if save_unfiltered:
            analyzer.save_analysis_results(df_scored, f"{base_filename}_unfiltered", save_json=False)
            logging.info(f"💾 Saved unfiltered analysis (all posts with scores) for debugging")
        
        # Save filtered results
        csv_file, json_file = analyzer.save_analysis_results(df_final, base_filename, save_json=False)
        
        logging.info(f"\nAnalysis complete! Results saved to:")
        logging.info(f"- {csv_file}")
        if json_file:
            logging.info(f"- {json_file}")
        
        return df_filtered
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        return None

def analyze_combined_reddit_data(file_paths, engagement_weights=None, filter_thresholds=None, save_unfiltered=False):
    """
    Combine multiple Reddit data files and analyze them together
    
    Args:
        file_paths (list): List of file paths to combine and analyze
        engagement_weights (dict): Custom engagement score weights
        filter_thresholds (dict): Filtering thresholds
        save_unfiltered (bool): Save unfiltered results for debugging (default: False)
    """
    try:
        analyzer = RedditDataAnalyzer()
        combined_df = pd.DataFrame()
        
        # Load and combine all files with schema harmonization
        logging.info(f"Loading and combining {len(file_paths)} files...")
        for file_path in file_paths:
            logging.info(f"Loading: {os.path.basename(file_path)}")
            df = analyzer.load_data(file_path)
            if df is not None and not df.empty:
                # Add source file info and collection type
                df['source_file'] = os.path.basename(file_path)
                
                # Detect collection type from filename/content
                if 'search' in file_path.lower() or 'data_tools' in file_path.lower():
                    df['collection_type'] = 'keyword_search'
                elif 'daily' in file_path.lower() or 'activity_based' in file_path.lower():
                    df['collection_type'] = 'daily_subreddits'
                else:
                    df['collection_type'] = 'unknown'
                
                # Harmonize schema - ensure all DataFrames have the same core columns
                df = analyzer._harmonize_schema(df)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            else:
                logging.warning(f"Skipped empty or invalid file: {file_path}")
        
        if combined_df.empty:
            logging.error("No data loaded from any files")
            return None
            
        # Remove duplicates based on Reddit post ID
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['id'], keep='first')
        duplicate_count = initial_count - len(combined_df)
        if duplicate_count > 0:
            logging.info(f"🔗 Removed {duplicate_count} duplicate posts")
        
        logging.info(f"Combined dataset: {len(combined_df)} unique posts from {len(file_paths)} files")
        
        # Perform sentiment analysis
        logging.info("Performing sentiment analysis on combined data...")
        combined_df = analyzer.add_sentiment_analysis(combined_df)
        
        # Perform topic modeling
        logging.info("Performing topic modeling on combined data...")
        combined_df = analyzer.perform_topic_modeling(combined_df)
        
        # Filter by primary relevance (business focus)
        logging.info("Applying primary relevance filter...")
        combined_df = analyzer.filter_by_relevance(combined_df)
        
        # Calculate engagement scores
        logging.info("Calculating engagement scores...")
        combined_df = analyzer.calculate_engagement_score(combined_df, engagement_weights)
        
        # Keep a copy for unfiltered debug output
        combined_df_scored = combined_df.copy()
        
        # Apply two-tier filtering system: preserve highly relevant posts
        if filter_thresholds:
            # Tier 1: High relevance posts (50+) - pass with minimal filtering
            high_relevance = combined_df[combined_df['relevance_score'] >= 50].copy()
            
            # Tier 2: Medium relevance posts (20-49) - apply normal filtering
            medium_relevance = combined_df[combined_df['relevance_score'] < 50].copy()
            medium_filtered = analyzer.filter_by_thresholds(medium_relevance, filter_thresholds) if len(medium_relevance) > 0 else pd.DataFrame()
            
            # Combine both tiers
            if len(high_relevance) > 0 and len(medium_filtered) > 0:
                combined_df = pd.concat([high_relevance, medium_filtered], ignore_index=True)
            elif len(high_relevance) > 0:
                combined_df = high_relevance
            elif len(medium_filtered) > 0:
                combined_df = medium_filtered
            else:
                combined_df = pd.DataFrame()
            
            # Log the two-tier filtering results
            if len(high_relevance) > 0:
                logging.info(f"🎯 High relevance posts (≥50): {len(high_relevance)} posts auto-passed filtering")
            if len(medium_filtered) > 0:
                logging.info(f"📊 Medium relevance posts (<50): {len(medium_filtered)}/{len(medium_relevance)} posts passed filtering")
        
        # Apply final sorting to prioritize best content
        combined_df = analyzer.apply_final_sorting(combined_df)
        
        # Generate analysis report
        analyzer.generate_analysis_report(combined_df)
        
        # Save results with combined filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = f"reddit_combined_analysis_{timestamp}"
        
        # Save unfiltered results with all scores for debugging (optional)
        if save_unfiltered:
            analyzer.save_analysis_results(combined_df_scored, f"{combined_filename}_unfiltered", save_json=False)
            logging.info(f"💾 Saved unfiltered analysis (all posts with scores) for debugging")
        
        csv_file, json_file = analyzer.save_analysis_results(combined_df, combined_filename, save_json=False)
        
        logging.info(f"\nCombined analysis complete! Results saved to:")
        logging.info(f"- {csv_file}")
        if json_file:
            logging.info(f"- {json_file}")
        
        return combined_df
        
    except Exception as e:
        logging.error(f"Combined analysis failed: {e}")
        return None

if __name__ == "__main__":
    # Example usage - analyze most recent data file
    print("Starting Reddit Data Analyzer...")
    logging.info("Reddit Data Analyzer starting...")
    
    # Check for debug mode
    debug_mode = len(sys.argv) > 1 and '--debug' in sys.argv
    if debug_mode:
        logging.info("🐛 Debug mode enabled - will save unfiltered results")
    
    # Find most recent data files in analysis_results folder (CSV only by default)
    csv_files = glob.glob("analysis_results/*.csv")
    print(f"Found {len(csv_files)} CSV files")
    
    # Filter for Reddit data files (exclude already analyzed files)
    reddit_files = [f for f in csv_files if any(keyword in f.lower() 
                   for keyword in ['reddit', 'data_tools', 'activity_based', 'hybrid_search']) 
                   and '_analyzed_' not in f.lower()]
    
    if reddit_files:
        print(f"Found {len(reddit_files)} Reddit data files:")
        logging.info(f"Found {len(reddit_files)} Reddit data files:")
        for f in sorted(reddit_files, key=os.path.getctime, reverse=True):
            print(f"  - {f}")
            logging.info(f"  - {f}")
        
        # SMART ANALYSIS: Choose approach based on file count and best practices
        if len(reddit_files) == 1:
            # Single file - run individual analysis but with tracking columns
            logging.info("\n🔍 Single file detected - running individual analysis with tracking")
            file_path = reddit_files[0]
            
            # Determine optimal settings for this file type
            custom_weights, quality_thresholds = get_analysis_settings_for_file(file_path)
            
            # Run individual analysis with tracking (uses combined analysis internally for consistency)
            results = analyze_combined_reddit_data(
                file_paths=[file_path],  # Single file in list
                engagement_weights=custom_weights,
                filter_thresholds=quality_thresholds,
                save_unfiltered=debug_mode
            )
            
        else:
            # Multiple files - automatically use combined analysis for optimal results
            logging.info(f"\n🎯 Multiple files detected - running combined analysis for optimal results")
            logging.info("📊 This will harmonize schemas and remove duplicates across all collections")
            
            # Use balanced settings for combined analysis
            custom_weights = {
                'score': 0.3,
                'upvote_ratio': 0.25,
                'comments': 0.25,
                'comments_quality': 0.15,
                'recency': 0.05
            }
            
            quality_thresholds = {
                'min_engagement_score': 30,  # Balanced threshold
                'min_score': 3,
                'min_comments': 1,
                'min_upvote_ratio': 0.6,
                'max_days_ago': 14
            }
            
            # Run combined analysis
            results = analyze_combined_reddit_data(
                file_paths=reddit_files,
                engagement_weights=custom_weights,
                filter_thresholds=quality_thresholds,
                save_unfiltered=debug_mode
            )
        
        # Report results
        if results is not None:
            logging.info("✅ Analysis pipeline completed successfully")
            logging.info("📁 Check analysis_results/ folder for output files")
        else:
            logging.error("❌ Analysis pipeline failed")
        
        logging.info("\n" + "-"*60)
        
    else:
        print("No Reddit data files found in current directory")
        print("Looking for files with keywords: reddit, data_tools, activity_based, hybrid_search")
        print(f"All CSV files: {csv_files}")
        print(f"Available CSV files: {csv_files}")
        logging.warning("No Reddit data files found in current directory")
        logging.info("To analyze specific file, run:")
        logging.info("python reddit_data_analyzer.py")
        logging.info("Or use the analyze_reddit_data() function directly")