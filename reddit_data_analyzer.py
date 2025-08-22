#!/usr/bin/env python3
"""
Reddit Data Analyzer
Analyze collected Reddit data with engagement scoring and filtering
"""

import pandas as pd
import numpy as np
import logging
import json
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
                df = pd.read_csv(file_path)
                logging.info(f"Loaded {len(df)} posts from CSV: {file_path}")
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
            'python', 'windsor.ai'
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
        
        # Filter: Keep posts with relevance score >= 20
        relevance_threshold = 20
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
        
        filtered_count = len(df_filtered)
        logging.info(f"Filtering complete: {filtered_count}/{original_count} posts passed thresholds ({filtered_count/original_count*100:.1f}%)")
        
        return df_filtered
    
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
    
    def save_analysis_results(self, df, filename_prefix):
        """Save analyzed data to CSV and JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_filename = f"{filename_prefix}_analyzed_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        logging.info(f"Saved analyzed data to: {csv_filename}")
        
        # Save JSON
        json_filename = f"{filename_prefix}_analyzed_{timestamp}.json"
        df_json = df.copy()
        
        # Convert datetime columns to strings
        for col in df_json.columns:
            if df_json[col].dtype == 'datetime64[ns]':
                df_json[col] = df_json[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        df_json.to_json(json_filename, indent=2, orient='records')
        logging.info(f"Saved analyzed data to: {json_filename}")
        
        return csv_filename, json_filename

def analyze_reddit_data(file_path, engagement_weights=None, filter_thresholds=None):
    """
    Main analysis function
    
    Args:
        file_path (str): Path to data file
        engagement_weights (dict): Custom engagement score weights
        filter_thresholds (dict): Filtering thresholds
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
        
        # Apply secondary filters if specified
        if filter_thresholds:
            df_filtered = analyzer.filter_by_thresholds(df_scored, filter_thresholds)
        else:
            df_filtered = df_scored
        
        # Generate analysis report
        analyzer.generate_analysis_report(df_filtered)
        
        # Save results
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        csv_file, json_file = analyzer.save_analysis_results(df_filtered, base_filename)
        
        logging.info(f"\nAnalysis complete! Results saved to:")
        logging.info(f"- {csv_file}")
        logging.info(f"- {json_file}")
        
        return df_filtered
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        return None

if __name__ == "__main__":
    # Example usage - analyze most recent data file
    print("Starting Reddit Data Analyzer...")
    logging.info("Reddit Data Analyzer starting...")
    
    # Find most recent data files
    csv_files = glob.glob("*.csv")
    json_files = glob.glob("*.json")
    print(f"Found {len(csv_files)} CSV files and {len(json_files)} JSON files")
    
    # Filter for Reddit data files
    reddit_files = [f for f in csv_files + json_files if any(keyword in f.lower() 
                   for keyword in ['reddit', 'data_tools', 'activity_based', 'hybrid_search'])]
    
    if reddit_files:
        print(f"Found {len(reddit_files)} Reddit data files:")
        logging.info(f"Found {len(reddit_files)} Reddit data files:")
        for f in sorted(reddit_files, key=os.path.getctime, reverse=True):
            print(f"  - {f}")
            logging.info(f"  - {f}")
        
        # Analyze multiple files or just the latest
        files_to_analyze = reddit_files[:3]  # Analyze up to 3 most recent files
        
        for file_path in sorted(files_to_analyze, key=os.path.getctime, reverse=True):
            logging.info(f"\n{'='*60}")
            logging.info(f"ANALYZING: {file_path}")
            logging.info(f"{'='*60}")
            
            # Adjust weights based on file type
            if 'activity_based' in file_path.lower():
                # Activity-based collection - focus more on engagement
                custom_weights = {
                    'score': 0.35,          
                    'upvote_ratio': 0.15,    
                    'comments': 0.3,        # Higher weight for comments
                    'comments_quality': 0.15,
                    'recency': 0.05         # Lower weight for recency (systematic collection)
                }
                quality_thresholds = {
                    'min_engagement_score': 25,    
                    'min_score': 3,                
                    'min_comments': 1,             
                    'min_upvote_ratio': 0.55,      
                    'max_days_ago': 14             
                }
            elif 'hybrid_search' in file_path.lower() or 'search' in file_path.lower():
                # Search-based collection - focus on relevance and quality
                custom_weights = {
                    'score': 0.3,          
                    'upvote_ratio': 0.25,   # Higher weight for community approval
                    'comments': 0.25,       
                    'comments_quality': 0.15,
                    'recency': 0.05         
                }
                quality_thresholds = {
                    'min_engagement_score': 35,    # Higher threshold for search results
                    'min_score': 5,                
                    'min_comments': 2,             
                    'min_upvote_ratio': 0.65,      # Higher threshold
                    'max_days_ago': 10             
                }
            else:
                # Default weights
                custom_weights = {
                    'score': 0.35,          
                    'upvote_ratio': 0.2,    
                    'comments': 0.25,       
                    'comments_quality': 0.1,
                    'recency': 0.1          
                }
                quality_thresholds = {
                    'min_engagement_score': 30,    
                    'min_score': 5,                
                    'min_comments': 2,             
                    'min_upvote_ratio': 0.6,       
                    'max_days_ago': 10             
                }
            
            # Run analysis
            results = analyze_reddit_data(
                file_path=file_path,
                engagement_weights=custom_weights,
                filter_thresholds=quality_thresholds
            )
            
            if results is not None:
                logging.info(f"✓ Analysis complete for {file_path}")
            else:
                logging.error(f"✗ Analysis failed for {file_path}")
            
            logging.info("\n" + "-"*60)
        
    else:
        print("No Reddit data files found in current directory")
        print("Looking for files with keywords: reddit, data_tools, activity_based, hybrid_search")
        print(f"All CSV files: {csv_files}")
        print(f"All JSON files: {json_files}")
        logging.warning("No Reddit data files found in current directory")
        logging.info("To analyze specific file, run:")
        logging.info("python reddit_data_analyzer.py")
        logging.info("Or use the analyze_reddit_data() function directly")