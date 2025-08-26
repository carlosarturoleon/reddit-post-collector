#!/usr/bin/env python3
"""
Daily Reddit Data Collector using PRAW
Collects newest posts from specified subreddits for analysis
"""

import praw
import pandas as pd
import logging
import json
import csv
import os
from datetime import datetime, timezone
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_collector.log'),
        logging.StreamHandler()
    ]
)

class RedditCollector:
    def __init__(self, config_file='praw.ini'):
        """Initialize Reddit collector with PRAW"""
        try:
            self.reddit = praw.Reddit()  # Reads from praw.ini
            # Test connection
            logging.info(f"Connected to Reddit API. Read-only: {self.reddit.read_only}")
        except Exception as e:
            logging.error(f"Failed to initialize Reddit connection: {e}")
            raise
    
    def collect_comments(self, submission, max_comments=200):
        """
        Collect top comments from a submission
        
        Args:
            submission: PRAW submission object
            max_comments (int): Maximum number of comments to collect (default 200)
            
        Returns:
            list: List of comment dictionaries sorted by score (best first)
        """
        try:
            submission.comment_sort = 'best'  # Sort comments by best/top
            submission.comments.replace_more(limit=0)  # Remove "more comments" objects
            comments = []
            
            # Get all comments and sort by score
            all_comments = submission.comments.list()
            
            # Filter out deleted comments and sort by score
            valid_comments = []
            for comment in all_comments:
                if hasattr(comment, 'body') and comment.body != '[deleted]' and hasattr(comment, 'score'):
                    valid_comments.append(comment)
            
            # Sort by score (descending) to get top comments first
            valid_comments.sort(key=lambda x: x.score, reverse=True)
            
            # Take top comments up to max_comments limit
            for comment in valid_comments[:max_comments]:
                comment_data = {
                    'comment_id': comment.id,
                    'parent_id': comment.parent_id,
                    'author': str(comment.author) if comment.author else '[deleted]',
                    'body': comment.body,
                    'score': comment.score,
                    'created_utc': comment.created_utc,
                    'created_datetime': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc),
                    'is_submitter': comment.is_submitter,
                    'stickied': comment.stickied,
                    'depth': getattr(comment, 'depth', 0)
                }
                comments.append(comment_data)
            
            return comments
            
        except Exception as e:
            logging.warning(f"Error collecting comments for post {submission.id}: {e}")
            return []

    def collect_subreddit_posts(self, subreddit_name, limit=100, sort_type='new', time_filter='day', collect_comments=False, max_comments=200):
        """
        Collect posts from a subreddit
        
        Args:
            subreddit_name (str): Name of subreddit (without r/)
            limit (int): Number of posts to collect (max ~1000)
            sort_type (str): 'new', 'hot', 'top', 'rising'
            time_filter (str): 'hour', 'day', 'week', 'month', 'year', 'all' (only for 'top')
            collect_comments (bool): Whether to collect comments for each post
            max_comments (int): Maximum number of comments to collect per post
        
        Returns:
            list: List of post dictionaries with optional comments
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            if sort_type == 'top':
                logging.info(f"Collecting {limit} {sort_type} posts from r/{subreddit_name} (time_filter: {time_filter})")
            else:
                logging.info(f"Collecting {limit} {sort_type} posts from r/{subreddit_name}")
            
            posts = []
            
            # Choose sorting method
            if sort_type == 'new':
                submissions = subreddit.new(limit=limit)
            elif sort_type == 'hot':
                submissions = subreddit.hot(limit=limit)
            elif sort_type == 'top':
                submissions = subreddit.top(limit=limit, time_filter=time_filter)
            elif sort_type == 'rising':
                submissions = subreddit.rising(limit=limit)
            else:
                raise ValueError(f"Invalid sort_type: {sort_type}")
            
            for submission in submissions:
                post_data = {
                    'id': submission.id,
                    'title': submission.title,
                    'author': str(submission.author) if submission.author else '[deleted]',
                    'score': submission.score,
                    'upvote_ratio': submission.upvote_ratio,
                    'num_comments': submission.num_comments,
                    'created_utc': submission.created_utc,
                    'created_datetime': datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                    'hours_ago': (datetime.now(timezone.utc) - datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)).total_seconds() / 3600,
                    'url': submission.url,
                    'permalink': f"https://reddit.com{submission.permalink}",
                    'selftext': submission.selftext,
                    'is_self': submission.is_self,
                    'domain': submission.domain,
                    'subreddit': subreddit_name,
                    'flair_text': submission.link_flair_text,
                    'is_video': submission.is_video,
                    'stickied': submission.stickied,
                    'over_18': submission.over_18,
                    'spoiler': submission.spoiler,
                    'locked': submission.locked,
                    'collection_timestamp': datetime.now(timezone.utc),
                    'sort_type': sort_type,
                    'time_filter': time_filter if sort_type == 'top' else None
                }
                
                # Collect comments if requested
                if collect_comments:
                    post_data['comments'] = self.collect_comments(submission, max_comments)
                    post_data['comments_collected'] = len(post_data['comments'])
                else:
                    post_data['comments'] = []
                    post_data['comments_collected'] = 0
                
                posts.append(post_data)
            
            logging.info(f"Successfully collected {len(posts)} posts from r/{subreddit_name}")
            return posts
            
        except Exception as e:
            logging.error(f"Error collecting from r/{subreddit_name}: {e}")
            return []
    
    def filter_posts_by_age(self, posts, max_hours=72):
        """
        Filter posts by age in hours (for low activity subreddits)
        
        Args:
            posts (list): List of post dictionaries
            max_hours (int): Maximum age in hours
            
        Returns:
            list: Filtered posts within the time limit
        """
        filtered = []
        for post in posts:
            if post['hours_ago'] <= max_hours:
                filtered.append(post)
        
        return filtered
    
    def collect_activity_based(self, subreddit_config, collect_comments=False, max_comments=200):
        """
        Collect posts using activity-based strategy with error recovery
        
        Args:
            subreddit_config (dict): Configuration for different activity tiers
            collect_comments (bool): Whether to collect comments for each post
            max_comments (int): Maximum number of comments to collect per post
            
        Returns:
            list: All collected posts with tier information
        """
        all_posts = []
        collection_summary = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for tier_name, config in subreddit_config.items():
            tier_posts = []
            subreddits = config['subreddits']
            posts_per_sub = config['posts_per_sub']
            time_filter = config['time_filter']
            description = config['description']
            
            logging.info(f"\n=== {tier_name.upper().replace('_', ' ')} ({description}) ===")
            logging.info(f"Collecting {posts_per_sub} posts per subreddit, time_filter: {time_filter}")
            
            for subreddit in subreddits:
                try:
                    logging.info(f"Processing r/{subreddit}...")
                    posts = self.collect_subreddit_posts(
                        subreddit, 
                        limit=posts_per_sub, 
                        sort_type='top', 
                        time_filter=time_filter,
                        collect_comments=collect_comments,
                        max_comments=max_comments
                    )
                    
                    # Add tier information to each post
                    for post in posts:
                        post['activity_tier'] = tier_name
                        post['tier_description'] = description
                        post['target_posts'] = posts_per_sub
                    
                    tier_posts.extend(posts)
                    all_posts.extend(posts)  # Add to main collection immediately
                    
                    # Save progress after each subreddit
                    if posts:
                        self.save_progress(all_posts, f"progress_{timestamp}")
                        logging.info(f"✓ Progress saved: {len(all_posts)} total posts collected")
                    
                    time.sleep(1)  # Be respectful between requests
                    
                except Exception as e:
                    logging.error(f"Failed to collect from r/{subreddit}: {e}")
                    logging.info("Continuing with next subreddit...")
            
            logging.info(f"{tier_name.replace('_', ' ').title()}: {len(tier_posts)} posts collected")
            collection_summary.append({
                'tier': tier_name,
                'description': description,
                'subreddits': len(subreddits),
                'target_total': len(subreddits) * posts_per_sub,
                'actual_total': len(tier_posts)
            })
            
            # Posts already added to all_posts in the inner loop (line 231)
            # Removed duplicate: all_posts.extend(tier_posts)
        
        # Log collection summary
        logging.info(f"\n=== COLLECTION SUMMARY ===")
        total_target = 0
        total_actual = 0
        for summary in collection_summary:
            logging.info(f"{summary['tier'].replace('_', ' ').title()}: "
                        f"{summary['actual_total']}/{summary['target_total']} posts "
                        f"({summary['subreddits']} subreddits)")
            total_target += summary['target_total']
            total_actual += summary['actual_total']
        
        logging.info(f"TOTAL: {total_actual}/{total_target} posts collected")
        
        return all_posts
    
    def save_to_csv(self, posts, filename=None):
        """Save posts to CSV file in analysis_results folder with proper comments serialization"""
        if not posts:
            logging.warning("No posts to save")
            return
        
        # Create analysis_results folder if it doesn't exist
        output_folder = "analysis_results"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_posts_{timestamp}.csv"
        
        # Save to analysis_results folder
        filepath = os.path.join(output_folder, filename)
        
        # Prepare posts for CSV by serializing complex data
        csv_posts = []
        for post in posts:
            csv_post = post.copy()
            
            # Convert datetime objects to strings
            for key, value in csv_post.items():
                if isinstance(value, datetime):
                    csv_post[key] = value.isoformat()
                elif key == 'comments' and isinstance(value, list):
                    # Serialize comments as JSON string to prevent CSV corruption
                    # Use separators to minimize whitespace and ensure_ascii to avoid encoding issues
                    csv_post[key] = json.dumps(value, default=str, separators=(',', ':'), ensure_ascii=True)
            
            csv_posts.append(csv_post)
        
        df = pd.DataFrame(csv_posts)
        df.to_csv(filepath, index=False, escapechar='\\', quoting=csv.QUOTE_NONNUMERIC)
        logging.info(f"Saved {len(posts)} posts to {filepath}")
    
    def save_to_json(self, posts, filename=None):
        """Save posts to JSON file in analysis_results folder"""
        if not posts:
            logging.warning("No posts to save")
            return
        
        # Create analysis_results folder if it doesn't exist
        output_folder = "analysis_results"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_posts_{timestamp}.json"
        
        # Save to analysis_results folder
        filepath = os.path.join(output_folder, filename)
        
        # Convert datetime objects to strings for JSON serialization
        json_posts = []
        for post in posts:
            json_post = post.copy()
            for key, value in json_post.items():
                if isinstance(value, datetime):
                    json_post[key] = value.isoformat()
                elif key == 'comments' and isinstance(value, list):
                    # Handle datetime objects in comments
                    json_comments = []
                    for comment in value:
                        json_comment = comment.copy()
                        for comment_key, comment_value in json_comment.items():
                            if isinstance(comment_value, datetime):
                                json_comment[comment_key] = comment_value.isoformat()
                        json_comments.append(json_comment)
                    json_post[key] = json_comments
            json_posts.append(json_post)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_posts, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved {len(posts)} posts to {filepath}")

    def save_progress(self, posts, filename_prefix, save_json=False):
        """Save progress to CSV (and optionally JSON) to prevent data loss in analysis_results folder"""
        if not posts:
            return
        
        try:
            # Create analysis_results folder if it doesn't exist
            output_folder = "analysis_results"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            # Save CSV (with proper comments serialization)
            csv_filename = os.path.join(output_folder, f"{filename_prefix}.csv")
            csv_posts = []
            for post in posts:
                csv_post = post.copy()
                for key, value in csv_post.items():
                    if isinstance(value, datetime):
                        csv_post[key] = value.isoformat()
                    elif key == 'comments' and isinstance(value, list):
                        csv_post[key] = json.dumps(value, default=str, separators=(',', ':'), ensure_ascii=True)
                csv_posts.append(csv_post)
            
            df = pd.DataFrame(csv_posts)
            df.to_csv(csv_filename, index=False, escapechar='\\', quoting=csv.QUOTE_NONNUMERIC)
            
            # Save JSON (optional, disabled by default)
            if save_json:
                json_filename = os.path.join(output_folder, f"{filename_prefix}.json")
                json_posts = []
                for post in posts:
                    json_post = post.copy()
                    for key, value in json_post.items():
                        if isinstance(value, datetime):
                            json_post[key] = value.isoformat()
                        elif key == 'comments' and isinstance(value, list):
                            json_comments = []
                            for comment in value:
                                json_comment = comment.copy()
                                for comment_key, comment_value in json_comment.items():
                                    if isinstance(comment_value, datetime):
                                        json_comment[comment_key] = comment_value.isoformat()
                                json_comments.append(json_comment)
                            json_post[key] = json_comments
                    json_posts.append(json_post)
                
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(json_posts, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logging.warning(f"Error saving progress: {e}")
    
    def cleanup_progress_files(self):
        """Clean up progress files after successful final save"""
        try:
            output_folder = "analysis_results"
            if not os.path.exists(output_folder):
                return
            
            # Find and delete progress files
            import glob
            progress_files = glob.glob(os.path.join(output_folder, "progress_*.csv")) + \
                           glob.glob(os.path.join(output_folder, "progress_*.json"))
            
            deleted_count = 0
            for progress_file in progress_files:
                try:
                    os.remove(progress_file)
                    deleted_count += 1
                    logging.info(f"🗑️  Cleaned up progress file: {os.path.basename(progress_file)}")
                except Exception as e:
                    logging.warning(f"Could not delete progress file {progress_file}: {e}")
            
            if deleted_count > 0:
                logging.info(f"✨ Cleaned up {deleted_count} progress files")
            
        except Exception as e:
            logging.warning(f"Error during progress file cleanup: {e}")

def daily_collection(collect_comments=False, max_comments=200):
    """Main function for daily data collection with activity-based strategy
    
    Args:
        collect_comments (bool): Whether to collect comments for each post
        max_comments (int): Maximum number of comments to collect per post
    """
    
    # Activity-based subreddit configuration
    SUBREDDIT_CONFIG = {
        # Tier 1: High Activity (1M+ members) - 150 posts, 24 hours
        'high_activity': {
            'subreddits': [
                'datascience', 
                'marketing', 
                'startups', 
                'Entrepreneur'
                ],
            'posts_per_sub': 150,
            'time_filter': 'week',  
            'description': '1M+ members'
        },
        
        # Tier 2: Medium Activity (100K-1M members) - DISABLED FOR TEST
        'medium_activity': {
            'subreddits': [
                'shopify', 'digital_marketing', 'PPC',
                'analytics', 'BusinessIntelligence', 'PowerBI', 'FacebookAds',
                'programacion', 'dataanalysis','AskMarketing','ProductManagement',
                'SaaS','ecommerce'   
            ],
            'posts_per_sub': 100,
            'time_filter': 'week', 
            'description': '100K-1M members'
        },
        
        # Tier 3: Low Activity - DISABLED FOR TEST
        'low_activity': {
            'subreddits': [
                'tableau', 'GoogleAds', 'MarketingAutomation', 'GoogleAnalytics',
                'bigquery', 'snowflake', 'GoogleDataStudio', 'databricks',
                'nocode', 'devsarg', 'taquerosprogramadores', 'dataanalyst',             
                'chileIT', 'automation','datavisualization','Dynamics365',
                'googlecloud','shopifyDev','roastmystartup','agency'               
            ],
            'posts_per_sub': 50,
            'time_filter': 'week',
            'description': '10K-100K members'
        },
        
        # Tier 4: Very Low Activity - DISABLED FOR TEST
        'very_low_activity': {
            'subreddits': [
                'Looker', 'LinkedInAds', 'LookerStudio', 'Fivetran', 'datawarehouse', 'ETL',
                'ColombiaDevs' 
            ],
            'posts_per_sub': 25,
            'time_filter': 'week',
            'description': '<10K members'
        }
    }
    
    SORT_TYPE = 'top'  # Use top posts with time filters
    
    try:
        # Initialize collector
        collector = RedditCollector()
        
        # Collect data using activity-based strategy
        logging.info("Starting activity-based Reddit data collection")
        logging.info("Strategy: More posts from active subreddits, longer timeframes for less active ones")
        
        posts = collector.collect_activity_based(SUBREDDIT_CONFIG, collect_comments, max_comments)
        
        if posts:
            # Save data with activity-based naming
            timestamp = datetime.now().strftime("%Y%m%d")
            collector.save_to_csv(posts, f"reddit_activity_based_{timestamp}.csv")
            # JSON output disabled by default
            # collector.save_to_json(posts, f"reddit_activity_based_{timestamp}.json")
            
            # Clean up progress files after successful final save
            collector.cleanup_progress_files()
            
            # Enhanced analysis for activity-based collection
            df = pd.DataFrame(posts)
            logging.info(f"\n=== FINAL ANALYSIS ===")
            logging.info(f"- Total posts collected: {len(df)}")
            logging.info(f"- Unique subreddits: {df['subreddit'].nunique()}")
            logging.info(f"- Average score: {df['score'].mean():.2f}")
            logging.info(f"- Average age (hours): {df['hours_ago'].mean():.2f}")
            
            # Analysis by activity tier
            logging.info(f"\n--- Posts by Activity Tier ---")
            tier_analysis = df.groupby('activity_tier').agg({
                'id': 'count',
                'score': 'mean',
                'hours_ago': 'mean',
                'subreddit': 'nunique'
            }).round(2)
            tier_analysis.columns = ['posts', 'avg_score', 'avg_hours_ago', 'subreddits']
            logging.info(f"\n{tier_analysis}")
            
            # Top performing posts by tier
            logging.info(f"\n--- Top Posts by Tier ---")
            for tier in df['activity_tier'].unique():
                tier_posts = df[df['activity_tier'] == tier].nlargest(3, 'score')
                logging.info(f"\n{tier.replace('_', ' ').title()}:")
                for _, post in tier_posts.iterrows():
                    logging.info(f"  • r/{post['subreddit']}: {post['title'][:50]}... "
                                f"(score: {post['score']}, {post['hours_ago']:.1f}h ago)")
            
            # Subreddit performance summary
            logging.info(f"\n--- Posts per Subreddit ---")
            subreddit_counts = df['subreddit'].value_counts()
            logging.info(f"\n{subreddit_counts}")
            
        else:
            logging.warning("No posts collected")
            
    except Exception as e:
        logging.error(f"Daily collection failed: {e}")

if __name__ == "__main__":
    # To collect posts WITHOUT comments (default):
    # daily_collection()
    
    # To collect posts WITH top 200 comments per post:
    daily_collection(collect_comments=True, max_comments=200)
    
    # To collect posts WITH top 50 comments per post:  
    # daily_collection(collect_comments=True, max_comments=50)