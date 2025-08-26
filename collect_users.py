#!/usr/bin/env python3
"""
Collect all comments from specific Reddit users
"""

import praw
import pandas as pd
import logging
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
            logging.info(f"Connected to Reddit API. Read-only: {self.reddit.read_only}")
        except Exception as e:
            logging.error(f"Failed to initialize Reddit connection: {e}")
            raise
    
    def collect_user_comments(self, username, limit=1000, sort_type='new'):
        """
        Collect comments from a specific user profile
        
        Args:
            username (str): Reddit username (without u/)
            limit (int): Number of comments to collect (None for all available)
            sort_type (str): 'new', 'top', 'hot' 
        
        Returns:
            list: List of comment dictionaries
        """
        try:
            user = self.reddit.redditor(username)
            logging.info(f"Collecting comments from u/{username} (limit: {limit})")
            
            comments = []
            
            # Choose sorting method
            if sort_type == 'new':
                submissions = user.comments.new(limit=limit)
            elif sort_type == 'top':
                submissions = user.comments.top(limit=limit)
            elif sort_type == 'hot':
                submissions = user.comments.hot(limit=limit)
            else:
                raise ValueError(f"Invalid sort_type: {sort_type}")
            
            comment_count = 0
            for comment in submissions:
                # Skip deleted/removed comments
                if comment.author is None or comment.body in ['[deleted]', '[removed]']:
                    continue
                
                # Get post title safely
                try:
                    post_title = comment.submission.title
                    post_id = comment.submission.id
                except:
                    post_title = "[Unable to fetch post title]"
                    post_id = "unknown"
                
                comment_data = {
                    'comment_id': comment.id,
                    'body': comment.body,
                    'body_length': len(comment.body),
                    'author': str(comment.author),
                    'score': comment.score,
                    'created_utc': comment.created_utc,
                    'created_datetime': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'hours_ago': (datetime.now(timezone.utc) - datetime.fromtimestamp(comment.created_utc, tz=timezone.utc)).total_seconds() / 3600,
                    'days_ago': (datetime.now(timezone.utc) - datetime.fromtimestamp(comment.created_utc, tz=timezone.utc)).days,
                    'permalink': f"https://reddit.com{comment.permalink}",
                    'subreddit': str(comment.subreddit),
                    'post_title': post_title,
                    'post_id': post_id,
                    'parent_id': comment.parent_id,
                    'is_submitter': comment.is_submitter,
                    'stickied': comment.stickied,
                    'gilded': comment.gilded,
                    'controversiality': comment.controversiality,
                    'edited': comment.edited if comment.edited else False,
                    'collection_timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'sort_type': sort_type
                }
                comments.append(comment_data)
                comment_count += 1
                
                # Log progress every 50 comments
                if comment_count % 50 == 0:
                    logging.info(f"  Collected {comment_count} comments from u/{username}...")
            
            logging.info(f"Successfully collected {len(comments)} comments from u/{username}")
            return comments
            
        except Exception as e:
            logging.error(f"Error collecting comments from u/{username}: {e}")
            return []

def collect_target_users():
    """Collect all comments from the two specified users"""
    
    # Initialize collector
    collector = RedditCollector()
    
    # Target usernames - Replace with actual usernames you want to collect
    target_users = ['target_user1', 'target_user2']
    
    all_comments = []
    user_stats = {}
    
    logging.info("=" * 60)
    logging.info("COLLECTING COMMENTS FROM TARGET USERS")
    logging.info("=" * 60)
    
    for username in target_users:
        try:
            logging.info(f"\n--- Processing u/{username} ---")
            
            # Collect comments (using high limit to get as many as possible)
            user_comments = collector.collect_user_comments(
                username=username, 
                limit=1000,  # Reddit API typically limits to ~1000 anyway
                sort_type='new'
            )
            
            if user_comments:
                # Add username identifier to each comment
                for comment in user_comments:
                    comment['target_username'] = username
                
                all_comments.extend(user_comments)
                
                # Calculate user statistics
                df_user = pd.DataFrame(user_comments)
                user_stats[username] = {
                    'total_comments': len(user_comments),
                    'avg_score': df_user['score'].mean(),
                    'total_score': df_user['score'].sum(),
                    'avg_length': df_user['body_length'].mean(),
                    'unique_subreddits': df_user['subreddit'].nunique(),
                    'top_subreddits': df_user['subreddit'].value_counts().head(5).to_dict(),
                    'oldest_comment_days': df_user['days_ago'].max(),
                    'newest_comment_days': df_user['days_ago'].min()
                }
                
                logging.info(f"✓ u/{username}: {len(user_comments)} comments collected")
            else:
                logging.warning(f"✗ u/{username}: No comments collected")
                user_stats[username] = {'total_comments': 0}
            
            # Be respectful with rate limiting
            time.sleep(3)
            
        except Exception as e:
            logging.error(f"Failed to process u/{username}: {e}")
            user_stats[username] = {'error': str(e)}
    
    # Save all data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if all_comments:
        # Create DataFrame
        df_all = pd.DataFrame(all_comments)
        
        # Save to CSV
        filename_csv = f"reddit_comments_target_users_{timestamp}.csv"
        df_all.to_csv(filename_csv, index=False)
        logging.info(f"Saved {len(all_comments)} total comments to {filename_csv}")
        
        # Save individual user files
        for username in target_users:
            user_comments = [c for c in all_comments if c['target_username'] == username]
            if user_comments:
                df_user = pd.DataFrame(user_comments)
                user_filename = f"reddit_comments_{username}_{timestamp}.csv"
                df_user.to_csv(user_filename, index=False)
                logging.info(f"Saved {len(user_comments)} comments from u/{username} to {user_filename}")
        
        # Print detailed analysis
        print_analysis(df_all, user_stats)
        
    else:
        logging.warning("No comments were collected from any users")
    
    return all_comments, user_stats

def print_analysis(df_all, user_stats):
    """Print detailed analysis of collected comments"""
    
    logging.info("\n" + "=" * 60)
    logging.info("ANALYSIS RESULTS")
    logging.info("=" * 60)
    
    # Overall statistics
    logging.info(f"\n--- OVERALL STATISTICS ---")
    logging.info(f"Total comments collected: {len(df_all)}")
    logging.info(f"Unique subreddits: {df_all['subreddit'].nunique()}")
    logging.info(f"Average comment score: {df_all['score'].mean():.2f}")
    logging.info(f"Average comment length: {df_all['body_length'].mean():.0f} characters")
    logging.info(f"Date range: {df_all['days_ago'].max():.0f} to {df_all['days_ago'].min():.0f} days ago")
    
    # Per-user breakdown
    logging.info(f"\n--- PER-USER BREAKDOWN ---")
    for username, stats in user_stats.items():
        if 'total_comments' in stats and stats['total_comments'] > 0:
            logging.info(f"\nu/{username}:")
            logging.info(f"  Comments: {stats['total_comments']}")
            logging.info(f"  Average score: {stats['avg_score']:.2f}")
            logging.info(f"  Total score: {stats['total_score']}")
            logging.info(f"  Average length: {stats['avg_length']:.0f} chars")
            logging.info(f"  Active in {stats['unique_subreddits']} subreddits")
            logging.info(f"  Top subreddits: {list(stats['top_subreddits'].keys())[:3]}")
            logging.info(f"  Activity span: {stats['oldest_comment_days']:.0f} days")
        else:
            logging.info(f"\nu/{username}: No data collected")
    
    # Top subreddits overall
    logging.info(f"\n--- TOP SUBREDDITS (ALL USERS) ---")
    top_subs = df_all['subreddit'].value_counts().head(10)
    for sub, count in top_subs.items():
        logging.info(f"  r/{sub}: {count} comments")
    
    # Recent vs older comments
    recent_comments = df_all[df_all['days_ago'] <= 30]  # Last 30 days
    logging.info(f"\n--- RECENT ACTIVITY (LAST 30 DAYS) ---")
    logging.info(f"Recent comments: {len(recent_comments)} of {len(df_all)} total")
    if len(recent_comments) > 0:
        logging.info(f"Recent average score: {recent_comments['score'].mean():.2f}")

if __name__ == "__main__":
    # Collect comments from the target users
    comments, stats = collect_target_users()
    
    logging.info("\n" + "=" * 60)
    logging.info("COLLECTION COMPLETE!")
    logging.info("=" * 60)