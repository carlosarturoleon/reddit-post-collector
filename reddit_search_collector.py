#!/usr/bin/env python3
"""
Reddit Search Collector using PRAW
Search for posts by keywords across Reddit or specific subreddits
"""

import praw
import pandas as pd
import logging
import json
from datetime import datetime, timezone
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_search.log'),
        logging.StreamHandler()
    ]
)

class RedditSearchCollector:
    def __init__(self, config_file='praw.ini'):
        """Initialize Reddit search collector with PRAW"""
        try:
            self.reddit = praw.Reddit()  # Reads from praw.ini
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
            submission.comment_sort = 'best'
            submission.comments.replace_more(limit=0)
            comments = []
            
            all_comments = submission.comments.list()
            
            valid_comments = []
            for comment in all_comments:
                if hasattr(comment, 'body') and comment.body != '[deleted]' and hasattr(comment, 'score'):
                    # Filter comments older than 2 weeks (14 days)
                    comment_age_hours = (datetime.now(timezone.utc) - datetime.fromtimestamp(comment.created_utc, tz=timezone.utc)).total_seconds() / 3600
                    if comment_age_hours <= 14 * 24:  # 14 days * 24 hours
                        valid_comments.append(comment)
            
            valid_comments.sort(key=lambda x: x.score, reverse=True)
            
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

    def search_reddit(self, query, sort='relevance', time_filter='all', limit=100, 
                      subreddit=None, collect_comments=False, max_comments=200):
        """
        Search Reddit for posts matching a query (max 2 weeks old)
        
        Args:
            query (str): Search query/keywords
            sort (str): 'relevance', 'hot', 'top', 'new', 'comments'
            time_filter (str): 'all', 'day', 'week', 'month', 'year' (for 'top' sort)
            limit (int): Number of results to return (max ~1000)
            subreddit (str): Optional - search within specific subreddit
            collect_comments (bool): Whether to collect comments for each post
            max_comments (int): Maximum number of comments to collect per post
            
        Returns:
            list: List of post dictionaries matching the search (max 2 weeks old)
        """
        try:
            if subreddit:
                search_target = self.reddit.subreddit(subreddit)
                logging.info(f"Searching r/{subreddit} for '{query}' (sort: {sort}, limit: {limit})")
            else:
                search_target = self.reddit.subreddit('all')
                logging.info(f"Searching all Reddit for '{query}' (sort: {sort}, limit: {limit})")
            
            if sort == 'top':
                submissions = search_target.search(query, sort=sort, time_filter=time_filter, limit=limit)
            else:
                submissions = search_target.search(query, sort=sort, limit=limit)
            
            posts = []
            for submission in submissions:
                # Filter posts older than 2 weeks (14 days)
                post_age_hours = (datetime.now(timezone.utc) - datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)).total_seconds() / 3600
                
                if post_age_hours <= 14 * 24:  # 14 days * 24 hours
                    post_data = {
                        'id': submission.id,
                        'title': submission.title,
                        'author': str(submission.author) if submission.author else '[deleted]',
                        'score': submission.score,
                        'upvote_ratio': submission.upvote_ratio,
                        'num_comments': submission.num_comments,
                        'created_utc': submission.created_utc,
                        'created_datetime': datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                        'hours_ago': post_age_hours,
                        'days_ago': post_age_hours / 24,
                        'url': submission.url,
                        'permalink': f"https://reddit.com{submission.permalink}",
                        'selftext': submission.selftext,
                        'is_self': submission.is_self,
                        'domain': submission.domain,
                        'subreddit': str(submission.subreddit),
                        'flair_text': submission.link_flair_text,
                        'is_video': submission.is_video,
                        'stickied': submission.stickied,
                        'over_18': submission.over_18,
                        'spoiler': submission.spoiler,
                        'locked': submission.locked,
                        'collection_timestamp': datetime.now(timezone.utc),
                        'search_query': query,
                        'search_sort': sort,
                        'search_time_filter': time_filter if sort == 'top' else None,
                        'search_subreddit': subreddit
                    }
                    
                    if collect_comments:
                        post_data['comments'] = self.collect_comments(submission, max_comments)
                        post_data['comments_collected'] = len(post_data['comments'])
                    else:
                        post_data['comments'] = []
                        post_data['comments_collected'] = 0
                    
                    posts.append(post_data)
                    time.sleep(0.5)
            
            logging.info(f"Successfully collected {len(posts)} posts (max 2 weeks old) for query '{query}'")
            return posts
            
        except Exception as e:
            logging.error(f"Error searching for '{query}': {e}")
            return []

    def multi_keyword_search(self, keywords, sort='relevance', time_filter='all', 
                           limit=100, subreddit=None, collect_comments=False, max_comments=200):
        """
        Search for multiple keywords and combine results
        
        Args:
            keywords (list): List of search terms
            sort (str): 'relevance', 'hot', 'top', 'new', 'comments'
            time_filter (str): 'all', 'day', 'week', 'month', 'year'
            limit (int): Number of results per keyword
            subreddit (str): Optional - search within specific subreddit
            collect_comments (bool): Whether to collect comments for each post
            max_comments (int): Maximum number of comments to collect per post
            
        Returns:
            list: Combined list of posts from all keyword searches (max 2 weeks old)
        """
        all_posts = []
        seen_ids = set()
        
        logging.info(f"Starting multi-keyword search for: {keywords}")
        
        for keyword in keywords:
            posts = self.search_reddit(
                query=keyword,
                sort=sort,
                time_filter=time_filter,
                limit=limit,
                subreddit=subreddit,
                collect_comments=collect_comments,
                max_comments=max_comments
            )
            
            unique_posts = []
            for post in posts:
                if post['id'] not in seen_ids:
                    seen_ids.add(post['id'])
                    unique_posts.append(post)
            
            all_posts.extend(unique_posts)
            logging.info(f"Keyword '{keyword}': {len(unique_posts)} unique posts added")
            time.sleep(1)
        
        logging.info(f"Total unique posts collected: {len(all_posts)}")
        return all_posts

    def save_to_csv(self, posts, filename=None):
        """Save posts to CSV file"""
        if not posts:
            logging.warning("No posts to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_search_{timestamp}.csv"
        
        df = pd.DataFrame(posts)
        df.to_csv(filename, index=False)
        logging.info(f"Saved {len(posts)} posts to {filename}")
    
    def save_to_json(self, posts, filename=None):
        """Save posts to JSON file"""
        if not posts:
            logging.warning("No posts to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_search_{timestamp}.json"
        
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
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_posts, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved {len(posts)} posts to {filename}")

def data_tools_search():
    """Search for data engineering and analytics tools"""
    try:
        collector = RedditSearchCollector()
        
        # Your specific keywords
        keywords = ["airbyte", "bigquery", "elt", "etl", "fivetran", "supermetrics", "windsor.ai"]
        
        # Search across all Reddit
        posts = collector.multi_keyword_search(
            keywords=keywords,
            sort="relevance",
            limit=50,  # 50 posts per keyword
            collect_comments=True,
            max_comments=200
        )
        
        if posts:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            collector.save_to_csv(posts, f"data_tools_search_{timestamp}.csv")
            collector.save_to_json(posts, f"data_tools_search_{timestamp}.json")
            
            df = pd.DataFrame(posts)
            logging.info(f"\n=== DATA TOOLS SEARCH RESULTS ===")
            logging.info(f"- Total posts: {len(df)}")
            logging.info(f"- Unique subreddits: {df['subreddit'].nunique()}")
            logging.info(f"- Average score: {df['score'].mean():.2f}")
            logging.info(f"- Average age (days): {df['days_ago'].mean():.1f}")
            logging.info(f"- Posts with comments: {df[df['comments_collected'] > 0].shape[0]}")
            
            # Results by keyword
            logging.info(f"\n--- Posts by Keyword ---")
            keyword_counts = df['search_query'].value_counts()
            logging.info(f"\n{keyword_counts}")
            
            # Top subreddits
            logging.info(f"\n--- Top Subreddits ---")
            logging.info(f"\n{df['subreddit'].value_counts().head(10)}")
            
        else:
            logging.warning("No posts found")
            
    except Exception as e:
        logging.error(f"Data tools search failed: {e}")

if __name__ == "__main__":
    data_tools_search()