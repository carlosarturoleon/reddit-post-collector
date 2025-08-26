#!/usr/bin/env python3
"""
Reddit Search Collector using PRAW
Search for posts by keywords across Reddit or specific subreddits
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
            filename = f"reddit_search_{timestamp}.csv"
        
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
            filename = f"reddit_search_{timestamp}.json"
        
        # Save to analysis_results folder
        filepath = os.path.join(output_folder, filename)
        
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

    def cleanup_progress_files(self):
        """Clean up any temporary/progress files after successful save"""
        try:
            output_folder = "analysis_results"
            if not os.path.exists(output_folder):
                return
            
            # Find and delete any temporary search files
            import glob
            temp_files = glob.glob(os.path.join(output_folder, "*_temp_*.csv")) + \
                        glob.glob(os.path.join(output_folder, "*_temp_*.json"))
            
            deleted_count = 0
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    deleted_count += 1
                    logging.info(f"🗑️  Cleaned up temp file: {os.path.basename(temp_file)}")
                except Exception as e:
                    logging.warning(f"Could not delete temp file {temp_file}: {e}")
            
            if deleted_count > 0:
                logging.info(f"✨ Cleaned up {deleted_count} temporary files")
                
        except Exception as e:
            logging.warning(f"Error during temp file cleanup: {e}")

    def hybrid_search(self, target_keywords, broad_terms, relevant_subreddits=None, 
                     limit_per_term=100, collect_comments=True, max_comments=200):
        """
        Hybrid search: broad terms + keyword filtering for wider coverage
        
        Args:
            target_keywords (list): Specific keywords to find in posts/comments
            broad_terms (list): Broader search terms to cast a wider net
            relevant_subreddits (list): Optional list of relevant subreddits to search
            limit_per_term (int): Posts to collect per broad term
            collect_comments (bool): Whether to collect comments
            max_comments (int): Max comments per post
            
        Returns:
            list: Posts containing target keywords in title/content/comments
        """
        all_matches = []
        seen_ids = set()
        target_keywords_lower = [kw.lower() for kw in target_keywords]
        
        logging.info(f"Starting hybrid search:")
        logging.info(f"  Target keywords: {target_keywords}")
        logging.info(f"  Broad search terms: {broad_terms}")
        logging.info(f"  Relevant subreddits: {relevant_subreddits or 'All Reddit'}")
        
        # Search locations: specific subreddits + all Reddit
        search_locations = []
        if relevant_subreddits:
            search_locations.extend([(sub, f"r/{sub}") for sub in relevant_subreddits])
        search_locations.append((None, "all Reddit"))
        
        for subreddit, location_name in search_locations:
            for broad_term in broad_terms:
                try:
                    logging.info(f"Searching {location_name} for '{broad_term}'...")
                    
                    posts = self.search_reddit(
                        query=broad_term,
                        sort="relevance", 
                        time_filter="month",
                        limit=limit_per_term,
                        subreddit=subreddit,
                        collect_comments=collect_comments,
                        max_comments=max_comments
                    )
                    
                    # Filter posts that contain target keywords
                    matches = []
                    for post in posts:
                        if post['id'] in seen_ids:
                            continue
                            
                        # Check title and selftext
                        text_to_search = f"{post['title']} {post['selftext']}".lower()
                        found_in_post = any(keyword in text_to_search for keyword in target_keywords_lower)
                        
                        # Check comments if available
                        found_in_comments = False
                        if post['comments']:
                            for comment in post['comments']:
                                comment_text = comment['body'].lower()
                                if any(keyword in comment_text for keyword in target_keywords_lower):
                                    found_in_comments = True
                                    break
                        
                        if found_in_post or found_in_comments:
                            post['found_in_post'] = found_in_post
                            post['found_in_comments'] = found_in_comments
                            post['matched_keywords'] = [kw for kw in target_keywords_lower 
                                                       if kw in text_to_search or 
                                                       (post['comments'] and any(kw in c['body'].lower() for c in post['comments']))]
                            post['broad_search_term'] = broad_term
                            post['search_location'] = location_name
                            matches.append(post)
                            seen_ids.add(post['id'])
                    
                    logging.info(f"  Found {len(matches)} posts with target keywords")
                    all_matches.extend(matches)
                    time.sleep(1)
                    
                except Exception as e:
                    logging.error(f"Error searching {location_name} for '{broad_term}': {e}")
                    continue
        
        logging.info(f"Hybrid search complete: {len(all_matches)} total matches found")
        return all_matches

def data_tools_search():
    """Search for data engineering and analytics tools using hybrid approach"""
    try:
        collector = RedditSearchCollector()
        
        # Replace with your specific target keywords
        target_keywords = ["data tools", "bigquery", "elt", "etl", "data pipeline", "data integration"]
        
        # Broader search terms to cast a wider net
        broad_terms = [
            "data pipeline tools",
            "ETL solutions", 
            "data integration platform",
            "business intelligence tools",
            "data warehouse tools",
            "analytics stack",
            "data connector",
            "data sync tools",
            "marketing data tools"
        ]
        
        # Relevant subreddits for data engineering discussions
        relevant_subreddits = [
            # "dataengineering",
            "analytics", 
            "BusinessIntelligence",
            "bigquery",
            "snowflake",
            "datascience",
            "dataanalysis",
            "PowerBI",
            "tableau",
            "marketing",
            "digital_marketing",
            "MarketingAutomation"
        ]
        
        # Hybrid search approach
        posts = collector.hybrid_search(
            target_keywords=target_keywords,
            broad_terms=broad_terms,
            relevant_subreddits=relevant_subreddits,
            limit_per_term=30,  # 30 posts per broad term per location
            collect_comments=True,
            max_comments=200
        )
        
        if posts:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            collector.save_to_csv(posts, f"data_tools_hybrid_search_{timestamp}.csv")
            # JSON output disabled by default
            # collector.save_to_json(posts, f"data_tools_hybrid_search_{timestamp}.json")
            
            # Clean up any temporary files after successful save
            collector.cleanup_progress_files()
            
            df = pd.DataFrame(posts)
            logging.info(f"\n=== HYBRID SEARCH RESULTS ===")
            logging.info(f"- Total matching posts: {len(df)}")
            logging.info(f"- Unique subreddits: {df['subreddit'].nunique()}")
            logging.info(f"- Average score: {df['score'].mean():.2f}")
            logging.info(f"- Average age (days): {df['days_ago'].mean():.1f}")
            logging.info(f"- Posts with comments: {df[df['comments_collected'] > 0].shape[0]}")
            
            # Analysis by match type
            logging.info(f"\n--- Match Analysis ---")
            found_in_post = df[df['found_in_post']].shape[0]
            found_in_comments = df[df['found_in_comments']].shape[0]
            logging.info(f"Keywords found in post title/content: {found_in_post}")
            logging.info(f"Keywords found in comments: {found_in_comments}")
            logging.info(f"Found in both: {df[df['found_in_post'] & df['found_in_comments']].shape[0]}")
            
            # Results by broad search term
            logging.info(f"\n--- Posts by Broad Search Term ---")
            broad_term_counts = df['broad_search_term'].value_counts()
            logging.info(f"\n{broad_term_counts}")
            
            # Results by search location
            logging.info(f"\n--- Posts by Search Location ---")
            location_counts = df['search_location'].value_counts()
            logging.info(f"\n{location_counts}")
            
            # Top subreddits with matches
            logging.info(f"\n--- Top Subreddits with Matches ---")
            logging.info(f"\n{df['subreddit'].value_counts().head(10)}")
            
            # Matched keywords frequency
            all_matched_keywords = []
            for keywords_list in df['matched_keywords']:
                all_matched_keywords.extend(keywords_list)
            if all_matched_keywords:
                from collections import Counter
                keyword_freq = Counter(all_matched_keywords)
                logging.info(f"\n--- Most Mentioned Target Keywords ---")
                for keyword, count in keyword_freq.most_common():
                    logging.info(f"{keyword}: {count} mentions")
            
        else:
            logging.warning("No posts found matching target keywords")
            
    except Exception as e:
        logging.error(f"Data tools search failed: {e}")

if __name__ == "__main__":
    data_tools_search()