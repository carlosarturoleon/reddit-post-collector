#!/usr/bin/env python3
"""
Reddit LLM Post Scorer
Post-processing tool that scores Reddit posts using local LLM and re-sorts by relevance
"""

import pandas as pd
import numpy as np
import logging
import json
import sys
import os
import glob
import time
import re
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_llm_scoring.log'),
        logging.StreamHandler()
    ],
    force=True
)

class RedditLLMScorer:
    """
    Post-processing LLM scorer for Reddit posts
    Reads analyzed CSV files and adds LLM relevance scores
    """
    
    def __init__(self, model="llama3.2:3b", base_url="http://localhost:11434"):
        """Initialize the LLM scorer"""
        self.model = model
        self.base_url = base_url
        self.cache = {}
        
        # Import requests
        try:
            import requests
            self.requests = requests
            self.available = True
            logging.info(f"🤖 RedditLLMScorer initialized with model: {model}")
        except ImportError:
            logging.error("❌ RedditLLMScorer requires 'requests' library. Run: pip install requests")
            self.available = False
    
    def test_ollama_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            response = self.requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if any(self.model in name for name in model_names):
                    logging.info(f"✅ Ollama connection verified, model {self.model} available")
                    return True
                else:
                    logging.warning(f"⚠️  Model {self.model} not found. Available models: {model_names}")
                    return False
            return False
        except Exception as e:
            logging.warning(f"⚠️  Cannot connect to Ollama: {e}")
            return False
    
    def get_all_comments_text(self, comments):
        """Extract ALL comments text for comprehensive analysis"""
        if not comments or pd.isna(comments):
            return ""
        
        try:
            if isinstance(comments, str):
                # Handle empty list case
                if comments.strip() == '[]':
                    return ""
                
                if comments.startswith('['):
                    # Try different JSON parsing approaches for malformed JSON
                    try:
                        comments = json.loads(comments)
                    except json.JSONDecodeError:
                        # Try fixing common CSV escaping issues
                        try:
                            # Replace escaped quotes and try again
                            fixed_comments = comments.replace('""', '"').replace('\\"', '"')
                            comments = json.loads(fixed_comments)
                        except json.JSONDecodeError:
                            # Try using ast.literal_eval as backup
                            try:
                                import ast
                                comments = ast.literal_eval(comments)
                            except (ValueError, SyntaxError):
                                # Last resort: try parsing manually
                                logging.debug(f"Failed to parse comments JSON, skipping: {comments[:100]}...")
                                return ""
                else:
                    return ""
            
            if not isinstance(comments, list):
                return ""
            
            # Extract ALL comment bodies
            all_comment_text = []
            for comment in comments:
                if isinstance(comment, dict) and comment.get('body'):
                    body = comment['body'].strip()
                    if body and len(body) > 5:  # Skip very short comments
                        # Clean up comment text - handle escaped characters properly
                        body = body.replace('\\n', ' ').replace('\\r', ' ')
                        body = body.replace('\n', ' ').replace('\r', ' ')
                        body = ' '.join(body.split())  # Normalize whitespace
                        all_comment_text.append(body)
            
            if not all_comment_text:
                return ""
            
            full_comments = " | ".join(all_comment_text)
            
            # Truncate if too long
            max_chars = 2000
            if len(full_comments) > max_chars:
                full_comments = full_comments[:max_chars] + "... [truncated]"
            
            return full_comments
            
        except Exception as e:
            logging.debug(f"Error processing comments: {e}")
            return ""
    
    def create_scoring_prompt(self, title, content, all_comments=""):
        """Create data engineering relevance scoring prompt"""
        prompt = f"""You are a data engineering expert evaluating Reddit posts for relevance to data integration and ETL workflows.

Rate this post from 0-100 based on how well it matches modern data engineering challenges and solutions.

FOCUS AREAS - SCORE HIGH (70-100) FOR:
✅ ETL/ELT pipeline discussions and implementation challenges
✅ Data integration between various platforms and systems
✅ API integration difficulties and solutions
✅ Data warehouse and analytics platform discussions
✅ Business intelligence and reporting automation
✅ Data pipeline architecture and best practices
✅ Performance optimization and monitoring
✅ Data quality and transformation challenges
✅ Questions about connecting platforms: "How to get data from X to Y"
✅ Manual process automation and efficiency improvements
✅ Alternative tooling discussions (Fivetran, Airbyte, dbt, custom solutions)

MEDIUM RELEVANCE (40-69):
✅ General analytics and business intelligence concepts
✅ Database administration and optimization
✅ Cloud platform discussions (AWS, GCP, Azure)
✅ Data modeling and schema design
✅ Workflow orchestration and scheduling

LOW RELEVANCE (0-39):
❌ Pure programming tutorials unrelated to data workflows
❌ Career advice, salary discussions, general business topics
❌ Academic/theoretical discussions without practical application
❌ Frontend development, mobile apps, or unrelated software topics

POST TO EVALUATE:
Title: {title[:400]}
Content: {content[:800] if content else "No content"}  
Comments: {all_comments if all_comments else "No comments available"}

Focus on practical data engineering problems, implementation questions, and discussions where users need technical solutions.

Return ONLY a number 0-100."""
        
        return prompt
    
    def extract_score(self, response_text):
        """Extract numerical score from LLM response"""
        try:
            numbers = re.findall(r'\b(\d{1,3})\b', response_text.strip())
            if numbers:
                score = int(numbers[0])
                return min(max(score, 0), 100)
            else:
                logging.warning(f"No score found in: '{response_text[:50]}'")
                return 50
        except Exception as e:
            logging.warning(f"Error extracting score: {e}")
            return 50
    
    def score_post(self, post_row):
        """Score a single post with LLM using ALL comments"""
        post_id = post_row.get('id', 'unknown')
        
        if post_id in self.cache:
            return self.cache[post_id]
        
        try:
            title = post_row.get('title', '')
            content = post_row.get('selftext', '')
            comments = post_row.get('comments', [])
            
            all_comments = self.get_all_comments_text(comments)
            prompt = self.create_scoring_prompt(title, content, all_comments)
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 10
                }
            }
            
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '50')
                score = self.extract_score(response_text)
                self.cache[post_id] = score
                return score
            else:
                logging.warning(f"API error for post {post_id}: {response.status_code}")
                return 50
                
        except Exception as e:
            logging.warning(f"Error scoring post {post_id}: {e}")
            return 50
    
    def load_checkpoint(self, checkpoint_file):
        """Load existing scores from checkpoint file"""
        try:
            if os.path.exists(checkpoint_file):
                checkpoint_df = pd.read_csv(checkpoint_file)
                logging.info(f"📁 Found checkpoint with {len(checkpoint_df)} scored posts")
                
                # Convert to cache format
                for _, row in checkpoint_df.iterrows():
                    post_id = row.get('id')
                    score = row.get('llm_relevance_score')
                    if post_id and not pd.isna(score):
                        self.cache[post_id] = score
                
                logging.info(f"✅ Loaded {len(self.cache)} scores from checkpoint")
                return True
            return False
        except Exception as e:
            logging.warning(f"⚠️  Could not load checkpoint: {e}")
            return False
    
    def save_checkpoint(self, df_with_scores, checkpoint_file):
        """Save current progress to checkpoint file"""
        try:
            df_with_scores.to_csv(checkpoint_file, index=False)
            logging.debug(f"💾 Checkpoint saved: {len(df_with_scores)} posts")
        except Exception as e:
            logging.warning(f"⚠️  Could not save checkpoint: {e}")
    
    def score_file(self, csv_file_path, checkpoint_interval=10):
        """Score all posts in a CSV file with checkpoint saving"""
        logging.info(f"🎯 Processing file: {os.path.basename(csv_file_path)}")
        
        if not self.test_ollama_connection():
            logging.error("❌ Cannot connect to Ollama - skipping LLM scoring")
            return None
        
        try:
            # Load original data
            df = pd.read_csv(csv_file_path)
            logging.info(f"📊 Loaded {len(df)} posts from CSV")
            
            # Set up checkpoint system
            base_name = os.path.splitext(csv_file_path)[0]
            checkpoint_file = f"{base_name}_llm_checkpoint.csv"
            
            # Try to load existing checkpoint
            if self.load_checkpoint(checkpoint_file):
                logging.info("🔄 Resuming from checkpoint...")
            
            # Initialize score column if not exists
            if 'llm_relevance_score' not in df.columns:
                df['llm_relevance_score'] = np.nan
            
            # Count total comments for info
            total_comments = 0
            posts_with_comments = 0
            for _, row in df.iterrows():
                comments = row.get('comments', [])
                if comments and not pd.isna(comments) and comments != '[]':
                    try:
                        if isinstance(comments, str) and comments.startswith('['):
                            # Use the same parsing logic as get_all_comments_text
                            try:
                                parsed_comments = json.loads(comments)
                            except json.JSONDecodeError:
                                try:
                                    fixed_comments = comments.replace('""', '"').replace('\\"', '"')
                                    parsed_comments = json.loads(fixed_comments)
                                except json.JSONDecodeError:
                                    try:
                                        import ast
                                        parsed_comments = ast.literal_eval(comments)
                                    except (ValueError, SyntaxError):
                                        continue
                        else:
                            parsed_comments = comments
                            
                        if isinstance(parsed_comments, list) and len(parsed_comments) > 0:
                            total_comments += len(parsed_comments)
                            posts_with_comments += 1
                    except:
                        pass
            
            logging.info(f"💬 Total comments to analyze: {total_comments} across {posts_with_comments} posts")
            
            # Process posts with checkpoint saving
            total = len(df)
            scored_count = 0
            skipped_count = 0
            
            logging.info(f"🤖 Starting LLM scoring with checkpoints every {checkpoint_interval} posts...")
            
            for idx, (df_idx, row) in enumerate(df.iterrows(), 1):
                post_id = row.get('id')
                
                # Skip if already scored (from checkpoint or previous run)
                if post_id in self.cache:
                    df.loc[df_idx, 'llm_relevance_score'] = self.cache[post_id]
                    skipped_count += 1
                elif not pd.isna(row.get('llm_relevance_score')):
                    skipped_count += 1
                else:
                    # Score the post
                    score = self.score_post(row.to_dict())
                    df.loc[df_idx, 'llm_relevance_score'] = score
                    scored_count += 1
                    
                    time.sleep(0.3)  # Rate limiting
                
                # Progress updates
                if idx % 5 == 0 or idx == total:
                    current_scores = df['llm_relevance_score'].dropna()
                    avg_score = current_scores.mean() if len(current_scores) > 0 else 0
                    logging.info(f"  📈 Processed {idx}/{total} posts | New: {scored_count} | Cached: {skipped_count} | Avg: {avg_score:.1f}")
                
                # Checkpoint saving
                if idx % checkpoint_interval == 0:
                    self.save_checkpoint(df, checkpoint_file)
                    logging.info(f"  💾 Checkpoint saved at post {idx}")
            
            # Final validation - make sure all posts have scores
            missing_scores = df['llm_relevance_score'].isna().sum()
            if missing_scores > 0:
                logging.warning(f"⚠️  {missing_scores} posts still missing scores, filling with default")
                df['llm_relevance_score'] = df['llm_relevance_score'].fillna(50)
            
            # Sort by LLM score (highest first)
            df_sorted = df.sort_values('llm_relevance_score', ascending=False)
            
            # Generate final filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{base_name}_llm_scored_{timestamp}.csv"
            
            # Save final results
            df_sorted.to_csv(new_filename, index=False)
            
            # Clean up checkpoint file
            try:
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
                    logging.info("🧹 Checkpoint file cleaned up")
            except:
                pass
            
            # Calculate and log results
            all_scores = df_sorted['llm_relevance_score'].tolist()
            high_scores = sum(1 for s in all_scores if s >= 70)
            medium_scores = sum(1 for s in all_scores if 50 <= s < 70)
            low_scores = sum(1 for s in all_scores if s < 50)
            
            logging.info(f"\n🎉 LLM scoring complete!")
            logging.info(f"📄 Results saved to: {new_filename}")
            logging.info(f"🔢 Processing summary:")
            logging.info(f"   Newly scored: {scored_count} posts")
            logging.info(f"   From cache/checkpoint: {skipped_count} posts")
            logging.info(f"💬 Analyzed {total_comments} comments across {total} posts")
            logging.info(f"📊 Final score distribution:")
            logging.info(f"   High (70-100): {high_scores} posts ({high_scores/total*100:.1f}%)")
            logging.info(f"   Medium (50-69): {medium_scores} posts ({medium_scores/total*100:.1f}%)")  
            logging.info(f"   Low (0-49): {low_scores} posts ({low_scores/total*100:.1f}%)")
            logging.info(f"   Average score: {np.mean(all_scores):.1f}")
            logging.info(f"📈 Posts sorted by LLM relevance score (highest first)")
            
            return new_filename
            
        except Exception as e:
            logging.error(f"❌ Error processing file: {e}")
            return None

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python reddit_llm_scorer.py <csv_file_path>")
        return
    
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return
    
    scorer = RedditLLMScorer()
    if not scorer.available:
        print("❌ Cannot initialize LLM scorer")
        return
    
    result = scorer.score_file(csv_file)
    if result:
        print(f"✅ Success! LLM scored file: {result}")
    else:
        print("❌ Failed to process file")

if __name__ == "__main__":
    main()