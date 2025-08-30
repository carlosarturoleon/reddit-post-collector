# Reddit Data Collector & Analyzer

A comprehensive Python toolkit for collecting, analyzing, and extracting insights from Reddit posts across multiple subreddits. Features intelligent collection strategies, advanced sentiment analysis, topic modeling, and automated orchestration.

## Features

### 🚀 **Data Collection**
- **Activity-Based Collection**: Automatically adjusts collection parameters based on subreddit activity levels
- **Hybrid Search Collection**: Keyword and phrase-based targeted collection with relevance scoring
- **User Profile Collection**: Collect posts and comments from specific Reddit users
- **Multi-Format Export**: Save data in both CSV and JSON formats
- **Rate Limiting**: Built-in delays to respect Reddit's API limits

### 📊 **Advanced Analytics**
- **Sentiment Analysis**: VADER sentiment analyzer for posts, titles, and comments
- **Topic Modeling**: NMF-based unsupervised topic discovery and classification
- **Engagement Scoring**: Sophisticated scoring system weighing multiple factors
- **Relevance Filtering**: AI-powered content relevance assessment
- **LLM-Based Scoring**: Local LLM integration for intelligent post relevance scoring
- **Combined Analysis**: Merge and analyze data from multiple collection runs

### 🤖 **Automation & Orchestration**
- **Full Pipeline Automation**: One-command data collection → analysis workflow
- **Intelligent Processing**: Skip duplicates, handle errors, resume from interruptions
- **Comprehensive Logging**: Detailed logs for monitoring and debugging
- **Flexible Configuration**: Easy customization for different use cases

## Quick Start

### Prerequisites

- Python 3.7+
- Reddit API credentials (free)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/reddit-data-collector.git
cd reddit-data-collector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Reddit API credentials:
   - Create a Reddit app at https://www.reddit.com/prefs/apps
   - Copy `praw.ini.example` to `praw.ini`
   - Add your credentials to `praw.ini`

4. Run the complete pipeline (recommended):
```bash
# Run both collection and analysis
python reddit_orchestrator.py

# Or run specific collection types:
python reddit_orchestrator.py daily    # Activity-based collection only
python reddit_orchestrator.py search   # Search-based collection only
python reddit_orchestrator.py both     # Both types (default)
```

**Alternative: Individual Scripts**
```bash
# Manual collection and analysis
python reddit_daily_collector.py       # Activity-based collection
python reddit_search_collector.py      # Search-based collection
python reddit_data_analyzer.py         # Analysis only
python collect_users.py                # User profile collection

# LLM-based post scoring (requires Ollama)
python reddit_llm_scorer.py <analyzed_csv_file>
```

## Configuration

### Reddit API Setup

1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Choose "script" as the app type
4. Fill in the required fields:
   - **Name**: Your app name
   - **Description**: Brief description
   - **About URL**: Can be blank
   - **Redirect URI**: Use `http://localhost:8080`

5. Note down your credentials:
   - **Client ID**: Found under your app name
   - **Client Secret**: The secret string
   - **User Agent**: A unique identifier for your script

### Subreddit Configuration

The collector uses a tier- ased approach:

- **High Activity** (1M+ members): 150 posts, 24-hour window
- **Medium Activity** (100K-1M members): 100 posts, 24-hour window  
- **Low Activity** (10K-100K members): 50 posts, 3-day window
- **Very Low Activity** (<10K members): 25 posts, 7-day window

Edit the `SUBREDDIT_CONFIG` in `reddit_daily_collector.py` to customize subreddits for your use case.

## Usage Examples

### 🎯 **Complete Pipeline (Recommended)**
```bash
# Full automated pipeline: collect → analyze → generate insights
python reddit_orchestrator.py

# Specific collection strategies
python reddit_orchestrator.py daily    # Activity-based posts only
python reddit_orchestrator.py search   # Keyword-targeted posts only
```

### 📈 **Individual Collection Scripts**

**Activity-Based Collection**
```python
from reddit_daily_collector import RedditCollector

collector = RedditCollector()
posts = collector.collect_subreddit_posts('datascience', limit=100)
collector.save_to_csv(posts, 'my_data.csv')
```

**Search-Based Collection**
```python
from reddit_search_collector import RedditSearchCollector

collector = RedditSearchCollector()
posts = collector.collect_data_tools_posts()
collector.save_combined_data(posts, 'search_results')
```

**User Profile Collection**
```python
# Edit target_users list in collect_users.py
target_users = ['username1', 'username2']
python collect_users.py
```

### 🔬 **Analysis Only**
```bash
# Analyze existing data files
python reddit_data_analyzer.py

# Or use the orchestrator for analysis only
python -c "from reddit_orchestrator import RedditOrchestrator; o = RedditOrchestrator(); o.generate_final_combined_analysis()"
```

### ⚙️ **Custom Configuration**
```python
# Modify SUBREDDIT_CONFIG in reddit_daily_collector.py
SUBREDDIT_CONFIG = {
    'high_activity': {
        'subreddits': ['MachineLearning', 'Python', 'datascience'],
        'posts_per_sub': 100,
        'time_filter': 'day',
        'description': 'ML focused'
    }
}

# Modify search terms in reddit_search_collector.py
SEARCH_QUERIES = {
    'data_tools': {
        'keywords': ['tableau', 'powerbi', 'looker'],
        'phrases': ['data visualization', 'business intelligence']
    }
}
```

## 🤖 LLM-Based Post Scoring

### **Overview**
The LLM scorer uses a local Large Language Model to intelligently evaluate Reddit posts for relevance to data engineering and ETL workflows. This provides more nuanced scoring than traditional keyword-based approaches.

### **Prerequisites**
```bash
# Install Ollama (local LLM runtime)
# macOS:
brew install ollama

# Linux/Windows: Visit https://ollama.com/download

# Start Ollama server
ollama serve

# Download a model (recommended: llama3.2:3b for speed)
ollama pull llama3.2:3b
```

### **Usage**
```bash
# Score an analyzed CSV file
python reddit_llm_scorer.py analysis_results/reddit_combined_analysis_YYYYMMDD_analyzed.csv

# The script will:
# 1. Connect to your local Ollama instance
# 2. Score each post individually (including all comments)
# 3. Save checkpoints every 10 posts for reliability
# 4. Generate a new CSV sorted by LLM relevance scores
```

### **Features**
- **Intelligent Analysis**: Evaluates post title, content, AND all comments together
- **Checkpoint System**: Automatically saves progress every 10 posts
- **Resume Capability**: If interrupted, resumes from last checkpoint
- **Robust Comment Parsing**: Handles malformed JSON and CSV escaping issues
- **Progress Tracking**: Real-time updates on scoring progress
- **Configurable Models**: Works with any Ollama-supported model

### **Scoring Criteria**
The LLM evaluates posts on a 0-100 scale based on:

**High Score (70-100):**
- ETL/ELT pipeline challenges and solutions
- Data integration between platforms
- API integration difficulties
- Data warehouse and analytics discussions
- Business intelligence automation
- Performance optimization topics

**Medium Score (40-69):**
- General analytics concepts
- Database administration
- Cloud platform discussions
- Data modeling topics

**Low Score (0-39):**
- Unrelated programming tutorials
- Career/salary discussions
- Academic topics without practical application

### **Output**
```
📊 Processing: reddit_analysis_20241120_analyzed.csv
💬 Total comments to analyze: 1,247 across 89 posts  
🤖 Starting LLM scoring with checkpoints every 10 posts...
📈 Processed 50/216 posts | New: 45 | Cached: 5 | Avg: 73.2
📄 Results saved to: reddit_analysis_20241120_analyzed_llm_scored_20241120_143022.csv
```

### **Model Recommendations**
- **llama3.2:3b** - Fast, good quality (recommended)
- **llama3.2:1b** - Fastest, lower quality
- **llama3:8b** - Slower, highest quality

## Data Schema

Each collected post includes:

### **Raw Data Schema**
| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique Reddit post ID |
| `title` | string | Post title |
| `author` | string | Username (or [deleted]) |
| `score` | integer | Net upvotes (upvotes - downvotes) |
| `upvote_ratio` | float | Ratio of upvotes to total votes |
| `num_comments` | integer | Number of comments |
| `created_utc` | timestamp | Creation time (UTC) |
| `hours_ago` | float | Hours since creation |
| `url` | string | Link URL |
| `permalink` | string | Reddit permalink |
| `selftext` | string | Post content (for text posts) |
| `subreddit` | string | Subreddit name |
| `activity_tier` | string | Collection tier classification |
| `comments` | json | Serialized comment threads (if collected) |

### **Analyzed Data Schema** (Additional Fields)
| Field | Type | Description |
|-------|------|-------------|
| `title_sentiment` | float | Sentiment score of title (-1 to 1) |
| `title_sentiment_label` | string | Sentiment classification (positive/neutral/negative) |
| `content_sentiment` | float | Sentiment score of post content |
| `content_sentiment_label` | string | Content sentiment classification |
| `avg_comment_sentiment` | float | Average sentiment of all comments |
| `topic` | string | Assigned topic from NMF modeling |
| `topic_score` | float | Confidence score for topic assignment |
| `engagement_score` | float | Computed engagement score (0-100) |
| `relevance_score` | integer | AI-assessed relevance to data tools/tech |
| `_sentiment_order` | integer | Internal ranking for sentiment prioritization |

## Output Files

### **Collection Outputs**
- `reddit_activity_based_YYYYMMDD.csv` - Activity-based collection (raw data)
- `reddit_activity_based_YYYYMMDD.json` - Activity-based collection (with metadata)
- `{query}_hybrid_search_YYYYMMDD_HHMMSS.csv` - Search-based collection results
- `reddit_users_YYYYMMDD_HHMMSS.csv` - User profile collection results

### **Analysis Outputs**  
- `reddit_combined_analysis_YYYYMMDD_HHMMSS_analyzed_YYYYMMDD_HHMMSS.csv` - **Final combined analysis** (recommended)
- Individual analyzed files: `*_analyzed_*.csv` (when running individual analysis)

### **Logs**
- `reddit_collector.log` - Collection execution logs
- `reddit_data_analyzer.log` - Analysis execution logs  
- `reddit_orchestrator.log` - Pipeline orchestration logs

### **Directories**
- `analysis_results/` - All output files are saved here
- `progress_tracking/` - Collection progress and state files (if enabled)

## Best Practices

### Rate Limiting
- Built-in 1-second delays between subreddit requests
- Respects Reddit's API rate limits (60 requests per minute)
- Monitor logs for any rate limiting warnings

### Data Quality
- Filters deleted/removed posts
- Handles missing data 
- Validates data types and formats

## Troubleshooting

### Common Issues

**Authentication Errors**
- Verify your `praw.ini` credentials
- Ensure your Reddit app is set to "script" type
- Check that your user agent is unique

**Empty Results**
- Some subreddits may have very low activity
- Try increasing time filters for low-activity subreddits
- Check subreddit names for typos

**Rate Limiting**
- If you see 429 errors, reduce collection frequency
- Increase delays between requests if needed

### Getting Help

1. Check the logs in `reddit_collector.log` for detailed error messages
2. Verify your Reddit API credentials
3. Test with a single, active subreddit first
4. Open an issue on GitHub with your error logs

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## Legal and Ethical Use

- This tool is for legitimate research and analysis purposes
- Respect Reddit's Terms of Service and API guidelines
- Consider the privacy and consent of Reddit users
- Use collected data responsibly and ethically
- Don't spam or overload Reddit's servers

## License

MIT License - see LICENSE file for details.

## Changelog

### v2.1.0 (Current)
- **NEW**: LLM-based post scoring with local Ollama integration
- **NEW**: Intelligent comment parsing with multiple JSON fallback methods
- **NEW**: Checkpoint system for reliable large dataset processing
- **NEW**: Resume capability for interrupted LLM scoring sessions
- **ENHANCED**: Post relevance evaluation using advanced AI models

### v2.0.0
- **NEW**: Complete pipeline orchestration with `reddit_orchestrator.py`
- **NEW**: Advanced sentiment analysis with VADER
- **NEW**: Topic modeling with NMF (Non-negative Matrix Factorization)
- **NEW**: Sophisticated engagement scoring system
- **NEW**: AI-powered relevance filtering
- **NEW**: Hybrid search collection with keyword + phrase targeting
- **NEW**: User profile collection functionality
- **NEW**: Combined analysis merging multiple data sources
- **ENHANCED**: Comprehensive logging across all components
- **ENHANCED**: Robust error handling and recovery
- **ENHANCED**: Smart duplicate detection and removal

### v1.0.0 
- Initial release
- Activity-based collection strategy  
- Multi-format export (CSV/JSON)
- Basic logging and error handling

## Advanced Features

### **Sentiment Analysis**
- **VADER Sentiment Analyzer**: Optimized for social media text
- **Multi-level Analysis**: Titles, content, and comments analyzed separately
- **Aggregate Scoring**: Combined sentiment metrics across all post components

### **Topic Modeling** 
- **NMF Algorithm**: Discovers hidden topics in collected data
- **Automatic Classification**: Posts automatically categorized by dominant topic
- **Configurable Topics**: Adjustable number of topics (default: 8)

### **Engagement Scoring Algorithm**
```python
# Weighted scoring system
engagement_score = (
    score * 0.3 +           # Reddit score weight
    upvote_ratio * 0.25 +   # Community approval
    comments * 0.25 +       # Discussion activity  
    comment_quality * 0.15 + # Comment sentiment
    recency_factor * 0.05   # Time decay
)
```

### **Smart Filtering**
- **Relevance Assessment**: AI-powered filtering for data tools/tech content
- **Quality Thresholds**: Configurable minimum engagement, score, comments
- **Duplicate Detection**: Intelligent deduplication across collection runs
- **Time-based Filtering**: Configurable recency requirements