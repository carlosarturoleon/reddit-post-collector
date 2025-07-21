# Reddit Data Collector

A Python tool for collecting and analyzing Reddit posts from multiple subreddits using an activity based collection strategy. 

## Features

- **Activity-Based Collection Strategy**: Manually adjusts collection parameters based on subreddit activity levels
- **Flexible Data Export**: Save data in both CSV and JSON formats
- **Logging**: Detailed logs for monitoring and debugging
- **Rate Limiting**: Built-in delays to respect Reddit's API limits
- **Rich Metadata**: Collects 20+ data points per post including engagement metrics
- **Time-Based Filtering**: Filtering based on subreddit activity patterns

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

4. Run the collector:
```bash
python reddit_daily_collector.py
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

### Collecting Comments from Specific Users

Use the `collect_specific_users.py` script to collect all comments from specific Reddit users:

```python
# Edit the target_users list in collect_specific_users.py
target_users = ['username1', 'username2']

# Run the script
python collect_specific_users.py
```

### Basic Collection
```python
from reddit_daily_collector import RedditCollector

collector = RedditCollector()
posts = collector.collect_subreddit_posts('datascience', limit=100)
collector.save_to_csv(posts, 'my_data.csv')
```

### Custom Configuration
```python
# Modify the SUBREDDIT_CONFIG dictionary in the script
SUBREDDIT_CONFIG = {
    'high_activity': {
        'subreddits': ['MachineLearning', 'Python'],
        'posts_per_sub': 100,
        'time_filter': 'day',
        'description': 'ML focused'
    }
}
```

### Basic Collection
```python
from reddit_daily_collector import RedditCollector

collector = RedditCollector()
posts = collector.collect_subreddit_posts('datascience', limit=100)
collector.save_to_csv(posts, 'my_data.csv')
```

## Data Schema

Each collected post includes:

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

## Output Files

The collector generates timestamped files:
- `reddit_activity_based_YYYYMMDD.csv` - Tabular data for analysis
- `reddit_activity_based_YYYYMMDD.json` - Structured data with metadata
- `reddit_collector.log` - Detailed execution logs

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

### v1.0.0
- Initial release
- Activity based collection strategy
- Multi format export (CSV/JSON)
- Logging and error handling