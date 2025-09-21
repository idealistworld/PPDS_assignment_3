# os: File system operations for creating directories and paths
# json: JSON parsing for API responses and article data
# pandas: DataFrame ops for efficient manipulation and CSV export
# datetime: Date arithmetic for price ranges and filtering
# requests: HTTP client for TheNewsAPI calls
# dotenv: Environment variable loading for API key management
# typing: Type hints for better documentation and IDE support
# tqdm: Progress bars for long operations (fetching, analysis)

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
from typing import Dict
from tqdm import tqdm
import time  # For rate limiting between API calls


# Constants (configuration and magic numbers)
NEWS_API_URL = "https://api.thenewsapi.com/v1/news/all"
RAW_DIR = 'data/raw'
ENRICHED_DIR = 'data/enriched'
CRYPTO_CSV_PATH = os.path.join(RAW_DIR, 'crypto_data.csv')
RAW_NEWS_FILENAME = 'solana_news.json'
ENRICHED_CSV_FILENAME = 'enriched_data.csv'
REQUEST_TIMEOUT_SECONDS = 10
INTER_DAY_SLEEP_SECONDS = 1
NEWS_MAX_RETRIES = 3
NEWS_BASE_DELAY_SECONDS = 1
DEFAULT_MAX_DAYS_NO_DATA = 30
MAX_LOOKBACK_LIMIT = 730
MAX_ARTICLES_PER_DAY = 10
DEFAULT_QUERY = 'solana'
DEFAULT_ARTICLES_PER_DAY = 3
DEFAULT_DAYS_BACK = 7
DEFAULT_DAYS_BACK_NO_END = 7
DATE_FMT = '%Y-%m-%d'
NEWS_LANGUAGE = 'en'
TQDM_NCOLS = 80


def get_available_date_range():
    """
    Determine the date range for which we have historical crypto price data.

    This function reads our local crypto dataset to establish the boundaries
    for news article fetching. We can only provide accurate sentiment vs. price
    analysis for dates where we have actual price data.

    Why:
        Ensures news analysis aligns with dates that have real price data,
        preventing misleading comparisons and empty joins.

    Returns:
        tuple: (start_date, end_date) as pandas Timestamps, or
        (None, None) if data unavailable
    """
    try:
        # Read local crypto data (more reliable than API calls
        # for historical data)
        crypto_df = pd.read_csv(CRYPTO_CSV_PATH)
        crypto_df['timestamp'] = pd.to_datetime(crypto_df['timestamp'])
        return crypto_df['timestamp'].min(), crypto_df['timestamp'].max()
    except Exception as exc:
        print(f"Error reading crypto data: {exc}")
        return None, None


def fetch_solana_price_data(date: str) -> Dict[str, float]:
    """
    Fetch Solana price data for a specific date from local CSV.

    Args:
        date: Date string in 'YYYY-MM-DD' format

    Returns:
        Dict containing price, price_change, and price_change_percent

    Why:
        Uses a local, deterministic source for prices to avoid external
        API variability and to compute consistent day-over-day metrics.
    """
    try:
        crypto_df = pd.read_csv(CRYPTO_CSV_PATH)
        crypto_df['timestamp'] = pd.to_datetime(crypto_df['timestamp'])
        crypto_df['date'] = crypto_df['timestamp'].dt.date

        target_date = datetime.strptime(date, '%Y-%m-%d').date()
        day_data = crypto_df[crypto_df['date'] == target_date]

        if day_data.empty:
            return {
                'price': 0.0,
                'price_change': 0.0,
                'price_change_percent': 0.0,
            }

        current_price = day_data['SOL_close'].iloc[-1]
        prev_date = target_date - timedelta(days=1)
        prev_day_data = crypto_df[crypto_df['date'] == prev_date]

        if not prev_day_data.empty:
            prev_price = prev_day_data['SOL_close'].iloc[-1]
            price_change = current_price - prev_price
            price_change_percent = (
                (price_change / prev_price * 100) if prev_price > 0 else 0.0
            )
        else:
            opening_price = day_data['SOL_close'].iloc[0]
            price_change = current_price - opening_price
            price_change_percent = (
                (price_change / opening_price * 100)
                if opening_price > 0
                else 0.0
            )

        return {
            'price': round(current_price, 2),
            'price_change': round(price_change, 2),
            'price_change_percent': round(price_change_percent, 2),
        }
    except Exception as exc:
        print(f"Error fetching price data for {date}: {exc}")
        return {'price': 0.0, 'price_change': 0.0, 'price_change_percent': 0.0}


def calculate_max_days_back(end_date):
    """
    Calculate maximum days we can fetch news based on available price data.

    This prevents users from requesting news for dates where we have no
    corresponding price data, ensuring all analysis is grounded in actual
    market movements.

    Args:
        end_date: Latest date in our price dataset

    Returns:
        int: Maximum number of days back we can safely fetch news

    Why:
        Bounds user input to the dataset’s coverage window so enrichment
        remains valid and comparable across dates.
    """
    if not end_date:
        return DEFAULT_DAYS_BACK_NO_END
    if hasattr(end_date, 'tz_localize'):
        end_date = end_date.tz_localize(None)

    # Calculate how many days back from the end of our price data to the start
    start_date, _ = get_available_date_range()
    if start_date:
        if hasattr(start_date, 'tz_localize'):
            start_date = start_date.tz_localize(None)
        total_data_days = (end_date - start_date).days
        # Limit to available data or MAX_LOOKBACK_LIMIT (2 years)
        return min(MAX_LOOKBACK_LIMIT, total_data_days)
    return MAX_LOOKBACK_LIMIT


def fetch_news_data(
    query=DEFAULT_QUERY,
    articles_per_day=DEFAULT_ARTICLES_PER_DAY,
    days_back=DEFAULT_DAYS_BACK,
    end_date=None,
):
    """
    Fetch Solana news articles from TheNewsAPI with rate limiting.

    Uses TheNewsAPI instead of NewsAPI because:
    1. Better rate limits for educational projects
    2. More reliable historical data access
    3. Cleaner response format for batch processing

    Implements rate limiting to respect API terms and prevent 429 errors.

    Args:
        query: Search terms for news filtering
        articles_per_day: Number of articles to fetch per day (max 10)
        days_back: Number of days to look back from end_date
        end_date: Starting point for lookback (defaults to today)

    Returns:
        dict: API response with articles list, or None if failed

    Why:
        Batches date-based requests against a reliable news source and
        rate‑limits to reduce 429s while keeping results aligned to price
        data end_date for accurate downstream joins.
    """
    load_dotenv()
    api_key = os.getenv('NEWS_API')

    if not api_key:
        print("Error: NEWS_API key not found in environment variables")
        return None

    url = NEWS_API_URL
    all_articles = []

    # Use the end_date of our price data, not today
    reference_date = end_date if end_date else datetime.now()
    if hasattr(reference_date, 'tz_localize'):
        reference_date = reference_date.tz_localize(None)

    for day_offset in tqdm(
        range(days_back),
        desc="📰 Fetching articles",
        unit="day",
        ncols=TQDM_NCOLS,
    ):
        target_date = reference_date - pd.Timedelta(days=day_offset)
        date_str = target_date.strftime(DATE_FMT)

        params = {
            'api_token': api_key,
            'search': query,
            'language': NEWS_LANGUAGE,
            'published_on': date_str,
            'limit': articles_per_day,
        }

        # Per-day retry with exponential backoff for resilience
        max_retries = NEWS_MAX_RETRIES
        base_delay = NEWS_BASE_DELAY_SECONDS
        fetched = False

        for attempt in range(max_retries + 1):
            try:
                response = requests.get(
                    url, params=params, timeout=REQUEST_TIMEOUT_SECONDS
                )
                response.raise_for_status()
                day_data = response.json()
                if day_data.get('data'):
                    articles_found = len(day_data['data'])
                    print(f"📅 {date_str}: found {articles_found} articles")
                    all_articles.extend(day_data['data'])
                else:
                    print(f"📅 {date_str}: no articles found")
                fetched = True
                break

            except requests.exceptions.HTTPError as exc:
                status = getattr(exc.response, 'status_code', None)
                if status == 429 and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    print(
                        f"⏳ TheNewsAPI rate limited {date_str}, retrying in "
                        f"{delay}s ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                print(f"⚠️  HTTP error for {date_str}: {exc}")
                break

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException,
            ) as exc:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    print(
                        f"🔄 Network error for {date_str}, retrying in "
                        f"{delay}s: {exc}"
                    )
                    time.sleep(delay)
                    continue
                print(f"❌ Failed to fetch news for {date_str}: {exc}")
                break

        # Rate limiting between days to respect API limits
        time.sleep(INTER_DAY_SLEEP_SECONDS)

    return {
        'status': 'ok',
        'totalResults': len(all_articles),
        'articles': all_articles,
    }


def save_raw_data(data, filename):
    """
    Save raw API response data to JSON file for debugging and backup.

    Preserves original API response format in case we need to reprocess
    data later without making additional API calls.

    Args:
        data: Raw API response data
        filename: Output filename (will be saved to data/raw/)

    Returns:
        bool: True if successful, False otherwise

    Why:
        Raw artifacts make runs reproducible, aid debugging, and allow
        re‑processing without refetching from external services.
    """
    try:
        os.makedirs(RAW_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as exc:
        print(f"Error saving raw data to {filename}: {exc}")
        return False


def create_articles_dataframe(news_data):
    """
    Convert raw news API response into a structured pandas DataFrame.

    Extracts key fields needed for sentiment analysis while preserving
    the full raw JSON for detailed analysis later. Uses pandas for
    efficient data manipulation and CSV export.

    Args:
        news_data: Raw API response containing articles list

    Returns:
        pd.DataFrame: Structured article data ready for enrichment

    Why:
        Normalizes raw JSON to tabular form so enrichment logic and
        exports can operate predictably and efficiently.
    """
    if not news_data or 'articles' not in news_data:
        return pd.DataFrame()

    df_data = [
        {
            'title': article.get('title', ''),
            'source_name': article.get('source', ''),
            'url': article.get('url', ''),
            'published_at': article.get('published_at', ''),
            'raw_article_json': json.dumps(article),
        }
        for article in news_data['articles']
    ]

    df = pd.DataFrame(df_data)
    if not df.empty:
        df['published_at'] = pd.to_datetime(df['published_at'])
    return df


def save_enriched_data(df, filename):
    """
    Export enriched DataFrame to CSV for analysis and reporting.

    Uses CSV format for maximum compatibility with analysis tools
    (Excel, Google Sheets, R, etc.) and human readability.

    Args:
        df: Enriched DataFrame with sentiment analysis
        filename: Output CSV filename

    Returns:
        bool: True if successful, False otherwise

    Why:
        Outputs a portable, widely supported artifact to share results
        and enable downstream analytics without special tooling.
    """
    try:
        os.makedirs(ENRICHED_DIR, exist_ok=True)
        filepath = os.path.join(ENRICHED_DIR, filename)
        df.to_csv(filepath, index=False, encoding='utf-8')
        return True
    except Exception as exc:
        print(f"Error saving enriched data to {filename}: {exc}")
        return False


def main():
    """
    Main entry point for Solana news sentiment analysis pipeline.

    Orchestrates user interaction, data fetching, and AI enrichment.
    """
    print("=== Solana News & Sentiment Analysis Tool ===")

    # Step 1: Choose focus area with streamlined selection
    FOCUS_AREAS = {
        '1': (
            'General Ecosystem',
            'solana',
        ),
        '2': (
            'DeFi',
            'solana DeFi',
        ),
        '3': (
            'NFTs, Gaming, Memecoins',
            'solana NFT',
        ),
    }

    print("\n🎯 Choose your Solana focus area:")
    for key, (desc, _) in FOCUS_AREAS.items():
        if key == '1':
            print(f"{key}. {desc} (FOR DEMO USE THIS BECAUSE THIS API HAS LIMITED VOLUME)")
        else:
            print(f"{key}. {desc}")

    focus_choice = input("\nSelect focus area (1-3): ").strip()
    while focus_choice not in FOCUS_AREAS:
        focus_choice = input("Please enter a number between 1 and 3: ").strip()

    focus_description, search_query = FOCUS_AREAS[focus_choice]
    print(f"✅ Selected: {focus_description}")
    print(f"🔍 Search query: '{search_query}'")

    # Step 2: Choose analysis persona with streamlined selection
    PERSONAS = {
        '1': (
            'Finance Analyst',
            'traditional financial analyst focused on market fundamentals '
            'and valuation metrics',
        ),
        '2': (
            'Tech Enthusiast',
            'technology expert evaluating blockchain innovation and '
            'technical advancement',
        ),
        '3': (
            'Crypto Investor',
            'experienced cryptocurrency investor analyzing market trends '
            'and adoption',
        ),
        '4': (
            'Risk Manager',
            'conservative risk assessment specialist focused on downside '
            'protection',
        ),
        '5': (
            'Growth Investor',
            'venture capital mindset looking for disruptive potential '
            'and scalability',
        ),
    }

    print("\n🎭 Choose analysis persona:")
    for key, (name, desc) in PERSONAS.items():
        print(f"{key}. {name} - {desc}")

    persona_choice = input("\nSelect persona (1-5): ").strip()
    while persona_choice not in PERSONAS:
        persona_choice = input(
            "Please enter a number between 1 and 5: "
        ).strip()

    selected_persona = PERSONAS[persona_choice]
    print(f"✅ Selected: {selected_persona[0]}")

    # Step 3: Date range and article settings
    start_date, end_date = get_available_date_range()
    if start_date and end_date:
        max_days = calculate_max_days_back(end_date)
        print(
            "\n📊 Available price data: "
            f"{start_date.strftime(DATE_FMT)} to "
            f"{end_date.strftime(DATE_FMT)}"
        )
        print(
            f"📅 You can fetch news up to {max_days} days back from "
            f"{end_date.strftime(DATE_FMT)}"
        )
    else:
        max_days = DEFAULT_MAX_DAYS_NO_DATA
        print(
            f"⚠️  Could not determine data range, defaulting to "
            f"{DEFAULT_MAX_DAYS_NO_DATA} days max"
        )

    # Streamlined input validation with helper function
    def get_valid_int(prompt, min_val, max_val):
        """Get valid integer input within specified range."""
        while True:
            try:
                value = int(input(prompt).strip())
                if min_val <= value <= max_val:
                    return value
                print(
                    f"Please enter a number between {min_val} and "
                    f"{max_val}."
                )
            except ValueError:
                print("Please enter a valid number.")

    # Limit to 14 days max due to API cost constraints
    limited_max_days = min(max_days, 14)
    days_back = get_valid_int(
        f"How many days back should we fetch news from? (1-{limited_max_days}, limited due to API costs): ",
        1,
        limited_max_days,
    )
    # Fixed to 1 article per day to avoid overloading DeepSeek API
    articles_per_day = 1
    print(f"📄 Using 1 article per day to avoid API overload")

    news_data = fetch_news_data(
        query=search_query,
        articles_per_day=articles_per_day,
        days_back=days_back,
        end_date=end_date,
    )
    if not news_data:
        print("❌ Failed to fetch news data")
        return

    article_count = len(news_data.get('articles', []))
    print(f"📰 Fetched {article_count} articles")

    if not save_raw_data(news_data, RAW_NEWS_FILENAME):
        print("⚠️  Failed to save raw news JSON")

    df = create_articles_dataframe(news_data)
    if not df.empty:
        print("🤖 Starting sentiment analysis...")
        from deepseek_enrichment import enrich_dataframe_with_sentiment
        enriched_df = enrich_dataframe_with_sentiment(
            df,
            fetch_solana_price_data,
            persona=selected_persona,
        )

        if enriched_df is not None:
            out_path = os.path.join(ENRICHED_DIR, ENRICHED_CSV_FILENAME)
            try:
                enriched_df.to_csv(out_path, index=False)
                print(
                    "✅ Analysis complete! Check "
                    "data/enriched/enriched_data.csv"
                )
            except Exception as exc:
                print(f"❌ Failed to write enriched CSV: {exc}")
        else:
            print("❌ Sentiment analysis failed")
    else:
        print("❌ No articles to process")


if __name__ == "__main__":
    main()
