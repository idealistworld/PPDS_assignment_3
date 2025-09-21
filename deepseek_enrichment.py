# Library imports with specific reasoning:
# os: Environment variable access for secure API key management
# json: JSON parsing for API responses and error handling
# pandas: DataFrame operations for efficient batch data processing
# requests: HTTP client for DeepSeek API calls with session management
# time: Rate limiting and exponential backoff implementation
# datetime: Date arithmetic for price data correlation
# typing: Type hints for better code documentation and error prevention
# dotenv: Environment variable loading for API credentials
# tqdm: Progress tracking for long-running sentiment analysis batches

import os
import json
import pandas as pd
import requests
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
from tqdm import tqdm
import random  # For jitter in exponential backoff


# Constants
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TIMEOUT_SECONDS = 30
DS_MAX_RETRIES = 3
DS_BASE_DELAY_SECONDS = 1
BATCH_SLEEP_SECONDS = 2
TOKENS_PER_ARTICLE = 200
SINGLE_MAX_TOKENS = 100
BULLISH_THRESHOLD = 2.0
BEARISH_THRESHOLD = -2.0
DATE_FMT = '%Y-%m-%d'
TQDM_NCOLS = 80

# Shared prompt schema block (kept identical across batch/single)
PROMPT_JSON_SCHEMA = (
    '{\n'
    '  "sentiment": "bullish|bearish|neutral",\n'
    '  "score": float_between_-1_and_1,\n'
    '  "predicted_change_percent": float_price_change_prediction,\n'
    '  "reasoning": "brief_explanation_in_10_words_or_less"\n'
    '}\n'
)

# Full prompt templates (batch and single)
PROMPT_MULTI_TEMPLATE = (
    "{persona_context}Analyze the sentiment of these Solana cryptocurrency\n"
    "news articles and predict price impact.\n"
    "For each article, respond with a JSON object in this exact format:\n"
    "{schema}\n\n"
    "Respond with a JSON array containing one object per article in order.\n\n"
    "{articles_text}\n"
)

PROMPT_SINGLE_TEMPLATE = (
    "{persona_context}Analyze the sentiment of this Solana cryptocurrency\n"
    "news article and predict price impact.\n"
    "Respond with ONLY a JSON object in this exact format:\n"
    "{schema}\n\n"
    "Article text: {text_excerpt}\n"
)

def calculate_prediction_accuracy(
    predicted_sentiment: str,
    actual_change_percent: float,
) -> tuple[float, str]:
    """
    Calculate accuracy score for sentiment prediction vs actual price change.

    Args:
        predicted_sentiment: Predicted sentiment (bullish/bearish/neutral).
        actual_change_percent: Actual price change percentage.

    Returns:
        tuple: (accuracy_score, letter_grade)

    Why:
        Provides a simple, interpretable measure to evaluate how well the
        model’s directional predictions match subsequent market moves.
    """
    # Define thresholds for what constitutes significant movement
    bullish_threshold = BULLISH_THRESHOLD  # 2% increase
    bearish_threshold = BEARISH_THRESHOLD  # 2% decrease

    if predicted_sentiment == "bullish":
        if actual_change_percent >= bullish_threshold:
            accuracy = 1.0  # Perfect prediction
        elif actual_change_percent > 0:
            accuracy = 0.7  # Correct direction, smaller magnitude
        elif actual_change_percent >= bearish_threshold:
            accuracy = 0.3  # Neutral territory
        else:
            accuracy = 0.0  # Wrong direction

    elif predicted_sentiment == "bearish":
        if actual_change_percent <= bearish_threshold:
            accuracy = 1.0  # Perfect prediction
        elif actual_change_percent < 0:
            accuracy = 0.7  # Correct direction, smaller magnitude
        elif actual_change_percent <= bullish_threshold:
            accuracy = 0.3  # Neutral territory
        else:
            accuracy = 0.0  # Wrong direction

    else:  # neutral
        if bearish_threshold < actual_change_percent < bullish_threshold:
            accuracy = 1.0  # Correct neutral prediction
        elif -5.0 < actual_change_percent < 5.0:
            accuracy = 0.5  # Close to neutral
        else:
            accuracy = 0.0  # Significant movement when predicted neutral

    # Convert to letter grade
    if accuracy >= 0.95:
        grade = "A+"
    elif accuracy >= 0.9:
        grade = "A"
    elif accuracy >= 0.8:
        grade = "A-"
    elif accuracy >= 0.75:
        grade = "B+"
    elif accuracy >= 0.7:
        grade = "B"
    elif accuracy >= 0.65:
        grade = "B-"
    elif accuracy >= 0.6:
        grade = "C+"
    elif accuracy >= 0.5:
        grade = "C"
    elif accuracy >= 0.4:
        grade = "C-"
    elif accuracy >= 0.3:
        grade = "D"
    else:
        grade = "F"

    return accuracy, grade


def analyze_multiple_articles(
    articles_data: List[Dict[str, str]],
    persona=None,
) -> List[Dict[str, Any]]:
    """
    Analyze sentiment for multiple articles in a single API call
    with exponential backoff.

    Batch processing advantages:
    1. Reduces API calls from 20+ individual requests to 1 batch request
    2. Lower latency and faster processing
    3. More efficient token usage
    4. Consistent analysis context across articles

    Implements exponential backoff for resilience against:
    - Rate limiting (429 errors)
    - Temporary API unavailability (503 errors)
    - Network timeouts

    Args:
        articles_data: List of dicts with 'title' and 'description'.
        persona: Optional tuple of (name, description) for perspective.

    Returns:
        List of sentiment analysis results with fallback handling

    Why:
        Batching reduces API calls and latency, keeps analysis context
        consistent across items, and improves robustness under rate limits.
    """
    load_dotenv()
    api_key = os.getenv('DEEPSEEK_API_KEY')

    if not api_key:
        print("Error: DEEPSEEK_API_KEY not found in environment variables")
        return [
            {
                "sentiment": "neutral",
                "score": 0.0,
                "reasoning": "API key missing",
            }
            for _ in articles_data
        ]

    url = DEEPSEEK_URL
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Prepare articles for batch processing
    articles_text = ""
    for index, article in enumerate(articles_data, 1):
        title = article.get('title', '')
        description = article.get('description', '')
        snippet = article.get('snippet', '')
        text = f"{title} {description} {snippet}".strip()
        articles_text += f"\nArticle {index}: {text[:800]}\n"

    # Build persona-specific prompt
    persona_context = ""
    if persona:
        persona_name, persona_desc = persona
        persona_context = f"Analyze as a {persona_name}: {persona_desc}. "

    prompt = PROMPT_MULTI_TEMPLATE.format(
        persona_context=persona_context,
        schema=PROMPT_JSON_SCHEMA,
        articles_text=articles_text,
    )

    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        # Scale with number of articles
        "max_tokens": TOKENS_PER_ARTICLE * len(articles_data),
        "temperature": 0.1,
    }

    # Implement exponential backoff for API resilience
    max_retries = DS_MAX_RETRIES
    base_delay = DS_BASE_DELAY_SECONDS

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=DEEPSEEK_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            break  # Success, exit retry loop

        except requests.exceptions.HTTPError as exc:
            if response.status_code == 429:  # Rate limit exceeded
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(
                        f"⏳ Rate limit hit, waiting {delay:.1f}s "
                        f"before retry {attempt + 1}/{max_retries}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    print("❌ Max retries exceeded for rate limiting")
                    return [
                        {
                            "sentiment": "neutral",
                            "score": 0.0,
                            "predicted_change_percent": 0.0,
                            "reasoning": "Rate limit exceeded",
                        }
                        for _ in articles_data
                    ]
            else:
                print(f"❌ HTTP error {response.status_code}: {exc}")
                return [
                    {
                        "sentiment": "neutral",
                        "score": 0.0,
                        "predicted_change_percent": 0.0,
                        "reasoning": "API error",
                    }
                    for _ in articles_data
                ]

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as exc:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"🔄 Network error, retrying in {delay}s: {exc}")
                time.sleep(delay)
                continue
            else:
                print(f"❌ Network error after {max_retries} retries: {exc}")
                return [
                    {
                        "sentiment": "neutral",
                        "score": 0.0,
                        "predicted_change_percent": 0.0,
                        "reasoning": "Network error",
                    }
                    for _ in articles_data
                ]

    try:
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()

        # Clean up the content
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()

        # Parse JSON response
        try:
            sentiment_results = json.loads(content)

            # Ensure we have a list
            if not isinstance(sentiment_results, list):
                sentiment_results = [sentiment_results]

            # Fill missing results with defaults
            while len(sentiment_results) < len(articles_data):
                sentiment_results.append(
                    {
                        "sentiment": "neutral",
                        "score": 0.0,
                        "reasoning": "Analysis incomplete",
                    }
                )

            # Validate each result
            for result in sentiment_results:
                for field in [
                    "sentiment",
                    "score",
                    "predicted_change_percent",
                    "reasoning",
                ]:
                    if field not in result:
                        if field == "sentiment":
                            result[field] = "neutral"
                        elif field in ["score", "predicted_change_percent"]:
                            result[field] = 0.0
                        else:
                            result[field] = "No explanation provided"

            # Return only as many as requested
            return sentiment_results[: len(articles_data)]

        except json.JSONDecodeError:
            print(f"Failed to parse batch JSON response: {content}")
            # Fallback to individual analysis
            return [
                analyze_sentiment_single(
                    f"{article.get('title', '')} "
                    f"{article.get('description', '')}",
                    persona,
                )
                for article in articles_data
            ]

    except Exception as exc:
        print(f"Error in batch analysis: {exc}")
        return [
            {
                "sentiment": "neutral",
                "score": 0.0,
                "predicted_change_percent": 0.0,
                "reasoning": "API error",
            }
            for _ in articles_data
        ]


def analyze_sentiment_single(text: str, persona=None) -> Dict[str, Any]:
    """
    Analyze sentiment of a single text (fallback function).

    Why:
        Ensures the pipeline can still enrich items when batch responses
        are malformed or unavailable by providing a resilient fallback.
    """
    load_dotenv()
    api_key = os.getenv('DEEPSEEK_API_KEY')

    if not api_key:
        return {
            "sentiment": "neutral",
            "score": 0.0,
            "reasoning": "API key missing",
        }

    url = DEEPSEEK_URL
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    persona_context = ""
    if persona:
        persona_name, persona_desc = persona
        persona_context = (
            f"Analyze as a {persona_name}: {persona_desc}. "
        )

    prompt = PROMPT_SINGLE_TEMPLATE.format(
        persona_context=persona_context,
        schema=PROMPT_JSON_SCHEMA,
        text_excerpt=text[:1000],
    )

    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": SINGLE_MAX_TOKENS,
        "temperature": 0.1
    }

    # Same exponential backoff strategy as batch processing
    max_retries = DS_MAX_RETRIES
    base_delay = DS_BASE_DELAY_SECONDS

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=DEEPSEEK_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            break

        except requests.exceptions.HTTPError as exc:
            if response.status_code == 429 and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(
                    f"⏳ Single analysis rate limited, waiting {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            else:
                return {
                    "sentiment": "neutral",
                    "score": 0.0,
                    "predicted_change_percent": 0.0,
                    "reasoning": "API error",
                }

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as exc:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(
                    f"🔄 Single analysis network error, retrying in {delay}s"
                )
                time.sleep(delay)
                continue
            else:
                return {
                    "sentiment": "neutral",
                    "score": 0.0,
                    "predicted_change_percent": 0.0,
                    "reasoning": "Network error",
                }

    try:

        result = response.json()
        content = result['choices'][0]['message']['content'].strip()

        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()

        sentiment_data = json.loads(content)

        # Validate required fields
        for field in [
            "sentiment",
            "score",
            "predicted_change_percent",
            "reasoning",
        ]:
            if field not in sentiment_data:
                if field == "sentiment":
                    sentiment_data[field] = "neutral"
                elif field in ["score", "predicted_change_percent"]:
                    sentiment_data[field] = 0.0
                else:
                    sentiment_data[field] = "No explanation provided"

        return sentiment_data

    except Exception as exc:
        return {
            "sentiment": "neutral",
            "score": 0.0,
            "predicted_change_percent": 0.0,
            "reasoning": "Analysis failed",
        }


def enrich_dataframe_with_sentiment(
    df: pd.DataFrame,
    fetch_price_data_func,
    persona=None,
    batch_size: int = 3,
) -> pd.DataFrame:
    """
    Enrich DataFrame with sentiment analysis using batch processing
    and rate limiting.

    This is the main orchestration function that:
    1. Processes articles in batches for efficiency
    2. Applies rate limiting between batches
    3. Fetches corresponding price data for accuracy measurement
    4. Handles errors gracefully with fallback values
    5. Provides progress tracking for user experience

    Rate limiting strategy:
    - 2-second delay between batches to respect API limits
    - Exponential backoff within each batch for retry logic
    - Graceful degradation if APIs fail

    Args:
        df: Input DataFrame with articles
        fetch_price_data_func: Function to get price data for date correlation
        persona: Optional analysis perspective (name, description)
        batch_size: Number of articles to process in each API call (default 3)

    Returns:
        pd.DataFrame: Enriched DataFrame with sentiment and price data

    Why:
        Central orchestration that scales enrichment efficiently, respects
        API limits, and aligns AI outputs with price data for evaluation.
    """
    try:
        # Add analysis columns
        sentiments = []
        scores = []
        predicted_changes = []
        reasonings = []
        actual_prices = []
        price_changes = []
        price_change_percents = []
        accuracy_scores = []
        accuracy_grades = []

        # Process articles in batches
        for start_idx in tqdm(
            range(0, len(df), batch_size),
            desc="🤖 AI analysis",
            unit="batch",
            ncols=TQDM_NCOLS,
        ):
            batch_df = df.iloc[start_idx:start_idx + batch_size]
            batch_articles = []

            for _, row in batch_df.iterrows():
                # Extract content from raw JSON for analysis
                raw_json = row.get('raw_article_json', '{}')
                try:
                    article_data = json.loads(raw_json)
                    description = article_data.get('description', '')
                    snippet = article_data.get('snippet', '')
                except json.JSONDecodeError:
                    description = ''
                    snippet = ''

                batch_articles.append({
                    'title': row.get('title', ''),
                    'description': description,
                    'snippet': snippet
                })


            # Analyze the batch with persona
            batch_results = analyze_multiple_articles(batch_articles, persona)

            # Add results to lists and fetch price data
            for idx, (result, (_, row)) in enumerate(
                zip(batch_results, batch_df.iterrows())
            ):
                sentiment = result.get('sentiment', 'neutral')
                sentiments.append(sentiment)
                scores.append(result.get('score', 0.0))
                predicted_changes.append(
                    round(result.get('predicted_change_percent', 0.0), 2)
                )
                reasonings.append(
                    result.get('reasoning', 'No explanation provided')
                )

                # Fetch actual price data for this article's date
                article_date = row.get('published_at', '')
                if article_date:
                    try:
                        # Convert to date string if it's a datetime
                        if isinstance(article_date, str):
                            article_date = pd.to_datetime(
                                article_date
                            ).strftime(DATE_FMT)
                        else:
                            article_date = article_date.strftime(DATE_FMT)

                        price_data = fetch_price_data_func(article_date)
                        actual_price = price_data['price']
                        price_change = price_data['price_change']
                        price_change_percent = price_data[
                            'price_change_percent'
                        ]

                    except Exception:
                        actual_price = 0.0
                        price_change = 0.0
                        price_change_percent = 0.0
                else:
                    actual_price = 0.0
                    price_change = 0.0
                    price_change_percent = 0.0

                actual_prices.append(actual_price)
                price_changes.append(price_change)
                price_change_percents.append(price_change_percent)

                # Calculate prediction accuracy
                predicted_sentiment = result.get('sentiment', 'neutral')
                predicted_change = result.get('predicted_change_percent', 0.0)
                accuracy_score, accuracy_grade = calculate_prediction_accuracy(
                    predicted_sentiment, price_change_percent
                )
                accuracy_scores.append(accuracy_score)
                accuracy_grades.append(accuracy_grade)

            # Rate limiting between batches to reduce 429 errors
            time.sleep(BATCH_SLEEP_SECONDS)

        # Ensure we have the right number of results with default values
        def pad_results(results_list, default_value, target_length):
            """Helper to pad result lists to target length."""
            while len(results_list) < target_length:
                results_list.append(default_value)

        # Pad all result lists to match DataFrame length
        target_len = len(df)
        pad_results(sentiments, 'neutral', target_len)
        pad_results(scores, 0.0, target_len)
        pad_results(predicted_changes, 0.0, target_len)
        pad_results(reasonings, 'Analysis incomplete', target_len)
        pad_results(actual_prices, 0.0, target_len)
        pad_results(price_changes, 0.0, target_len)
        pad_results(price_change_percents, 0.0, target_len)
        pad_results(accuracy_scores, 0.0, target_len)
        pad_results(accuracy_grades, 'F', target_len)

        # Add new columns in logical order
        df_enriched = df.copy()
        df_enriched['sentiment'] = sentiments[:len(df)]
        df_enriched['sentiment_score'] = scores[:len(df)]
        df_enriched['predicted_change_percent'] = predicted_changes[:len(df)]
        df_enriched['reasoning'] = reasonings[:len(df)]
        df_enriched['actual_price_usd'] = actual_prices[:len(df)]
        df_enriched['actual_change_percent'] = price_change_percents[:len(df)]
        df_enriched['prediction_accuracy'] = accuracy_scores[:len(df)]
        df_enriched['accuracy_grade'] = accuracy_grades[:len(df)]

        # Reorder columns for optimal human readability (most important first)
        column_order = [
            # Core article info (what you see first)
            'title',
            'published_at',
            'source_name',

            # Key analysis results (what you care about most)
            'sentiment',
            'predicted_change_percent',
            'actual_change_percent',
            'accuracy_grade',

            # Supporting details (context and validation)
            'sentiment_score',
            'prediction_accuracy',
            'reasoning',
            'actual_price_usd',

            # Reference data (least frequently needed)
            'url',
            'raw_article_json'
        ]

        df_enriched = df_enriched[column_order]
        return df_enriched

    except Exception as exc:
        print(f"Error enriching DataFrame: {exc}")
        return None
