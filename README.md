# NOTES

For the purposes of using this as demo please

1. Use the first option called "1. General Ecosystem (FOR DEMO USE THIS BECAUSE THIS API HAS LIMITED VOLUME)". This is because the amount of articles on this lower cost news API (which I used due to the robust ones being too $$$ and I already spent $ to even get this one) are sometimes spotty when it comes to article volume. So as a proof of concept I included the other options (which do work) just sometimes the data can be unreliable.
2. While there are 730 days or years of Solana price data, I will get rate limited and run out of credits if this is the option selected. Please select a number between 1 and 14 for the sake of the demo since I don't have full access to a stronger API.

# Clear description of data sources and purpose

1. Crypto Price Data CSV (from Kaggle)
   This data includes hourly prices for different crypto currencies. I'm most interested in the Solana ecosystem so that's the one we'll be focusing on. In the document the columns represent different currencies at either closing or opening prices and the rows each represent different times these prices were observed at. Data is returned in the csv file.

2. TheNewsAPI Data
   This API lets us retrieve Solana‑related news articles for analysis. It
   provides JSON responses that we save under `data/raw/` before enrichment.

   Project Purpose
   As someone who invests a lot in Solana across different segments of the industry (i.e. staking, defi, etc) I try my best to stay on top of news. At times though it's hard to do this on scale and aggregate the sentiment for a specific area in the Solana ecosystem. I wanted to build a project that at scale can aggregate news data to come to conclusions around bullish and bearish sentiment. The only issue is with pure scraping it's hard to understand if an article is likely to have a positive or negative impact on the price of the token. That's where the use of LLMs comes in. I leverage Deepseek here to give an assigned bullish or bearish score as well as the extremeness of this to get an idea of how the news might impact Solana's price. Additionally I wanted to offer the ability to learn about different areas of Solana as well as read the news from the perspective of different personas. Sometimes the LLM might be biased so getting a spread of the opinions of "roleplaying" different personas would be useful. On scale I'd use a more in-depth news api, but this is the best I could find for a reasonable price for this project. I also allow the user to select the time range and amount of articles they'd like per day and limited it for the sake of this project, but in the future with more API credits I'd expand it.

# Explanation of DeepSeek enhancements

I use Deepseek here to "roleplay" the perspective of different financial analysts to show a full spectrum of how articles might be interpreted. I tried to include an array of personas / characters that would make it more interesting. On one side you have more traditional financial analysts and on the other crypto advocates. It's a fun mini prediction data pipeline. After all this analysis the LLM provides a summary for this reasoning along with price predictions. We then compare this to the actual price of Solana and give the analyst a grade on how accurately they predicted an article would impact price.

# Before/after examples showing AI value

## Raw Article API Data (Before)

```
{"uuid": "44bdeac5-6252-43df-9667-451cff127be5",
"title": "Solana Price Prediction 2025\u20132050: 500% Gains by 2050 \u2013 Is It Worth Investing?",
"description": "Solana (SOL) could hit $850 by 2050, a 500% gain from 2025 levels, driven by mass adoption, scalability, and strong developer ecosystem.",
"keywords": "",
"snippet": "Key Takeaways:\n\nSolana (SOL) could hit $850 by 2050 , a 500% gain from 2025 levels, driven by mass adoption, scalability, and strong developer ecosystem.\n\nThe r...",
"url": "https://www.cryptoninjas.net/news/solana-price-prediction-2025-2050/",
"image_url": "https://www.cryptoninjas.net/wp-content/uploads/In-the-center-Solana-Price-Prediction-2025\u20132050\u2026.jpeg", "language": "en",
"published_at": "2025-07-29T12:43:53.000000Z",
"source": "cryptoninjas.net",
"categories": [],
"relevance_score": 25.15004}
```

## Raw Crypto Price Data (Before)

examples/raw_crypto_sample.csv

```
timestamp,BNB_close,BTC_close,ETH_close,SOL_close,XRP_close,BNB_high,BTC_high,ETH_high,SOL_high,XRP_high,BNB_low,BTC_low,ETH_low,SOL_low,XRP_low,BNB_open,BTC_open,ETH_open,SOL_open,XRP_open
2022-11-23 04:59:59.999000+00:00,273.38543714333775,16510.6873626981,1163.064745228847,12.917000248738187,0.37785134993630237,274.33225523523885,16568.179278833977,1167.119882870318,13.074110005533974,0.378053320735828,272.9819689389209,16496.73519700166,1161.9242126446738,12.91247400408389,0.3766212732931187,273.97314699324653,16561.33978681142,1165.247344844491,13.050830061742431,0.3771916991045406
```

## Enriched Dataset Row (After)

examples/after.csv

```
title,published_at,source_name,sentiment,predicted_change_percent,actual_change_percent,sentiment_score,reasoning,actual_price_usd,url
"Solana DEX volume surges as new AMM launches",2024-09-20T14:05:00Z,CryptoExample News,bullish,2.3,1.8,0.62,"Strong activity suggests upward momentum",19.42,https://news.example.com/solana-amm-launch
```

What AI added

- Sentiment: bullish/bearish/neutral label per article.
- Confidence: sentiment score in [-1, 1].
- Predicted impact: estimated price change percent.
- Reasoning: brief justification in plain language.

# Installation and usage instructions

## Before Installation

- Python 3.10+ installed (check with `python --version`).
- A virtual environment tool available (`venv` is built‑in).
- API keys ready (do NOT commit them):
  - `NEWS_API` for TheNewsAPI
  - `DEEPSEEK_API_KEY` for DeepSeek

### Setup Steps

1. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```
*Note: You should see `(venv)` in your terminal prompt when activated*

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
# Create .env file with your API keys (do not commit)
cat > .env << 'EOF'
NEWS_API=YOUR_THENEWSAPI_KEY
DEEPSEEK_API_KEY=YOUR_DEEPSEEK_KEY
EOF
```

## Running the Application

**IMPORTANT: Always activate the virtual environment first:**
```bash
source venv/bin/activate
python3 main.py
```

If you see `ModuleNotFoundError`, you forgot to activate the virtual environment.

- Follow prompts for focus area, persona, and date/article counts.
- Outputs:
  - Raw news JSON: `data/raw/solana_news.json`
  - Enriched CSV: `data/enriched/enriched_data.csv`
- Examples for AI value:
  - `examples/before.json` (raw article)
  - `examples/raw_crypto_sample.csv` (raw crypto price)
  - `examples/after.csv` (enriched)
