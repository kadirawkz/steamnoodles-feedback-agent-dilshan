# SteamNoodles Automated Feedback Agents

## Project Overview

This project implements an automated restaurant feedback system for **SteamNoodles** using a **multi-agent framework**. It includes two agents to enhance customer feedback processing.

### Agents

1. **Feedback Response Agent**

   * Accepts individual customer reviews.
   * Performs sentiment analysis using a local NLP model (`cardiffnlp/twitter-roberta-base-sentiment-latest`).
   * Detects key aspects (food, service, ambiance, price, delivery).
   * Generates polite, context-aware replies.
   * Optional local text generation (`distilgpt2`) to paraphrase replies for variety.

2. **Sentiment Visualization Agent**

   * Reads restaurant reviews dataset.
   * Filters reviews by user-specified date range.
   * Generates a line plot showing daily counts of positive, neutral, and negative reviews.

## Project Details

**Name:** Dilshan Kadira
**University:** \[Your University Name]
**Year:** \[Your Year, e.g., 3rd Year]

### Summary of Approach

The multi-agent CLI allows users to interactively run either agent and view results without requiring any API keys. The Feedback Response Agent provides real-time sentiment analysis and personalized replies, while the Sentiment Visualization Agent helps the restaurant visualize trends in customer feedback over time.

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/kadirawkz/steamnoodles-feedback-agent-dilshan.git
cd steamnoodles-feedback-agent-dilshan
```

2. **Create and activate a virtual environment**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Instructions to Test Both Agents

### 1. Feedback Response Agent

Run the agent using the CLI:

```bash
python agents/feedback_response_agent.py
```

* Enter customer name (optional)
* Enter customer review text
* The agent will output:

  * Sentiment (positive, neutral, negative)
  * Confidence score
  * Auto-generated polite reply

### 2. Sentiment Visualization Agent

Run the agent using the CLI:

```bash
python agents/sentiment_plotting_agent.py
```

* Enter start date (YYYY-MM-DD)
* Enter end date (YYYY-MM-DD)
* The agent will display a line plot showing sentiment trends within the selected date range.

## Files in Repository

* `agents/feedback_response_agent.py` – CLI for feedback sentiment and auto-reply.
* `agents/sentiment_plotting_agent.py` – CLI for plotting sentiment trends.
* `dataset/Yelp Restaurant Reviews.csv` – Sample dataset of restaurant reviews.
* `requirements.txt` – Required Python libraries.
* `README.md` – Project documentation.

## Demo Output

* Sample auto-response for a review.
* Example sentiment trend plot over a date range.

## License

This project is for academic purposes only.
