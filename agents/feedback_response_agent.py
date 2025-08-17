#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive Feedback Response Agent (no API key)
- Prompts the user to enter feedback
- Classifies sentiment and generates polite reply
"""

# ---------------------------
# Suppress warnings & logging
# ---------------------------
import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

# ---------------------------
# Imports
# ---------------------------
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from transformers import pipeline

# ---------------------------
# Models
# ---------------------------
_SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # neg/neu/pos
_GEN_MODEL_ID = "distilgpt2"  # optional local generator

_sentiment_pipe = None
_gen_pipe = None


def get_sentiment_pipe():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline(
            task="text-classification",
            model=_SENTIMENT_MODEL_ID,
            tokenizer=_SENTIMENT_MODEL_ID,
            return_all_scores=True,
            truncation=True,
        )
    return _sentiment_pipe


def get_gen_pipe():
    global _gen_pipe
    if _gen_pipe is None:
        _gen_pipe = pipeline(task="text-generation", model=_GEN_MODEL_ID)
    return _gen_pipe

# ---------------------------
# Aspect & sentiment logic
# ---------------------------
ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "food": ["food", "taste", "noodle", "soup", "broth", "dish", "meal", "menu", "flavor", "fresh", "cold", "hot"],
    "service": ["service", "staff", "waiter", "server", "slow", "rude", "friendly", "attentive"],
    "ambiance": ["ambience", "atmosphere", "music", "noise", "crowd", "clean", "dirty", "decor", "lighting", "seat"],
    "price": ["price", "cost", "expensive", "cheap", "value", "overpriced", "bill"],
    "delivery": ["delivery", "takeaway", "pickup", "packaging"]
}

NEG_STRONG_TERMS = ["terrible", "awful", "horrible", "worst", "disgusting", "refund", "complaint", "unacceptable"]


def detect_aspects(text: str, max_aspects: int = 2) -> List[str]:
    text_l = text.lower()
    hits: List[Tuple[str, int]] = []
    for aspect, kws in ASPECT_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in text_l)
        if score > 0:
            hits.append((aspect, score))
    hits.sort(key=lambda x: x[1], reverse=True)
    aspects = [a for a, _ in hits[:max_aspects]]
    if not aspects:
        return ["no specific aspect"]  # avoid "general" in reply
    return aspects


def is_strong_negative(text: str) -> bool:
    t = text.lower()
    return any(term in t for term in NEG_STRONG_TERMS)


@dataclass
class SentimentResult:
    label: str
    confidence: float


def classify_sentiment(review: str) -> SentimentResult:
    pipe = get_sentiment_pipe()
    scores = pipe(review)[0]
    top = max(scores, key=lambda s: s["score"])
    return SentimentResult(label=top["label"], confidence=float(top["score"]))


def build_template_reply(
    review: str,
    sentiment: SentimentResult,
    aspects: List[str],
    customer_name: Optional[str] = None
) -> str:
    name_prefix = f"Hi {customer_name}," if customer_name else "Hi there,"
    aspect_str = ", ".join(aspects)

    if sentiment.label == "positive":
        body = f" thank you for the wonderful feedback! We're thrilled that you enjoyed the {aspect_str}. Your kind words mean a lot to our team."
        closing = "Warm thanks, SteamNoodles Team"
    elif sentiment.label == "neutral":
        body = f" thanks for sharing your thoughts. We will keep your notes about the {aspect_str} in mind as we improve."
        closing = "Kind regards, SteamNoodles Team"
    else:
        apology = "We're sorry" if not is_strong_negative(review) else "We’re truly sorry"
        body = f" {apology} that your experience with the {aspect_str} didn't meet expectations. Your feedback helps us improve."
        closing = "Sincerely, SteamNoodles Support"

    return f"{name_prefix}{body}\n\n{closing}"


def maybe_paraphrase(reply: str, enable_generation: bool) -> str:
    if not enable_generation:
        return reply
    gen = get_gen_pipe()
    prompt = f"Paraphrase politely and concisely (2 sentences max). Original: {reply}\nParaphrase:"
    out = gen(prompt, max_new_tokens=70, do_sample=True, temperature=0.7, top_p=0.9, num_return_sequences=1)[0]["generated_text"]
    paraphrased = out.split("Paraphrase:", 1)[-1].strip()
    return paraphrased


def generate_feedback_response(review_text: str, customer_name: Optional[str] = None, use_local_generation: bool = False):
    sent = classify_sentiment(review_text)
    aspects = detect_aspects(review_text)
    reply_base = build_template_reply(review_text, sent, aspects, customer_name)
    reply_final = maybe_paraphrase(reply_base, enable_generation=use_local_generation)
    print("\n=== Classification ===")
    print(f"Sentiment : {sent.label} (confidence={sent.confidence:.3f})")
    print("\n=== Auto-Reply ===")
    print(reply_final)


# ---------------------------
# Interactive prompt
# ---------------------------
def main():
    print("=== SteamNoodles Feedback Response Agent ===")
    customer_name = input("Enter customer name (or leave blank): ").strip() or None
    review = input("Enter customer feedback: ").strip()
    generate_feedback_response(review, customer_name, use_local_generation=False)


if __name__ == "__main__":
    main()
