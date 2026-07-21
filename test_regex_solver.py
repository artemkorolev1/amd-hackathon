#!/usr/bin/env python3
"""
Detailed analysis of regex solver behavior.
"""
import json, os, sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from agent.category_filter import get_short_name
from agent.solvers.deterministic import solve_sentiment as regex_solve_sentiment

# Quick test on a few questions
test_cases = [
    'Classify the sentiment of this review as positive, negative, or neutral: "The battery lasts forever and the screen is gorgeous. Best purchase this year."',
    'Classify the sentiment: Oh brilliant, my flight got canceled. Just the highlight of my day.',
    'Classify the sentiment of this review as positive, negative, or neutral: "Arrived broken, support never replied, and the refund took a month."',
    'Classify the sentiment of this statement as positive, negative, or neutral: "The package was delivered on Tuesday afternoon."',
    'Classify the sentiment of this review as positive, negative, or neutral: "I wanted to love it, but it crashes constantly and deleted my files."',
    "Classify the sentiment of the following text (positive, negative, or neutral):\n\n\"Oh brilliant, my flight got canceled. Just the highlight of my day.\"",
    'Classify the sentiment: I am absolutely thrilled with this purchase! The quality exceeded all my expectations.',
]

for i, prompt in enumerate(test_cases):
    result = regex_solve_sentiment(prompt, "sentiment")
    print(f"Test {i}: pred='{result}'")
    print(f"  prompt: {prompt[:100]}...")
    print()
