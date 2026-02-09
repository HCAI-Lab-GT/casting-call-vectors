#!/usr/bin/env python
"""
Generate extraction questions designed to differentiate personas.

These questions are deliberately open-ended and probe values, preferences,
reasoning styles, and emotional responses -- dimensions where different
character archetypes should diverge.

Based on methodology from arXiv 2601.10387 (Christina Lu et al).

Usage:
  python scripts/marin/assistant_axis/generate_questions.py
  python scripts/marin/assistant_axis/generate_questions.py --num_questions 240
"""

import argparse
import json
from pathlib import Path

# Prototype: 48 questions across different dimensions.
# Scale to 240 for full run.
DEFAULT_QUESTIONS = [
    # Values & Priorities
    "What matters most to you in life?",
    "If you could change one thing about the world, what would it be?",
    "What does success mean to you?",
    "What is worth sacrificing comfort for?",
    "What is the most important quality in a person?",
    "What would you want your legacy to be?",

    # Decision-Making & Reasoning
    "How do you decide what to do when facing an impossible choice?",
    "What role does intuition play in your decision-making?",
    "How do you handle situations where the evidence is ambiguous?",
    "When is it better to act quickly versus carefully?",
    "How do you weigh the needs of the many against the needs of the few?",
    "What is the biggest mistake people make when solving problems?",

    # Emotional & Social
    "How do you handle conflict with someone you care about?",
    "What makes you angry?",
    "How do you comfort someone who is grieving?",
    "What is the hardest part about working with other people?",
    "How do you react when someone disagrees with you strongly?",
    "What does trust look like in your relationships?",

    # Creativity & Expression
    "Describe something beautiful you've experienced recently.",
    "How do you approach a creative problem?",
    "What role does art play in society?",
    "If you could create something that would last forever, what would it be?",
    "How do you find inspiration when you feel stuck?",
    "What is the relationship between constraint and creativity?",

    # Knowledge & Learning
    "What is the most important thing you've learned?",
    "How do you know when you truly understand something?",
    "What is the value of knowledge that has no practical application?",
    "How do you approach learning something completely new?",
    "What is the difference between wisdom and intelligence?",
    "What topic could you talk about for hours?",

    # Risk & Uncertainty
    "How do you approach risk in your life?",
    "What is your relationship with failure?",
    "How do you prepare for the unknown?",
    "When is caution more dangerous than boldness?",
    "How do you maintain composure under pressure?",
    "What is the most difficult situation you've had to navigate?",

    # Identity & Self
    "How would you describe yourself in three words?",
    "What shaped who you are more than anything else?",
    "What do you struggle with most about yourself?",
    "How have you changed over the past decade?",
    "What belief have you held that turned out to be wrong?",
    "What are you most proud of?",

    # Philosophy & Meaning
    "What is the nature of consciousness?",
    "Is there objective morality, or is it all relative?",
    "What gives life meaning?",
    "How should we think about our obligations to future generations?",
    "What is the relationship between freedom and responsibility?",
    "Is it possible to be truly selfless?",
]


def main():
    parser = argparse.ArgumentParser(description="Generate extraction questions.")
    parser.add_argument(
        "--num_questions",
        type=int,
        default=48,
        help="Number of questions to include.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/assistant_axis/questions.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    questions = DEFAULT_QUESTIONS[:args.num_questions]

    # Group questions by category for analysis
    categories = [
        "values", "values", "values", "values", "values", "values",
        "reasoning", "reasoning", "reasoning", "reasoning", "reasoning", "reasoning",
        "emotional", "emotional", "emotional", "emotional", "emotional", "emotional",
        "creativity", "creativity", "creativity", "creativity", "creativity", "creativity",
        "knowledge", "knowledge", "knowledge", "knowledge", "knowledge", "knowledge",
        "risk", "risk", "risk", "risk", "risk", "risk",
        "identity", "identity", "identity", "identity", "identity", "identity",
        "philosophy", "philosophy", "philosophy", "philosophy", "philosophy", "philosophy",
    ]

    output_data = {
        "questions": [
            {"id": i, "text": q, "category": categories[i] if i < len(categories) else "other"}
            for i, q in enumerate(questions)
        ],
        "metadata": {
            "num_questions": len(questions),
            "categories": sorted(set(categories[:len(questions)])),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Generated {len(questions)} questions across {len(output_data['metadata']['categories'])} categories.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
