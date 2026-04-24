#!/usr/bin/env python3
"""
Snowflake + Local LLM Integration Demo
Analyze IPL data using local NLP models
"""

import torch
from transformers import pipeline
import snowflake.connector
import warnings
warnings.filterwarnings("ignore")

def main():
    print("🏏 Snowflake + Local LLM Integration Demo")
    print("=" * 50)

    # Initialize local models
    print("Loading local NLP models...")
    sentiment_analyzer = pipeline("sentiment-analysis",
                               model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    emotion_classifier = pipeline("text-classification",
                                model="j-hartmann/emotion-english-distilroberta-base")

    print("✅ Models loaded successfully!")

    # Example IPL data (simulated - replace with actual Snowflake queries)
    sample_comments = [
        "What an amazing innings by Virat Kohli! CSK are unstoppable!",
        "Disappointing loss for RCB today. Need to improve batting.",
        "The crowd at Chepauk is absolutely electric! Love CSK fans!",
        "Nervous about the playoffs. Who will win the IPL this year?",
        "MS Dhoni's captaincy is legendary. CSK deserved the win!"
    ]

    print("\n📊 Analyzing IPL Fan Comments:")
    print("-" * 40)

    for i, comment in enumerate(sample_comments, 1):
        print(f"\n{i}. Comment: '{comment[:60]}...'")

        # Sentiment Analysis
        sentiment = sentiment_analyzer(comment)[0]
        print(f"   Sentiment: {sentiment['label']} ({sentiment['score']:.3f})")

        # Emotion Detection
        emotion = emotion_classifier(comment)[0]
        print(f"   Emotion: {emotion['label']} ({emotion['score']:.3f})")

    print("\n🔗 Integration with Snowflake:")
    print("-" * 40)
    print("""
    # Example: Query IPL comments from Snowflake and analyze

    import snowflake.connector

    # Connect to Snowflake
    conn = snowflake.connector.connect(
        account='your_account',
        user='your_user',
        password='your_password',
        database='IPL_ANALYSIS',
        schema='YEAR_2024_2025'
    )

    # Query fan comments
    cursor = conn.cursor()
    cursor.execute("SELECT comment_text FROM fan_comments WHERE match_id = %s", (match_id,))

    # Analyze each comment
    for row in cursor:
        comment = row[0]
        sentiment = sentiment_analyzer(comment)[0]
        emotion = emotion_classifier(comment)[0]

        # Store analysis back in Snowflake
        cursor.execute('''
            INSERT INTO comment_analysis (comment_text, sentiment, emotion, confidence)
            VALUES (%s, %s, %s, %s)
        ''', (comment, sentiment['label'], emotion['label'], sentiment['score']))

    conn.commit()
    cursor.close()
    conn.close()
    """)

    print("\n💡 Use Cases:")
    print("   • Real-time sentiment analysis of match commentary")
    print("   • Fan engagement analysis from social media")
    print("   • Automated content moderation")
    print("   • Player performance sentiment tracking")
    print("   • Match prediction based on fan mood")

    print("\n✅ Local LLM integration ready!")
    print("Your venv now supports both Snowflake data access and local AI models!")

if __name__ == "__main__":
    main()