#!/usr/bin/env python3
"""
IPL 2024 Top Run Scorers Analysis with Local LLM
"""

import snowflake.connector
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

def get_top_run_scorers():
    """Get top 10 run scorers from IPL 2024"""
    try:
        # Connect to Snowflake
        conn = snowflake.connector.connect(
            # Add your connection details here
            account='your_account',
            user='your_user',
            password='your_password',
            database='IPL_ANALYSIS',
            schema='YEAR_2024_2025'
        )

        cursor = conn.cursor()

        # Query top run scorers
        query = """
        SELECT BATTER, SUM(RUNS_BATTER) as TOTAL_RUNS
        FROM IPL_24_25_BALL_BY_BALL
        WHERE SEASON = '2024'
        GROUP BY BATTER
        ORDER BY TOTAL_RUNS DESC
        LIMIT 10
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        return results

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("💡 Using sample data for demonstration")
        return [
            ('V Kohli', 741),
            ('RD Gaikwad', 583),
            ('R Parag', 573),
            ('TM Head', 567),
            ('SV Samson', 531)
        ]

def analyze_with_llm(top_scorers):
    """Analyze top scorers using local LLM"""
    print("🤖 Analyzing IPL 2024 Top Run Scorers with Local LLM")
    print("=" * 60)

    # Load local models
    sentiment_analyzer = pipeline("sentiment-analysis",
                               model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    emotion_classifier = pipeline("text-classification",
                                model="j-hartmann/emotion-english-distilroberta-base")

    # Analyze each top scorer
    for i, (batsman, runs) in enumerate(top_scorers, 1):
        print(f"\n{i}. {batsman}: {runs} runs")

        # Create analysis text
        analysis_text = f"{batsman} scored {runs} runs in IPL 2024, showing excellent batting performance."

        # Sentiment analysis
        sentiment = sentiment_analyzer(analysis_text)[0]
        print(f"   📊 Performance Sentiment: {sentiment['label']} ({sentiment['score']:.3f})")

        # Emotion analysis
        emotion = emotion_classifier(analysis_text)[0]
        print(f"   🎭 Performance Emotion: {emotion['label']} ({emotion['score']:.3f})")

        # Performance insights
        if runs >= 700:
            insight = f"{batsman} had an outstanding season with {runs} runs!"
        elif runs >= 500:
            insight = f"{batsman} was consistent with {runs} runs throughout the season."
        else:
            insight = f"{batsman} contributed {runs} runs to the team."

        print(f"   💡 Insight: {insight}")

def main():
    print("🏏 IPL 2024 Top Run Scorers Analysis")
    print("=" * 50)

    # Get data from Snowflake
    top_scorers = get_top_run_scorers()

    print(f"\n📈 Top {len(top_scorers)} Run Scorers in IPL 2024:")
    print("-" * 40)

    for i, (batsman, runs) in enumerate(top_scorers, 1):
        print("2d")

    # Analyze with local LLM
    analyze_with_llm(top_scorers)

    print("\n✅ Analysis Complete!")
    print("💡 Your venv successfully combines Snowflake data with local AI models!")

if __name__ == "__main__":
    main()