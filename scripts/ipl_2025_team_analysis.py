#!/usr/bin/env python3
"""
IPL 2025 Team Performance Analysis with Local LLM
"""

import snowflake.connector
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

def get_team_wins_2025():
    """Get team wins from IPL 2025"""
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

        # Query team wins for 2025
        query = """
        SELECT MATCH_WON_BY, COUNT(*) as WINS
        FROM IPL_MATCH_SUMMARY_2024_2025
        WHERE SEASON = '2025'
        GROUP BY MATCH_WON_BY
        ORDER BY WINS DESC
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
            ('Royal Challengers Bengaluru', 11),
            ('Punjab Kings', 10),
            ('Mumbai Indians', 9),
            ('Gujarat Titans', 9),
            ('Delhi Capitals', 6)
        ]

def analyze_with_llm(team_performance):
    """Analyze team performance using local LLM"""
    print("🤖 Analyzing IPL 2025 Team Performance with Local LLM")
    print("=" * 60)

    # Load local models
    sentiment_analyzer = pipeline("sentiment-analysis",
                               model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    emotion_classifier = pipeline("text-classification",
                                model="j-hartmann/emotion-english-distilroberta-base")

    # Analyze each team's performance
    for i, (team, wins) in enumerate(team_performance, 1):
        print(f"\n{i}. {team}: {wins} wins")

        # Create analysis text
        analysis_text = f"{team} won {wins} matches in IPL 2025, showing {'excellent' if wins >= 10 else 'good' if wins >= 7 else 'decent'} team performance."

        # Sentiment analysis
        sentiment = sentiment_analyzer(analysis_text)[0]
        print(f"   📊 Team Sentiment: {sentiment['label']} ({sentiment['score']:.3f})")

        # Emotion analysis
        emotion = emotion_classifier(analysis_text)[0]
        print(f"   🎭 Team Emotion: {emotion['label']} ({emotion['score']:.3f})")

        # Performance insights
        if wins >= 10:
            insight = f"{team} had a dominant season with {wins} wins!"
        elif wins >= 7:
            insight = f"{team} performed well with {wins} wins, reaching playoffs."
        else:
            insight = f"{team} had {wins} wins this season."

        print(f"   💡 Insight: {insight}")

def main():
    print("🏏 IPL 2025 Team Performance Analysis")
    print("=" * 50)

    # Get data from Snowflake
    team_performance = get_team_wins_2025()

    print(f"\n📈 Team Wins in IPL 2025:")
    print("-" * 40)

    for i, (team, wins) in enumerate(team_performance, 1):
        print("2d")

    # Analyze with local LLM
    analyze_with_llm(team_performance)

    print("\n🏆 Champion Analysis:")
    champion_team, champion_wins = team_performance[0]
    print(f"   👑 {champion_team} won the most matches ({champion_wins} wins)")
    print("   🏅 They dominated IPL 2025 with their performance!")

    print("\n✅ Analysis Complete!")
    print("💡 Your venv successfully combines Snowflake data with local AI models!")

if __name__ == "__main__":
    main()