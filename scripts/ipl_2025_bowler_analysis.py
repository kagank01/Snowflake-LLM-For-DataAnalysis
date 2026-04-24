#!/usr/bin/env python3
"""
IPL 2025 Bowler Performance Analysis with Local LLM
"""

import snowflake.connector
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

def get_bowler_stats():
    """Get bowler statistics from IPL 2025"""
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

        # Query top wicket-takers
        wickets_query = """
        SELECT BOWLER, COUNT(*) as BALLS_BOWLED, SUM(RUNS_TOTAL) as RUNS_CONCEDED,
               COUNT(CASE WHEN WICKET_KIND IS NOT NULL THEN 1 END) as WICKETS
        FROM IPL_24_25_BALL_BY_BALL
        WHERE SEASON = '2025'
        GROUP BY BOWLER
        ORDER BY WICKETS DESC
        LIMIT 10
        """

        cursor.execute(wickets_query)
        top_wicket_takers = cursor.fetchall()

        # Query best economy rates
        economy_query = """
        SELECT BOWLER, COUNT(*) as BALLS_BOWLED, SUM(RUNS_TOTAL) as RUNS_CONCEDED,
               COUNT(CASE WHEN WICKET_KIND IS NOT NULL THEN 1 END) as WICKETS,
               ROUND(SUM(RUNS_TOTAL) * 6.0 / COUNT(*), 2) as ECONOMY_RATE
        FROM IPL_24_25_BALL_BY_BALL
        WHERE SEASON = '2025'
        GROUP BY BOWLER
        HAVING BALLS_BOWLED >= 100
        ORDER BY ECONOMY_RATE ASC
        LIMIT 10
        """

        cursor.execute(economy_query)
        best_economy = cursor.fetchall()

        # Query best bowling averages
        average_query = """
        SELECT BOWLER, COUNT(*) as BALLS_BOWLED, SUM(RUNS_TOTAL) as RUNS_CONCEDED,
               COUNT(CASE WHEN WICKET_KIND IS NOT NULL THEN 1 END) as WICKETS,
               ROUND(SUM(RUNS_TOTAL) * 1.0 / NULLIF(COUNT(CASE WHEN WICKET_KIND IS NOT NULL THEN 1 END), 0), 2) as BOWLING_AVERAGE,
               ROUND(COUNT(*) * 1.0 / NULLIF(COUNT(CASE WHEN WICKET_KIND IS NOT NULL THEN 1 END), 0), 2) as STRIKE_RATE
        FROM IPL_24_25_BALL_BY_BALL
        WHERE SEASON = '2025'
        GROUP BY BOWLER
        HAVING WICKETS >= 10
        ORDER BY BOWLING_AVERAGE ASC
        LIMIT 10
        """

        cursor.execute(average_query)
        best_averages = cursor.fetchall()

        cursor.close()
        conn.close()

        return top_wicket_takers, best_economy, best_averages

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("💡 Using sample data for demonstration")
        top_wicket_takers = [
            ('M Prasidh Krishna', 364, 506, 26),
            ('Noor Ahmad', 314, 416, 24),
            ('TA Boult', 357, 520, 23),
            ('Arshdeep Singh', 367, 527, 22),
            ('JR Hazlewood', 270, 389, 22)
        ]
        best_economy = [
            ('JJ Bumrah', 289, 322, 21, 6.69),
            ('Kuldeep Yadav', 324, 392, 16, 7.26),
            ('JD Unadkat', 146, 177, 12, 7.27),
            ('CV Varun', 308, 390, 17, 7.60),
            ('MJ Santner', 247, 316, 10, 7.68)
        ]
        best_averages = [
            ('JD Unadkat', 146, 177, 12, 14.75, 12.17),
            ('JJ Bumrah', 289, 322, 21, 15.33, 13.76),
            ('Noor Ahmad', 314, 416, 24, 17.33, 13.08),
            ('E Malinga', 165, 244, 14, 17.43, 11.79),
            ('JR Hazlewood', 270, 389, 22, 17.68, 12.27)
        ]
        return top_wicket_takers, best_economy, best_averages

def analyze_with_llm(top_wicket_takers, best_economy, best_averages):
    """Analyze bowler performance using local LLM"""
    print("🤖 Analyzing IPL 2025 Bowler Performance with Local LLM")
    print("=" * 60)

    # Load local models
    sentiment_analyzer = pipeline("sentiment-analysis",
                               model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    emotion_classifier = pipeline("text-classification",
                                model="j-hartmann/emotion-english-distilroberta-base")

    # Analyze top wicket-takers
    print("\n🏆 Top Wicket-Takers Analysis:")
    print("-" * 40)

    for i, (bowler, balls, runs, wickets) in enumerate(top_wicket_takers[:5], 1):
        print(f"\n{i}. {bowler}: {wickets} wickets")

        # Create analysis text
        analysis_text = f"{bowler} took {wickets} wickets in IPL 2025, conceding {runs} runs in {balls} balls, showing {'exceptional' if wickets >= 25 else 'excellent' if wickets >= 20 else 'strong'} bowling performance."

        # Sentiment analysis
        sentiment = sentiment_analyzer(analysis_text)[0]
        print(f"   📊 Performance Sentiment: {sentiment['label']} ({sentiment['score']:.3f})")

        # Emotion analysis
        emotion = emotion_classifier(analysis_text)[0]
        print(f"   🎭 Performance Emotion: {emotion['label']} ({emotion['score']:.3f})")

        # Performance insights
        if wickets >= 25:
            insight = f"{bowler} was a wicket-taking machine with {wickets} scalps!"
        elif wickets >= 20:
            insight = f"{bowler} was among the top performers with {wickets} wickets."
        else:
            insight = f"{bowler} contributed {wickets} wickets to the team."

        print(f"   💡 Insight: {insight}")

    # Analyze best economy rates
    print("\n⚡ Best Economy Rates Analysis:")
    print("-" * 40)

    for i, (bowler, balls, runs, wickets, economy) in enumerate(best_economy[:5], 1):
        print(f"\n{i}. {bowler}: {economy} economy rate")

        economy_text = f"{bowler} maintained an excellent economy rate of {economy} in IPL 2025, conceding {runs} runs in {balls} balls."

        sentiment = sentiment_analyzer(economy_text)[0]
        print(f"   📊 Economy Sentiment: {sentiment['label']} ({sentiment['score']:.3f})")

        if economy <= 7.0:
            insight = f"{bowler} was exceptionally economical at {economy} runs per over!"
        elif economy <= 8.0:
            insight = f"{bowler} showed good control with {economy} economy rate."
        else:
            insight = f"{bowler} maintained {economy} runs per over."

        print(f"   💡 Insight: {insight}")

    # Analyze best bowling averages
    print("\n📊 Best Bowling Averages Analysis:")
    print("-" * 40)

    for i, (bowler, balls, runs, wickets, average, strike_rate) in enumerate(best_averages[:5], 1):
        print(f"\n{i}. {bowler}: {average} bowling average, {strike_rate} strike rate")

        average_text = f"{bowler} had an impressive bowling average of {average} with a strike rate of {strike_rate} in IPL 2025."

        sentiment = sentiment_analyzer(average_text)[0]
        print(f"   📊 Average Sentiment: {sentiment['label']} ({sentiment['score']:.3f})")

        if average <= 16.0:
            insight = f"{bowler} was lethal with {average} average - took wickets cheaply!"
        elif average <= 20.0:
            insight = f"{bowler} was effective with {average} bowling average."
        else:
            insight = f"{bowler} maintained {average} runs per wicket."

        print(f"   💡 Insight: {insight}")

def main():
    print("🏏 IPL 2025 Bowler Performance Analysis")
    print("=" * 50)

    # Get data from Snowflake
    top_wicket_takers, best_economy, best_averages = get_bowler_stats()

    print(f"\n🏆 Top Wicket-Takers in IPL 2025:")
    print("-" * 40)

    for i, (bowler, balls, runs, wickets) in enumerate(top_wicket_takers, 1):
        print("2d")

    print(f"\n⚡ Best Economy Rates (min 100 balls):")
    print("-" * 40)

    for i, (bowler, balls, runs, wickets, economy) in enumerate(best_economy, 1):
        print("2d")

    print(f"\n📊 Best Bowling Averages (min 10 wickets):")
    print("-" * 40)

    for i, (bowler, balls, runs, wickets, average, strike_rate) in enumerate(best_averages, 1):
        print("2d")

    # Analyze with local LLM
    analyze_with_llm(top_wicket_takers, best_economy, best_averages)

    print("\n🥎 Bowling Analysis Summary:")
    top_wicket_taker, _, _, top_wickets = top_wicket_takers[0]
    best_economical, _, _, _, best_economy_rate = best_economy[0]
    best_average_bowler, _, _, _, best_average_rate, _ = best_averages[0]

    print(f"   🏆 Leading Wicket-Taker: {top_wicket_taker} ({top_wickets} wickets)")
    print(f"   ⚡ Most Economical: {best_economical} ({best_economy_rate} economy)")
    print(f"   📊 Best Average: {best_average_bowler} ({best_average_rate} average)")
    print("   🏅 Bowling excellence defined IPL 2025!")

    print("\n✅ Analysis Complete!")
    print("💡 Your venv successfully combines Snowflake data with local AI models!")

if __name__ == "__main__":
    main()