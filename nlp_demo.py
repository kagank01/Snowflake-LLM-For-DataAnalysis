#!/usr/bin/env python3
"""
Local LLM and Natural Language Processing Demo
Using transformers library with local models in your Snowflake venv
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

def main():
    print("🚀 Local LLM & NLP Demo in Virtual Environment")
    print("=" * 60)

    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    print("\n📝 Loading local NLP models...")

    try:
        # Sentiment Analysis with local model
        print("\n1. Sentiment Analysis (Local Model):")
        sentiment_analyzer = pipeline("sentiment-analysis",
                                   model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        texts = [
            "I love working with Snowflake and Python!",
            "This cricket analysis is amazing!",
            "The IPL final was absolutely thrilling!",
            "Data engineering can be challenging sometimes."
        ]

        for text in texts:
            result = sentiment_analyzer(text)
            sentiment = result[0]['label']
            confidence = result[0]['score']
            print(f"  '{text[:50]}...' → {sentiment} ({confidence:.3f})")

        # Emotion Detection
        print("\n2. Emotion Detection (Local Model):")
        emotion_classifier = pipeline("text-classification",
                                    model="j-hartmann/emotion-english-distilroberta-base")
        emotions = [
            "I'm so excited about the IPL cricket season!",
            "That was a disappointing loss for the team.",
            "The crowd's energy was incredible!",
            "I feel nervous about the upcoming match."
        ]

        for text in emotions:
            result = emotion_classifier(text)
            emotion = result[0]['label']
            confidence = result[0]['score']
            print(f"  '{text[:40]}...' → {emotion} ({confidence:.3f})")

        # Named Entity Recognition
        print("\n3. Named Entity Recognition (Local Model):")
        ner_pipeline = pipeline("ner",
                              model="dbmdz/bert-large-cased-finetuned-conll03-english",
                              aggregation_strategy="simple")
        cricket_text = "Virat Kohli scored 82 runs for Royal Challengers Bangalore in IPL 2024 against Gujarat Titans."
        entities = ner_pipeline(cricket_text)

        print(f"Text: {cricket_text}")
        for entity in entities:
            print(f"  {entity['word']} → {entity['entity_group']} ({entity['score']:.2f})")

        # Question Answering with IPL context
        print("\n4. Question Answering (Local Model):")
        questions = [
            "Who won IPL 2024?",
            "Who did CSK defeat in the final?",
            "Who was the captain of CSK?"
        ]

        try:
            qa_pipeline = pipeline("question-answering",
                                 model="deepset/roberta-base-squad2")
            context = """IPL 2024 was won by Chennai Super Kings. They defeated Gujarat Titans in the final
            match held at DY Patil Stadium in Navi Mumbai. MS Dhoni was the captain of CSK."""

            for question in questions:
                result = qa_pipeline(question=question, context=context)
                print(f"  Q: {question}")
                print(f"  A: {result['answer']} (confidence: {result['score']:.3f})")
        except Exception as e:
            print(f"❌ Question Answering failed: {str(e)}")
            print("💡 Alternative: Using text generation for Q&A simulation")
            # Fallback: Use text generation to simulate Q&A
            qa_simulator = pipeline("text-generation", model="distilgpt2")
            for question in questions[:2]:  # Just show 2 examples
                prompt = f"Question: {question} Answer:"
                result = qa_simulator(prompt, max_length=50, num_return_sequences=1, pad_token_id=50256)
                generated = result[0]['generated_text'].replace(prompt, "").strip()
                print(f"  Q: {question}")
                print(f"  A: {generated[:50]}... (simulated)")

        # Text Generation
        print("\n5. Text Generation (Local Model):")
        generator = pipeline("text-generation",
                           model="distilgpt2",
                           device=0 if torch.cuda.is_available() else -1)

        prompts = [
            "The IPL cricket tournament",
            "Chennai Super Kings are known for"
        ]

        for prompt in prompts:
            result = generator(prompt, max_length=30, num_return_sequences=1, pad_token_id=50256)
            generated = result[0]['generated_text']
            print(f"  Prompt: '{prompt}'")
            print(f"  Generated: '{generated}'")
            print()

        print("✅ All local NLP models completed successfully!")
        print("\n💡 Integration with Snowflake:")
        print("   • Analyze player sentiment from match comments")
        print("   • Extract player/team names from articles")
        print("   • Generate match summaries")
        print("   • Classify social media reactions")
        print("   • Answer questions about IPL statistics")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("If you see model download issues, check your internet connection.")

if __name__ == "__main__":
    main()