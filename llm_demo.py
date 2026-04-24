#!/usr/bin/env python3
"""
LLM and Natural Language Processing Demo
Using transformers library with your Snowflake venv
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

def main():
    print("🚀 LLM & NLP Demo in Virtual Environment")
    print("=" * 50)

    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    print("\n📝 Loading NLP models...")

    try:
        # Sentiment Analysis
        print("\n1. Sentiment Analysis:")
        sentiment_analyzer = pipeline("sentiment-analysis",
                                   model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        text = "I love working with Snowflake and Python! This is amazing."
        result = sentiment_analyzer(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result[0]['label']} (confidence: {result[0]['score']:.3f})")

        # Text Classification
        print("\n2. Text Classification (Emotion):")
        emotion_classifier = pipeline("text-classification",
                                    model="j-hartmann/emotion-english-distilroberta-base")
        text = "I'm so excited about the IPL cricket season!"
        result = emotion_classifier(text)
        print(f"Text: {text}")
        print(f"Emotion: {result[0]['label']} (confidence: {result[0]['score']:.3f})")

        # Question Answering
        print("\n3. Question Answering:")
        qa_pipeline = pipeline("question-answering",
                             model="deepset/roberta-base-squad2")
        context = "IPL 2024 was won by Chennai Super Kings. They defeated Gujarat Titans in the final."
        question = "Who won IPL 2024?"
        result = qa_pipeline(question=question, context=context)
        print(f"Question: {question}")
        print(f"Answer: {result['answer']} (confidence: {result['score']:.3f})")

        print("\n✅ All NLP tasks completed successfully!")
        print("\n💡 You can now integrate LLMs with your Snowflake data!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Note: If you see DLL errors, you may need to install Microsoft Visual C++ Redistributable")

if __name__ == "__main__":
    main()