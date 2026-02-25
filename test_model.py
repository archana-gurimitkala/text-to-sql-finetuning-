"""
test_model.py - Test the fine-tuned model

This script:
1. Loads the fine-tuned model ID
2. Tests it on sample questions
3. Compares with expected SQL
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from schema import SCHEMA

# Load API key
load_dotenv()
client = OpenAI()


def load_model_id():
    """Load the fine-tuned model ID from file."""
    try:
        with open("model_id.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("❌ model_id.txt not found!")
        print("Run fine_tune.py first.")
        return None


def query_model(model_id, question):
    """Send a question to the fine-tuned model."""

    system_prompt = f"""You are a SQL expert. Convert natural language questions to SQL queries.

{SCHEMA}

Rules:
- Return ONLY the SQL query, nothing else
- Do not explain the query
- Use proper SQL syntax
"""

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0,  # Deterministic output
        max_tokens=200
    )

    return response.choices[0].message.content.strip()


def test_on_examples():
    """Test the model on sample questions."""

    model_id = load_model_id()
    if not model_id:
        return

    print("=" * 60)
    print(f"Testing Fine-tuned Model: {model_id}")
    print("=" * 60)

    # Test questions
    test_questions = [
        "Show all customers",
        "Find products cheaper than 50 dollars",
        "Count orders by status",
        "Show top 5 most expensive products",
        "Find customers from New York",
        "What is the average product price?",
        "List products in Electronics category",
        "Show orders with customer names",
        "Find products with stock below 10",
        "Count customers by city"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n[Test {i}]")
        print(f"Question: {question}")

        sql = query_model(model_id, question)
        print(f"SQL:      {sql}")


def test_on_test_file():
    """Test on the held-out test set and calculate accuracy."""

    model_id = load_model_id()
    if not model_id:
        return

    print("=" * 60)
    print("Testing on test.jsonl")
    print("=" * 60)

    correct = 0
    total = 0

    with open("data/test.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            question = data["messages"][1]["content"]
            expected_sql = data["messages"][2]["content"]

            predicted_sql = query_model(model_id, question)

            # Simple exact match (case-insensitive)
            is_correct = predicted_sql.strip().lower() == expected_sql.strip().lower()

            total += 1
            if is_correct:
                correct += 1
            else:
                print(f"\n❌ Mismatch:")
                print(f"   Question: {question}")
                print(f"   Expected: {expected_sql}")
                print(f"   Got:      {predicted_sql}")

    accuracy = (correct / total) * 100
    print(f"\n{'=' * 60}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print("=" * 60)


def interactive_mode():
    """Interactive testing - ask your own questions."""

    model_id = load_model_id()
    if not model_id:
        return

    print("=" * 60)
    print("Interactive Mode - Ask your own questions!")
    print("Type 'quit' to exit")
    print("=" * 60)

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        sql = query_model(model_id, question)
        print(f"SQL: {sql}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_on_test_file()
        elif sys.argv[1] == "--interactive":
            interactive_mode()
        else:
            print("Usage:")
            print("  python test_model.py           # Test on sample questions")
            print("  python test_model.py --test    # Test on test.jsonl")
            print("  python test_model.py --interactive  # Ask your own questions")
    else:
        test_on_examples()
