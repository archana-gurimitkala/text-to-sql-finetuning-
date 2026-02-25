"""
evaluate.py - Evaluate the fine-tuned model like Ed Donner's Week 6

Metrics:
- Exact Match Accuracy
- Token-level Accuracy
- Error Analysis
- Visualizations
"""

import json
import re
import matplotlib.pyplot as plt
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

def load_model_id():
    with open("model_id.txt", "r") as f:
        return f.read().strip()

def load_test_data():
    examples = []
    with open("data/test.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            examples.append({
                "question": data["messages"][1]["content"],
                "expected_sql": data["messages"][2]["content"]
            })
    return examples

# ============================================================
# PREDICTION
# ============================================================

SYSTEM_PROMPT = """You are a SQL expert. Convert natural language questions to SQL queries.

Rules:
- Return ONLY the SQL query, nothing else
- Do not explain the query
- Use proper SQL syntax
"""

def predict_sql(model_id, question):
    """Get SQL prediction from model."""
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        temperature=0,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# ============================================================
# METRICS (Like Ed Donner's Week 6)
# ============================================================

def normalize_sql(sql):
    """Normalize SQL for comparison."""
    sql = sql.lower().strip()
    sql = re.sub(r'\s+', ' ', sql)  # Multiple spaces to single
    sql = sql.replace('"', "'")     # Normalize quotes
    sql = sql.replace(" ,", ",")    # Remove space before comma
    sql = sql.replace(", ", ",")    # Remove space after comma
    return sql

def exact_match(predicted, expected):
    """Check if SQL matches exactly (after normalization)."""
    return normalize_sql(predicted) == normalize_sql(expected)

def token_accuracy(predicted, expected):
    """Calculate token-level accuracy."""
    pred_tokens = set(normalize_sql(predicted).split())
    exp_tokens = set(normalize_sql(expected).split())

    if not exp_tokens:
        return 0.0

    intersection = pred_tokens & exp_tokens
    return len(intersection) / len(exp_tokens)

def calculate_error(predicted, expected):
    """Calculate 'error' as 1 - token_accuracy (like MSE concept)."""
    return 1 - token_accuracy(predicted, expected)

# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model_id, test_data, verbose=True):
    """Run full evaluation on test data."""

    results = []
    exact_matches = 0
    total_token_accuracy = 0
    total_error = 0

    print("=" * 60)
    print("EVALUATING MODEL")
    print(f"Model: {model_id}")
    print(f"Test examples: {len(test_data)}")
    print("=" * 60)

    for i, example in enumerate(test_data):
        # Get prediction
        predicted = predict_sql(model_id, example["question"])
        expected = example["expected_sql"]

        # Calculate metrics
        is_exact = exact_match(predicted, expected)
        tok_acc = token_accuracy(predicted, expected)
        error = calculate_error(predicted, expected)

        # Accumulate
        if is_exact:
            exact_matches += 1
        total_token_accuracy += tok_acc
        total_error += error

        # Store result
        results.append({
            "question": example["question"][:100],
            "expected": expected,
            "predicted": predicted,
            "exact_match": is_exact,
            "token_accuracy": tok_acc,
            "error": error
        })

        # Print progress
        if verbose and (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_data)}...")

    # Calculate final metrics
    n = len(test_data)
    metrics = {
        "exact_match_accuracy": exact_matches / n * 100,
        "mean_token_accuracy": total_token_accuracy / n * 100,
        "mean_error": total_error / n,
        "rmse": (total_error / n) ** 0.5,  # Root Mean Square Error
        "total_examples": n,
        "exact_matches": exact_matches
    }

    return results, metrics

# ============================================================
# ERROR ANALYSIS
# ============================================================

def analyze_errors(results):
    """Analyze where the model fails."""

    errors = [r for r in results if not r["exact_match"]]

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print(f"Total errors: {len(errors)} / {len(results)}")
    print("=" * 60)

    # Categorize errors
    error_types = Counter()

    for r in errors:
        exp = r["expected"].upper()
        pred = r["predicted"].upper()

        if "JOIN" in exp and "JOIN" not in pred:
            error_types["Missing JOIN"] += 1
        elif "GROUP BY" in exp and "GROUP BY" not in pred:
            error_types["Missing GROUP BY"] += 1
        elif "ORDER BY" in exp and "ORDER BY" not in pred:
            error_types["Missing ORDER BY"] += 1
        elif "WHERE" in exp and "WHERE" not in pred:
            error_types["Missing WHERE"] += 1
        elif "COUNT" in exp or "SUM" in exp or "AVG" in exp:
            error_types["Aggregate function error"] += 1
        else:
            error_types["Other"] += 1

    print("\n--- Error Categories ---")
    for error_type, count in error_types.most_common():
        print(f"  {error_type}: {count}")

    # Show sample errors
    print("\n--- Sample Errors ---")
    for i, r in enumerate(errors[:5]):
        print(f"\n[Error {i + 1}]")
        print(f"Question: {r['question'][:80]}...")
        print(f"Expected: {r['expected'][:80]}")
        print(f"Got:      {r['predicted'][:80]}")
        print(f"Token Acc: {r['token_accuracy']:.1%}")

    return error_types

# ============================================================
# VISUALIZATION (Like Ed Donner's Charts)
# ============================================================

def create_visualizations(results, metrics):
    """Create evaluation charts like Ed Donner's Week 6."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Text-to-SQL Model Evaluation", fontsize=14, fontweight='bold')

    # 1. Accuracy Bar Chart
    ax1 = axes[0, 0]
    metrics_names = ['Exact Match\nAccuracy', 'Token\nAccuracy']
    metrics_values = [metrics['exact_match_accuracy'], metrics['mean_token_accuracy']]
    colors = ['#2ecc71' if v > 70 else '#e74c3c' for v in metrics_values]
    bars = ax1.bar(metrics_names, metrics_values, color=colors)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Metrics')
    ax1.set_ylim(0, 100)
    for bar, val in zip(bars, metrics_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val:.1f}%', ha='center', fontweight='bold')

    # 2. Error Distribution
    ax2 = axes[0, 1]
    errors = [r['error'] for r in results]
    ax2.hist(errors, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
    ax2.axvline(metrics['mean_error'], color='red', linestyle='--',
                label=f"Mean Error: {metrics['mean_error']:.3f}")
    ax2.set_xlabel('Error (1 - Token Accuracy)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distribution')
    ax2.legend()

    # 3. Token Accuracy Distribution
    ax3 = axes[1, 0]
    token_accs = [r['token_accuracy'] * 100 for r in results]
    ax3.hist(token_accs, bins=20, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax3.axvline(metrics['mean_token_accuracy'], color='red', linestyle='--',
                label=f"Mean: {metrics['mean_token_accuracy']:.1f}%")
    ax3.set_xlabel('Token Accuracy (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Token Accuracy Distribution')
    ax3.legend()

    # 4. Exact Match Pie Chart
    ax4 = axes[1, 1]
    exact = metrics['exact_matches']
    not_exact = metrics['total_examples'] - exact
    ax4.pie([exact, not_exact],
            labels=[f'Exact Match\n({exact})', f'Not Exact\n({not_exact})'],
            colors=['#2ecc71', '#e74c3c'],
            autopct='%1.1f%%',
            startangle=90)
    ax4.set_title('Exact Match Results')

    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Chart saved to: evaluation_results.png")
    plt.show()

# ============================================================
# PRINT SUMMARY (Like Ed Donner's Tester)
# ============================================================

def print_summary(metrics):
    """Print summary like Ed Donner's evaluation."""

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"""
┌─────────────────────────────────────────────────────────┐
│  METRICS                                                │
├─────────────────────────────────────────────────────────┤
│  Exact Match Accuracy:  {metrics['exact_match_accuracy']:>6.2f}%                      │
│  Mean Token Accuracy:   {metrics['mean_token_accuracy']:>6.2f}%                      │
│  Mean Error:            {metrics['mean_error']:>6.4f}                       │
│  RMSE:                  {metrics['rmse']:>6.4f}                       │
├─────────────────────────────────────────────────────────┤
│  Total Examples:        {metrics['total_examples']:>6}                        │
│  Exact Matches:         {metrics['exact_matches']:>6}                        │
└─────────────────────────────────────────────────────────┘
""")

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("TEXT-TO-SQL MODEL EVALUATION")
    print("Like Ed Donner's Week 6!")
    print("=" * 60)

    # Load
    model_id = load_model_id()
    test_data = load_test_data()

    print(f"\nModel: {model_id}")
    print(f"Test data: {len(test_data)} examples")

    # Evaluate
    print("\n🔄 Running evaluation...")
    results, metrics = evaluate_model(model_id, test_data)

    # Summary
    print_summary(metrics)

    # Error analysis
    analyze_errors(results)

    # Visualizations
    print("\n📊 Creating visualizations...")
    create_visualizations(results, metrics)

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "results": results
        }, f, indent=2)
    print("\n💾 Results saved to: evaluation_results.json")


if __name__ == "__main__":
    main()
