"""
curate_data.py - Download and curate real Text-to-SQL data

This script:
1. Downloads a real dataset from Hugging Face
2. Explores the data
3. Cleans and filters
4. Balances across query types
5. Saves curated train/test files

Similar to Ed Donner's Week 6 curation process!
"""

import json
import random
from collections import Counter, defaultdict
from datasets import load_dataset

# ============================================================
# STEP 1: DOWNLOAD DATASET
# ============================================================

def download_dataset():
    """Download Text-to-SQL dataset from Hugging Face."""

    print("=" * 60)
    print("Step 1: Downloading dataset from Hugging Face...")
    print("=" * 60)

    # Using Clinton/Text-to-sql-v1 - a good Text-to-SQL dataset
    dataset = load_dataset("Clinton/Text-to-sql-v1", split="train")

    print(f"Downloaded {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")

    return dataset


# ============================================================
# STEP 2: EXPLORE DATA
# ============================================================

def explore_data(dataset):
    """Explore the dataset to understand its structure."""

    print("\n" + "=" * 60)
    print("Step 2: Exploring the data...")
    print("=" * 60)

    # Show sample examples
    print("\n--- Sample Examples ---")
    for i in range(3):
        example = dataset[i]
        print(f"\n[Example {i+1}]")
        for key, value in example.items():
            # Truncate long values
            val_str = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
            print(f"  {key}: {val_str}")

    # Analyze SQL query types
    print("\n--- SQL Query Type Distribution ---")
    query_types = []
    for example in dataset:
        sql = example.get("response", "").upper()
        if sql.startswith("SELECT"):
            if "JOIN" in sql:
                query_types.append("SELECT with JOIN")
            elif "GROUP BY" in sql:
                query_types.append("SELECT with GROUP BY")
            elif "ORDER BY" in sql:
                query_types.append("SELECT with ORDER BY")
            elif "WHERE" in sql:
                query_types.append("SELECT with WHERE")
            else:
                query_types.append("Simple SELECT")
        elif sql.startswith("INSERT"):
            query_types.append("INSERT")
        elif sql.startswith("UPDATE"):
            query_types.append("UPDATE")
        elif sql.startswith("DELETE"):
            query_types.append("DELETE")
        else:
            query_types.append("Other")

    type_counts = Counter(query_types)
    for qtype, count in type_counts.most_common(10):
        print(f"  {qtype}: {count}")

    # Analyze question lengths
    print("\n--- Question Length Distribution ---")
    lengths = [len(ex.get("instruction", "").split()) for ex in dataset]
    print(f"  Min: {min(lengths)} words")
    print(f"  Max: {max(lengths)} words")
    print(f"  Average: {sum(lengths)/len(lengths):.1f} words")

    return type_counts


# ============================================================
# STEP 3: CLEAN DATA
# ============================================================

def clean_data(dataset):
    """Remove bad or unusable examples."""

    print("\n" + "=" * 60)
    print("Step 3: Cleaning data...")
    print("=" * 60)

    cleaned = []
    removed_reasons = Counter()

    for example in dataset:
        # Get question and SQL (handle different column names)
        # This dataset has: instruction, input, response
        question = example.get("instruction", "")  # The natural language question
        sql = example.get("response", "")          # The SQL answer
        context = example.get("input", "")         # The schema/context

        # --- CLEANING RULES ---

        # Rule 1: Must have both question and SQL
        if not question or not sql:
            removed_reasons["Missing question or SQL"] += 1
            continue

        # Rule 2: Question must be reasonable length (3-50 words)
        word_count = len(question.split())
        if word_count < 3:
            removed_reasons["Question too short"] += 1
            continue
        if word_count > 50:
            removed_reasons["Question too long"] += 1
            continue

        # Rule 3: SQL must start with valid keyword
        sql_upper = sql.strip().upper()
        if not any(sql_upper.startswith(kw) for kw in ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH"]):
            removed_reasons["Invalid SQL start"] += 1
            continue

        # Rule 4: SQL must not be too long (keep it learnable)
        if len(sql) > 500:
            removed_reasons["SQL too long"] += 1
            continue

        # Rule 5: Remove examples with problematic characters
        if "\\n" in sql or "\t" in sql:
            removed_reasons["SQL has escape characters"] += 1
            continue

        # Passed all checks!
        cleaned.append({
            "question": question.strip(),
            "sql": sql.strip(),
            "context": context.strip() if context else ""
        })

    print(f"\nOriginal: {len(dataset)} examples")
    print(f"After cleaning: {len(cleaned)} examples")
    print(f"Removed: {len(dataset) - len(cleaned)} examples")

    print("\n--- Removal Reasons ---")
    for reason, count in removed_reasons.most_common():
        print(f"  {reason}: {count}")

    return cleaned


# ============================================================
# STEP 4: FILTER DATA
# ============================================================

def filter_data(cleaned_data):
    """Filter to keep only high-quality examples."""

    print("\n" + "=" * 60)
    print("Step 4: Filtering data...")
    print("=" * 60)

    filtered = []
    removed_reasons = Counter()

    for example in cleaned_data:
        sql = example["sql"].upper()

        # --- FILTER RULES ---

        # Rule 1: Focus on SELECT queries (most common use case)
        if not sql.startswith("SELECT"):
            removed_reasons["Not a SELECT query"] += 1
            continue

        # Rule 2: Remove queries with subqueries (too complex for learning)
        if sql.count("SELECT") > 2:
            removed_reasons["Too many subqueries"] += 1
            continue

        # Rule 3: Remove queries with UNION (complex)
        if "UNION" in sql:
            removed_reasons["Has UNION"] += 1
            continue

        # Passed all filters!
        filtered.append(example)

    print(f"\nAfter cleaning: {len(cleaned_data)} examples")
    print(f"After filtering: {len(filtered)} examples")
    print(f"Removed: {len(cleaned_data) - len(filtered)} examples")

    print("\n--- Filter Reasons ---")
    for reason, count in removed_reasons.most_common():
        print(f"  {reason}: {count}")

    return filtered


# ============================================================
# STEP 5: BALANCE DATA
# ============================================================

def categorize_query(sql):
    """Categorize a SQL query by its type."""
    sql_upper = sql.upper()

    if "JOIN" in sql_upper:
        return "JOIN"
    elif "GROUP BY" in sql_upper:
        return "GROUP BY"
    elif "ORDER BY" in sql_upper and "LIMIT" in sql_upper:
        return "ORDER BY + LIMIT"
    elif "ORDER BY" in sql_upper:
        return "ORDER BY"
    elif "HAVING" in sql_upper:
        return "HAVING"
    elif "COUNT(" in sql_upper or "SUM(" in sql_upper or "AVG(" in sql_upper:
        return "AGGREGATE"
    elif "WHERE" in sql_upper and ("AND" in sql_upper or "OR" in sql_upper):
        return "WHERE (complex)"
    elif "WHERE" in sql_upper:
        return "WHERE (simple)"
    elif "LIKE" in sql_upper:
        return "LIKE"
    else:
        return "Simple SELECT"


def balance_data(filtered_data, max_per_category=50, total_limit=300):
    """Balance data across query types."""

    print("\n" + "=" * 60)
    print("Step 5: Balancing data...")
    print("=" * 60)

    # Group by category
    by_category = defaultdict(list)
    for example in filtered_data:
        category = categorize_query(example["sql"])
        by_category[category].append(example)

    print("\n--- Before Balancing ---")
    for cat, examples in sorted(by_category.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(examples)}")

    # Sample from each category
    balanced = []
    for category, examples in by_category.items():
        # Take up to max_per_category from each
        sample_size = min(len(examples), max_per_category)
        sampled = random.sample(examples, sample_size)
        balanced.extend(sampled)

    # Shuffle
    random.shuffle(balanced)

    # Limit total
    if len(balanced) > total_limit:
        balanced = balanced[:total_limit]

    print(f"\n--- After Balancing ---")
    final_counts = Counter(categorize_query(ex["sql"]) for ex in balanced)
    for cat, count in final_counts.most_common():
        print(f"  {cat}: {count}")

    print(f"\nFinal dataset size: {len(balanced)}")

    return balanced


# ============================================================
# STEP 6: SAVE CURATED DATA
# ============================================================

def save_curated_data(balanced_data, train_ratio=0.85):
    """Save as OpenAI fine-tuning format."""

    print("\n" + "=" * 60)
    print("Step 6: Saving curated data...")
    print("=" * 60)

    # Split into train/test
    random.shuffle(balanced_data)
    split_idx = int(len(balanced_data) * train_ratio)
    train_data = balanced_data[:split_idx]
    test_data = balanced_data[split_idx:]

    print(f"Training examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")

    # System prompt
    system_prompt = """You are a SQL expert. Convert natural language questions to SQL queries.

Rules:
- Return ONLY the SQL query, nothing else
- Do not explain the query
- Use proper SQL syntax
- Use the table and column names from the question context
"""

    # Save training data
    with open("data/train.jsonl", "w") as f:
        for ex in train_data:
            # Include context (schema) in the user message if available
            user_content = ex["question"]
            if ex.get("context"):
                user_content = f"Context: {ex['context']}\n\nQuestion: {ex['question']}"

            conversation = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": ex["sql"]}
                ]
            }
            f.write(json.dumps(conversation) + "\n")

    # Save test data
    with open("data/test.jsonl", "w") as f:
        for ex in test_data:
            user_content = ex["question"]
            if ex.get("context"):
                user_content = f"Context: {ex['context']}\n\nQuestion: {ex['question']}"

            conversation = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": ex["sql"]}
                ]
            }
            f.write(json.dumps(conversation) + "\n")

    # Also save raw data for inspection
    with open("data/curated_raw.json", "w") as f:
        json.dump(balanced_data, f, indent=2)

    print("\nFiles saved:")
    print("  data/train.jsonl (OpenAI format)")
    print("  data/test.jsonl (OpenAI format)")
    print("  data/curated_raw.json (raw data for inspection)")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("TEXT-TO-SQL DATA CURATION")
    print("Like Ed Donner's Week 6!")
    print("=" * 60)

    # Step 1: Download
    dataset = download_dataset()

    # Step 2: Explore
    explore_data(dataset)

    # Step 3: Clean
    cleaned = clean_data(dataset)

    # Step 4: Filter
    filtered = filter_data(cleaned)

    # Step 5: Balance
    balanced = balance_data(filtered, max_per_category=50, total_limit=300)

    # Step 6: Save
    save_curated_data(balanced)

    print("\n" + "=" * 60)
    print("CURATION COMPLETE!")
    print("=" * 60)

    # Show sample
    print("\n--- Sample Curated Examples ---")
    for i, ex in enumerate(balanced[:3]):
        print(f"\n[Example {i+1}]")
        print(f"Question: {ex['question'][:100]}...")
        print(f"SQL: {ex['sql'][:100]}...")


if __name__ == "__main__":
    main()
