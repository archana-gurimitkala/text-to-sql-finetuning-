"""
generate_data.py - Create training data for fine-tuning

This script generates pairs of:
- Natural language question
- Correct SQL query

We'll create hundreds of examples for the model to learn from.
"""

import json
import random
from schema import SCHEMA, CITIES, CATEGORIES, STATUSES

# ============================================================
# TRAINING EXAMPLES
# ============================================================

# Each template has:
# - "question": What user might ask (with {placeholders})
# - "sql": The correct SQL (with {placeholders})

TEMPLATES = [
    # ----- SIMPLE SELECT ALL -----
    {
        "question": "Show all customers",
        "sql": "SELECT * FROM customers"
    },
    {
        "question": "List all products",
        "sql": "SELECT * FROM products"
    },
    {
        "question": "Get all orders",
        "sql": "SELECT * FROM orders"
    },
    {
        "question": "Display all order items",
        "sql": "SELECT * FROM order_items"
    },

    # ----- SELECT SPECIFIC COLUMNS -----
    {
        "question": "Show customer names and emails",
        "sql": "SELECT name, email FROM customers"
    },
    {
        "question": "List product names and prices",
        "sql": "SELECT name, price FROM products"
    },
    {
        "question": "Get order dates and totals",
        "sql": "SELECT order_date, total FROM orders"
    },

    # ----- WHERE CLAUSE (CITY) -----
    {
        "question": "Find customers from {city}",
        "sql": "SELECT * FROM customers WHERE city = '{city}'"
    },
    {
        "question": "Show all customers in {city}",
        "sql": "SELECT * FROM customers WHERE city = '{city}'"
    },
    {
        "question": "List customers who live in {city}",
        "sql": "SELECT * FROM customers WHERE city = '{city}'"
    },

    # ----- WHERE CLAUSE (CATEGORY) -----
    {
        "question": "Find products in {category} category",
        "sql": "SELECT * FROM products WHERE category = '{category}'"
    },
    {
        "question": "Show all {category} products",
        "sql": "SELECT * FROM products WHERE category = '{category}'"
    },
    {
        "question": "List products from {category} category",
        "sql": "SELECT * FROM products WHERE category = '{category}'"
    },

    # ----- WHERE CLAUSE (STATUS) -----
    {
        "question": "Find orders with status {status}",
        "sql": "SELECT * FROM orders WHERE status = '{status}'"
    },
    {
        "question": "Show all {status} orders",
        "sql": "SELECT * FROM orders WHERE status = '{status}'"
    },
    {
        "question": "Get orders that are {status}",
        "sql": "SELECT * FROM orders WHERE status = '{status}'"
    },

    # ----- WHERE CLAUSE (NUMERIC COMPARISON) -----
    {
        "question": "Find products cheaper than {price} dollars",
        "sql": "SELECT * FROM products WHERE price < {price}"
    },
    {
        "question": "Show products with price above {price}",
        "sql": "SELECT * FROM products WHERE price > {price}"
    },
    {
        "question": "List products under ${price}",
        "sql": "SELECT * FROM products WHERE price < {price}"
    },
    {
        "question": "Find products costing more than {price}",
        "sql": "SELECT * FROM products WHERE price > {price}"
    },
    {
        "question": "Show orders with total greater than {price}",
        "sql": "SELECT * FROM orders WHERE total > {price}"
    },
    {
        "question": "Find orders under ${price}",
        "sql": "SELECT * FROM orders WHERE total < {price}"
    },
    {
        "question": "List products with stock below {quantity}",
        "sql": "SELECT * FROM products WHERE stock < {quantity}"
    },
    {
        "question": "Show products with more than {quantity} in stock",
        "sql": "SELECT * FROM products WHERE stock > {quantity}"
    },

    # ----- COUNT -----
    {
        "question": "How many customers are there?",
        "sql": "SELECT COUNT(*) FROM customers"
    },
    {
        "question": "Count all products",
        "sql": "SELECT COUNT(*) FROM products"
    },
    {
        "question": "How many orders do we have?",
        "sql": "SELECT COUNT(*) FROM orders"
    },
    {
        "question": "Count customers from {city}",
        "sql": "SELECT COUNT(*) FROM customers WHERE city = '{city}'"
    },
    {
        "question": "How many products are in {category} category?",
        "sql": "SELECT COUNT(*) FROM products WHERE category = '{category}'"
    },
    {
        "question": "Count orders with status {status}",
        "sql": "SELECT COUNT(*) FROM orders WHERE status = '{status}'"
    },

    # ----- ORDER BY -----
    {
        "question": "Show products ordered by price",
        "sql": "SELECT * FROM products ORDER BY price"
    },
    {
        "question": "List products from cheapest to most expensive",
        "sql": "SELECT * FROM products ORDER BY price ASC"
    },
    {
        "question": "Show products from most expensive to cheapest",
        "sql": "SELECT * FROM products ORDER BY price DESC"
    },
    {
        "question": "List customers alphabetically by name",
        "sql": "SELECT * FROM customers ORDER BY name ASC"
    },
    {
        "question": "Show orders by date, newest first",
        "sql": "SELECT * FROM orders ORDER BY order_date DESC"
    },
    {
        "question": "List orders by total amount, highest first",
        "sql": "SELECT * FROM orders ORDER BY total DESC"
    },

    # ----- LIMIT -----
    {
        "question": "Show top 5 most expensive products",
        "sql": "SELECT * FROM products ORDER BY price DESC LIMIT 5"
    },
    {
        "question": "Get the 10 most recent orders",
        "sql": "SELECT * FROM orders ORDER BY order_date DESC LIMIT 10"
    },
    {
        "question": "Show 3 cheapest products",
        "sql": "SELECT * FROM products ORDER BY price ASC LIMIT 3"
    },
    {
        "question": "List top 5 highest value orders",
        "sql": "SELECT * FROM orders ORDER BY total DESC LIMIT 5"
    },

    # ----- AGGREGATE FUNCTIONS -----
    {
        "question": "What is the average product price?",
        "sql": "SELECT AVG(price) FROM products"
    },
    {
        "question": "Find the total value of all orders",
        "sql": "SELECT SUM(total) FROM orders"
    },
    {
        "question": "What is the maximum product price?",
        "sql": "SELECT MAX(price) FROM products"
    },
    {
        "question": "Find the minimum order total",
        "sql": "SELECT MIN(total) FROM orders"
    },
    {
        "question": "What is the average order value?",
        "sql": "SELECT AVG(total) FROM orders"
    },
    {
        "question": "Find total stock across all products",
        "sql": "SELECT SUM(stock) FROM products"
    },

    # ----- GROUP BY -----
    {
        "question": "Count customers by city",
        "sql": "SELECT city, COUNT(*) FROM customers GROUP BY city"
    },
    {
        "question": "Show number of products per category",
        "sql": "SELECT category, COUNT(*) FROM products GROUP BY category"
    },
    {
        "question": "Count orders by status",
        "sql": "SELECT status, COUNT(*) FROM orders GROUP BY status"
    },
    {
        "question": "Find average product price by category",
        "sql": "SELECT category, AVG(price) FROM products GROUP BY category"
    },
    {
        "question": "Show total sales by order status",
        "sql": "SELECT status, SUM(total) FROM orders GROUP BY status"
    },

    # ----- JOIN (Simple) -----
    {
        "question": "Show orders with customer names",
        "sql": "SELECT orders.*, customers.name FROM orders JOIN customers ON orders.customer_id = customers.id"
    },
    {
        "question": "List order items with product names",
        "sql": "SELECT order_items.*, products.name FROM order_items JOIN products ON order_items.product_id = products.id"
    },
    {
        "question": "Find all orders for customers from {city}",
        "sql": "SELECT orders.* FROM orders JOIN customers ON orders.customer_id = customers.id WHERE customers.city = '{city}'"
    },

    # ----- DATE QUERIES -----
    {
        "question": "Find customers who joined in 2024",
        "sql": "SELECT * FROM customers WHERE YEAR(joined_date) = 2024"
    },
    {
        "question": "Show orders from this month",
        "sql": "SELECT * FROM orders WHERE MONTH(order_date) = MONTH(CURRENT_DATE) AND YEAR(order_date) = YEAR(CURRENT_DATE)"
    },
    {
        "question": "Find orders placed today",
        "sql": "SELECT * FROM orders WHERE order_date = CURRENT_DATE"
    },

    # ----- LIKE (Pattern matching) -----
    {
        "question": "Find customers whose name starts with J",
        "sql": "SELECT * FROM customers WHERE name LIKE 'J%'"
    },
    {
        "question": "Show products containing 'phone' in name",
        "sql": "SELECT * FROM products WHERE name LIKE '%phone%'"
    },
    {
        "question": "Find customers with gmail email",
        "sql": "SELECT * FROM customers WHERE email LIKE '%gmail.com'"
    },

    # ----- BETWEEN -----
    {
        "question": "Find products priced between {price1} and {price2} dollars",
        "sql": "SELECT * FROM products WHERE price BETWEEN {price1} AND {price2}"
    },
    {
        "question": "Show orders with total between {price1} and {price2}",
        "sql": "SELECT * FROM orders WHERE total BETWEEN {price1} AND {price2}"
    },

    # ----- IN -----
    {
        "question": "Find customers from New York or Los Angeles",
        "sql": "SELECT * FROM customers WHERE city IN ('New York', 'Los Angeles')"
    },
    {
        "question": "Show products in Electronics or Clothing category",
        "sql": "SELECT * FROM products WHERE category IN ('Electronics', 'Clothing')"
    },
    {
        "question": "Find orders that are pending or shipped",
        "sql": "SELECT * FROM orders WHERE status IN ('pending', 'shipped')"
    },
]


# ============================================================
# GENERATE VARIATIONS
# ============================================================

def fill_template(template):
    """Fill placeholders with random values."""
    question = template["question"]
    sql = template["sql"]

    # Replace placeholders with random values
    if "{city}" in question:
        city = random.choice(CITIES)
        question = question.replace("{city}", city)
        sql = sql.replace("{city}", city)

    if "{category}" in question:
        category = random.choice(CATEGORIES)
        question = question.replace("{category}", category)
        sql = sql.replace("{category}", category)

    if "{status}" in question:
        status = random.choice(STATUSES)
        question = question.replace("{status}", status)
        sql = sql.replace("{status}", status)

    if "{price}" in question:
        price = random.choice([10, 25, 50, 100, 200, 500, 1000])
        question = question.replace("{price}", str(price))
        sql = sql.replace("{price}", str(price))

    if "{quantity}" in question:
        quantity = random.choice([5, 10, 20, 50, 100])
        question = question.replace("{quantity}", str(quantity))
        sql = sql.replace("{quantity}", str(quantity))

    if "{price1}" in question:
        price1 = random.choice([10, 25, 50, 100])
        price2 = price1 + random.choice([50, 100, 200, 500])
        question = question.replace("{price1}", str(price1))
        question = question.replace("{price2}", str(price2))
        sql = sql.replace("{price1}", str(price1))
        sql = sql.replace("{price2}", str(price2))

    return question, sql


def generate_dataset(num_train=160, num_test=40):
    """Generate training and test datasets."""

    all_examples = []

    # Generate multiple variations of each template
    for _ in range(50):  # Generate 50 variations per template
        for template in TEMPLATES:
            question, sql = fill_template(template)
            all_examples.append({
                "question": question,
                "sql": sql
            })

    # Shuffle
    random.shuffle(all_examples)

    # Remove duplicates
    seen = set()
    unique_examples = []
    for ex in all_examples:
        key = (ex["question"], ex["sql"])
        if key not in seen:
            seen.add(key)
            unique_examples.append(ex)

    print(f"Generated {len(unique_examples)} unique examples")

    # Split into train and test
    train_data = unique_examples[:num_train]
    test_data = unique_examples[num_train:num_train + num_test]

    return train_data, test_data


# ============================================================
# FORMAT FOR OPENAI FINE-TUNING
# ============================================================

def format_for_openai(examples, output_file):
    """
    Convert examples to OpenAI fine-tuning format (JSONL).

    Each line is a conversation:
    {
        "messages": [
            {"role": "system", "content": "You are a SQL expert..."},
            {"role": "user", "content": "Show all customers"},
            {"role": "assistant", "content": "SELECT * FROM customers"}
        ]
    }
    """

    system_prompt = f"""You are a SQL expert. Convert natural language questions to SQL queries.

{SCHEMA}

Rules:
- Return ONLY the SQL query, nothing else
- Do not explain the query
- Use proper SQL syntax
"""

    with open(output_file, "w") as f:
        for ex in examples:
            conversation = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ex["question"]},
                    {"role": "assistant", "content": ex["sql"]}
                ]
            }
            f.write(json.dumps(conversation) + "\n")

    print(f"Saved {len(examples)} examples to {output_file}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 50)
    print("Generating Text-to-SQL Training Data")
    print("=" * 50)

    # Generate data
    train_data, test_data = generate_dataset(num_train=160, num_test=40)

    print(f"\nTraining examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")

    # Save in OpenAI format
    format_for_openai(train_data, "data/train.jsonl")
    format_for_openai(test_data, "data/test.jsonl")

    # Show a few examples
    print("\n" + "=" * 50)
    print("Sample Training Examples:")
    print("=" * 50)
    for i, ex in enumerate(train_data[:5]):
        print(f"\n[Example {i+1}]")
        print(f"Question: {ex['question']}")
        print(f"SQL:      {ex['sql']}")

    print("\n" + "=" * 50)
    print("Data generation complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
