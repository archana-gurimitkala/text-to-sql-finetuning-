"""
schema.py - Defines our fake e-commerce database structure

This file describes what tables and columns exist.
The model will learn to write SQL queries for these tables.
"""

# Database schema as a string (we'll include this in prompts)
SCHEMA = """
Database Schema:

Table: customers
- id (INT, PRIMARY KEY)
- name (VARCHAR)
- email (VARCHAR)
- city (VARCHAR)
- joined_date (DATE)

Table: products
- id (INT, PRIMARY KEY)
- name (VARCHAR)
- price (DECIMAL)
- category (VARCHAR)
- stock (INT)

Table: orders
- id (INT, PRIMARY KEY)
- customer_id (INT, FOREIGN KEY -> customers.id)
- order_date (DATE)
- total (DECIMAL)
- status (VARCHAR: 'pending', 'shipped', 'delivered', 'cancelled')

Table: order_items
- id (INT, PRIMARY KEY)
- order_id (INT, FOREIGN KEY -> orders.id)
- product_id (INT, FOREIGN KEY -> products.id)
- quantity (INT)
- price (DECIMAL)
"""

# List of tables (for reference)
TABLES = ["customers", "products", "orders", "order_items"]

# Column details (for generating varied questions)
COLUMNS = {
    "customers": ["id", "name", "email", "city", "joined_date"],
    "products": ["id", "name", "price", "category", "stock"],
    "orders": ["id", "customer_id", "order_date", "total", "status"],
    "order_items": ["id", "order_id", "product_id", "quantity", "price"]
}

# Sample values (for generating realistic questions)
CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Seattle", "Boston", "Miami"]
CATEGORIES = ["Electronics", "Clothing", "Books", "Home", "Sports", "Toys", "Food", "Beauty"]
STATUSES = ["pending", "shipped", "delivered", "cancelled"]
