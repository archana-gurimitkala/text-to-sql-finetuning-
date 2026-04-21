"""
app.py - Gradio UI for Text-to-SQL

Run this after fine-tuning is complete:
    python app.py
"""

import gradio as gr
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Load fine-tuned model ID — env var takes priority (for HF Spaces), fallback to file
def load_model_id():
    env_id = os.getenv("MODEL_ID")
    if env_id:
        return env_id
    try:
        with open("model_id.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

MODEL_ID = load_model_id()

# System prompt
SYSTEM_PROMPT = """You are a SQL expert. Convert natural language questions to SQL queries.

Rules:
- Return ONLY the SQL query, nothing else
- Do not explain the query
- Use proper SQL syntax
- Use the table and column names from the context provided
"""


def generate_sql(question, context=""):
    """Generate SQL from natural language question."""

    if not MODEL_ID:
        return "Error: model_id.txt not found. Run fine-tuning first."

    # Build user message
    if context:
        user_message = f"Context: {context}\n\nQuestion: {question}"
    else:
        user_message = question

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"


def load_evaluation_metrics():
    """Load evaluation metrics from JSON file."""
    try:
        with open("evaluation_results.json", "r") as f:
            data = json.load(f)
            return data.get("metrics", {})
    except FileNotFoundError:
        return None


def format_metrics_display():
    """Format metrics for display in Gradio."""
    metrics = load_evaluation_metrics()
    
    if not metrics:
        return "Evaluation results not found. Please run evaluate.py first."
    
    # Format metrics as a nice HTML table
    html = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px;">
        <h2 style="color: #2c3e50; margin-bottom: 20px;">Model Evaluation Metrics</h2>
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
            <tr style="background-color: #3498db; color: white;">
                <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Metric</th>
                <th style="padding: 12px; text-align: right; border: 1px solid #ddd;">Value</th>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Exact Match Accuracy</strong></td>
                <td style="padding: 10px; text-align: right; border: 1px solid #ddd; color: {'#27ae60' if metrics.get('exact_match_accuracy', 0) >= 50 else '#e74c3c'}; font-weight: bold;">{metrics.get('exact_match_accuracy', 0):.2f}%</td>
            </tr>
            <tr style="background-color: #f8f9fa;">
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Token Accuracy</strong></td>
                <td style="padding: 10px; text-align: right; border: 1px solid #ddd; color: {'#27ae60' if metrics.get('mean_token_accuracy', 0) >= 80 else '#e74c3c'}; font-weight: bold;">{metrics.get('mean_token_accuracy', 0):.2f}%</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Mean Error</strong></td>
                <td style="padding: 10px; text-align: right; border: 1px solid #ddd;">{metrics.get('mean_error', 0):.4f}</td>
            </tr>
            <tr style="background-color: #f8f9fa;">
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>RMSE</strong></td>
                <td style="padding: 10px; text-align: right; border: 1px solid #ddd;">{metrics.get('rmse', 0):.4f}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Total Examples</strong></td>
                <td style="padding: 10px; text-align: right; border: 1px solid #ddd;">{metrics.get('total_examples', 0)}</td>
            </tr>
            <tr style="background-color: #f8f9fa;">
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Exact Matches</strong></td>
                <td style="padding: 10px; text-align: right; border: 1px solid #ddd;">{metrics.get('exact_matches', 0)}</td>
            </tr>
        </table>
    </div>
    """
    return html


def get_evaluation_image():
    """Get the path to the evaluation results image."""
    image_path = "evaluation_results.png"
    if os.path.exists(image_path):
        return image_path
    return None


# Create tabbed interface
with gr.Blocks(title="Text-to-SQL Generator") as demo:
    gr.Markdown(f"# Text-to-SQL Generator\n\nConvert natural language to SQL using fine-tuned model: {MODEL_ID}")
    
    with gr.Tabs():
        # Tab 1: SQL Generator
        with gr.Tab("SQL Generator"):
            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Show all customers from New York",
                        lines=2
                    )
                    schema_input = gr.Textbox(
                        label="Database Schema (optional)",
                        placeholder="CREATE TABLE customers (id INT, name VARCHAR, city VARCHAR)",
                        lines=5
                    )
                    with gr.Row():
                        clear_btn = gr.Button("Clear", variant="secondary")
                        submit_btn = gr.Button("Submit", variant="primary")
                
                with gr.Column():
                    sql_output = gr.Textbox(
                        label="Generated SQL",
                        lines=5,
                        interactive=False
                    )
                    flag_btn = gr.Button("Flag", variant="secondary")
            
            # Examples
            gr.Examples(
                examples=[
                    ["Show all customers", "CREATE TABLE customers (id INT, name VARCHAR, city VARCHAR)"],
                    ["Find products cheaper than $50", "CREATE TABLE products (id INT, name VARCHAR, price DECIMAL)"],
                    ["Count orders by status", "CREATE TABLE orders (id INT, customer_id INT, status VARCHAR)"],
                    ["Show top 5 most expensive products", "CREATE TABLE products (id INT, name VARCHAR, price DECIMAL)"],
                ],
                inputs=[question_input, schema_input]
            )
            
            # Connect buttons
            submit_btn.click(
                fn=generate_sql,
                inputs=[question_input, schema_input],
                outputs=sql_output
            )
            
            clear_btn.click(
                fn=lambda: ("", "", ""),
                outputs=[question_input, schema_input, sql_output]
            )
        
        # Tab 2: Evaluation Metrics
        with gr.Tab("Evaluation Metrics"):
            gr.Markdown("## Model Performance Evaluation")
            
            # Display metrics table
            metrics_html = gr.HTML(value=format_metrics_display())
            
            # Display evaluation results image
            evaluation_image = get_evaluation_image()
            if evaluation_image:
                gr.Markdown("### Evaluation Visualizations")
                gr.Image(
                    value=evaluation_image,
                    label="Evaluation Results",
                    type="filepath"
                )
            else:
                gr.Markdown("⚠️ Evaluation results image not found. Please run `python evaluate.py` to generate it.")
            
            # Display fine-tuning screenshots
            gr.Markdown("### Fine-Tuning Process")
            finetuning1_path = "finetunning1.png"
            finetuning2_path = "fineTunning2.png"
            
            if os.path.exists(finetuning1_path):
                gr.Image(
                    value=finetuning1_path,
                    label="Fine-Tuning Screenshot 1",
                    type="filepath"
                )
            
            if os.path.exists(finetuning2_path):
                gr.Image(
                    value=finetuning2_path,
                    label="Fine-Tuning Screenshot 2",
                    type="filepath"
                )


if __name__ == "__main__":
    if not MODEL_ID:
        print("Warning: model_id.txt not found!")
        print("Run fine-tuning first.")
    else:
        print(f"Using model: {MODEL_ID}")

    print("Starting Gradio app...")
    demo.launch()
