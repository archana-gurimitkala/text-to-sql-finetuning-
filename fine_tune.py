"""
fine_tune.py - Fine-tune GPT model on our Text-to-SQL data

This script:
1. Uploads training data to OpenAI
2. Starts a fine-tuning job
3. Monitors progress
4. Saves the model ID when done
"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# ============================================================
# STEP 1: UPLOAD TRAINING FILE
# ============================================================

def upload_training_file(file_path):
    """Upload training data to OpenAI."""

    print(f"Uploading {file_path}...")

    with open(file_path, "rb") as f:
        response = client.files.create(
            file=f,
            purpose="fine-tune"
        )

    file_id = response.id
    print(f"Uploaded! File ID: {file_id}")

    return file_id


# ============================================================
# STEP 2: START FINE-TUNING JOB
# ============================================================

def start_fine_tuning(training_file_id):
    """Start a fine-tuning job."""

    print("\nStarting fine-tuning job...")

    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model="gpt-4o-mini-2024-07-18",  # Base model to fine-tune
        hyperparameters={
            "n_epochs": 3  # Number of training passes
        }
    )

    job_id = response.id
    print(f"Fine-tuning job started! Job ID: {job_id}")

    return job_id


# ============================================================
# STEP 3: MONITOR PROGRESS
# ============================================================

def monitor_job(job_id):
    """Monitor fine-tuning job until complete."""

    print("\nMonitoring progress...")
    print("(This can take 10-30 minutes)\n")

    while True:
        response = client.fine_tuning.jobs.retrieve(job_id)
        status = response.status

        print(f"Status: {status}")

        if status == "succeeded":
            model_id = response.fine_tuned_model
            print(f"\n✅ Fine-tuning complete!")
            print(f"Model ID: {model_id}")
            return model_id

        elif status == "failed":
            print(f"\n❌ Fine-tuning failed!")
            print(f"Error: {response.error}")
            return None

        elif status == "cancelled":
            print(f"\n⚠️ Fine-tuning was cancelled")
            return None

        # Wait 30 seconds before checking again
        time.sleep(30)


# ============================================================
# STEP 4: SAVE MODEL ID
# ============================================================

def save_model_id(model_id):
    """Save the fine-tuned model ID for later use."""

    with open("model_id.txt", "w") as f:
        f.write(model_id)

    print(f"\nModel ID saved to model_id.txt")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 50)
    print("Text-to-SQL Fine-Tuning")
    print("=" * 50)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY not found!")
        print("Please create a .env file with your API key:")
        print("  OPENAI_API_KEY=sk-...")
        return

    # Step 1: Upload training file
    training_file_id = upload_training_file("data/train.jsonl")

    # Step 2: Start fine-tuning
    job_id = start_fine_tuning(training_file_id)

    # Step 3: Monitor until complete
    model_id = monitor_job(job_id)

    # Step 4: Save model ID
    if model_id:
        save_model_id(model_id)

        print("\n" + "=" * 50)
        print("Next steps:")
        print("1. Run: python test_model.py")
        print("2. Or run: python app.py (for Gradio UI)")
        print("=" * 50)


if __name__ == "__main__":
    main()
