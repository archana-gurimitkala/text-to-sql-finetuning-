"""
check_status.py - Check fine-tuning job status

Run this to see if your fine-tuning is complete:
    python check_status.py
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Load job ID
with open("job_id.txt", "r") as f:
    job_id = f.read().strip()

# Get status
job = client.fine_tuning.jobs.retrieve(job_id)

print("=" * 50)
print("Fine-tuning Job Status")
print("=" * 50)
print(f"Job ID: {job_id}")
print(f"Status: {job.status}")

if job.status == "succeeded":
    print(f"\n✅ COMPLETE!")
    print(f"Model ID: {job.fine_tuned_model}")

    # Save model ID
    with open("model_id.txt", "w") as f:
        f.write(job.fine_tuned_model)
    print("Model ID saved to model_id.txt")
    print("\nNext step: python app.py")

elif job.status == "failed":
    print(f"\n❌ FAILED")
    print(f"Error: {job.error}")

elif job.status == "cancelled":
    print(f"\n⚠️ CANCELLED")

else:
    print(f"\n⏳ Still in progress...")
    print("Run this script again in a few minutes.")
