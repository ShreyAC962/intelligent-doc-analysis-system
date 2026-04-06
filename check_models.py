"""
Model Checker - Find Available Gemini Models
"""

import google.generativeai as genai
from config import settings

# Configure API
genai.configure(api_key=settings.google_ai_api_key)

print("=" * 60)
print("Available Gemini Models")
print("=" * 60)
print()

# List all available models
print("📋 Checking available models...")
print()

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✅ {model.name}")
        print(f"   Display Name: {model.display_name}")
        print(f"   Supported: {model.supported_generation_methods}")
        print()

print("=" * 60)