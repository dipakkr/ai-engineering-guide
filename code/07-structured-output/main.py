"""
Structured Output with Instructor
===================================
Demonstrates reliable structured extraction using:
- Instructor library with Pydantic models
- Type-safe LLM responses
- Validation and retry on bad output
- Multiple extraction use cases

Run: python main.py
"""

import os
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
import anthropic
import instructor
from pydantic import BaseModel, Field, field_validator

load_dotenv()

# Patch Anthropic client with Instructor
client = instructor.from_anthropic(anthropic.Anthropic())

# ─── Schema Definitions ───────────────────────────────────────────────────────

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class SentimentAnalysis(BaseModel):
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence from 0 to 1")
    key_phrases: list[str] = Field(max_length=5, description="Up to 5 key phrases")
    summary: str = Field(max_length=100, description="One sentence summary")


class ContactInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if v and "@" not in v:
            raise ValueError(f"Invalid email: {v}")
        return v


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SupportTicket(BaseModel):
    category: str = Field(description="Type of issue: billing, technical, account, other")
    priority: Priority
    sentiment: Sentiment
    requires_human: bool = Field(description="True if issue needs human agent")
    suggested_action: str = Field(max_length=200, description="Recommended next step")
    extracted_order_id: Optional[str] = Field(default=None, description="Order ID if mentioned")


class RecipeIngredient(BaseModel):
    name: str
    amount: str
    unit: Optional[str] = None


class Recipe(BaseModel):
    title: str
    servings: int = Field(ge=1, le=100)
    prep_time_minutes: int = Field(ge=0)
    cook_time_minutes: int = Field(ge=0)
    ingredients: list[RecipeIngredient]
    steps: list[str] = Field(min_length=1)
    difficulty: str = Field(description="easy, medium, or hard")


# ─── Extraction Functions ─────────────────────────────────────────────────────

def analyze_sentiment(text: str) -> SentimentAnalysis:
    return client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        response_model=SentimentAnalysis,
        messages=[{"role": "user", "content": f"Analyze the sentiment of this text:\n\n{text}"}]
    )


def extract_contact(text: str) -> ContactInfo:
    return client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        response_model=ContactInfo,
        messages=[{"role": "user", "content": f"Extract contact information from:\n\n{text}"}]
    )


def classify_ticket(ticket_text: str) -> SupportTicket:
    return client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        response_model=SupportTicket,
        messages=[{
            "role": "user",
            "content": f"Classify this support ticket:\n\n{ticket_text}"
        }]
    )


def extract_recipe(recipe_text: str) -> Recipe:
    return client.messages.create(
        model="claude-sonnet-4-6",  # More capable for complex extraction
        max_tokens=800,
        response_model=Recipe,
        messages=[{"role": "user", "content": f"Extract the recipe details from:\n\n{recipe_text}"}]
    )


def main():
    # ── Sentiment Analysis ──
    print("1. SENTIMENT ANALYSIS")
    print("="*60)
    reviews = [
        "This product is absolutely amazing! Best purchase I've made all year.",
        "Terrible customer service. Waited 2 weeks and still no response.",
        "It's okay, nothing special. Does what it says but nothing more.",
    ]
    for review in reviews:
        result = analyze_sentiment(review)
        print(f"Text: {review[:60]}...")
        print(f"Sentiment: {result.sentiment.value} (confidence: {result.confidence:.0%})")
        print(f"Summary: {result.summary}\n")

    # ── Contact Extraction ──
    print("\n2. CONTACT EXTRACTION")
    print("="*60)
    texts = [
        "Please reach out to Sarah Johnson at sarah.j@acmecorp.com or call +1-555-0123. She's the VP of Engineering.",
        "Contact our team: support@techco.io",
    ]
    for text in texts:
        contact = extract_contact(text)
        print(f"Text: {text}")
        print(f"Extracted: name={contact.name}, email={contact.email}, phone={contact.phone}")
        print(f"           company={contact.company}, role={contact.role}\n")

    # ── Support Ticket Classification ──
    print("\n3. SUPPORT TICKET CLASSIFICATION")
    print("="*60)
    tickets = [
        "I was charged twice for order #ORD-2847. Please refund $59.99 immediately!",
        "The app keeps crashing when I try to upload photos. Running iOS 17 on iPhone 14.",
    ]
    for ticket in tickets:
        result = classify_ticket(ticket)
        print(f"Ticket: {ticket[:70]}...")
        print(f"Category: {result.category} | Priority: {result.priority.value}")
        print(f"Needs human: {result.requires_human} | Action: {result.suggested_action}")
        if result.extracted_order_id:
            print(f"Order ID: {result.extracted_order_id}")
        print()

    # ── Recipe Extraction ──
    print("\n4. RECIPE EXTRACTION")
    print("="*60)
    recipe_text = """
    Classic Pancakes (serves 4, 10 min prep, 15 min cook)
    You need 2 cups flour, 2 eggs, 1.5 cups milk, 3 tbsp butter (melted),
    1 tbsp sugar, 2 tsp baking powder, and a pinch of salt.
    Mix dry ingredients. Whisk wet ingredients separately. Combine gently.
    Cook on a greased pan over medium heat, 2-3 min per side. Easy difficulty.
    """
    recipe = extract_recipe(recipe_text)
    print(f"Recipe: {recipe.title} (serves {recipe.servings})")
    print(f"Time: {recipe.prep_time_minutes}min prep + {recipe.cook_time_minutes}min cook")
    print(f"Ingredients: {[f'{i.amount} {i.unit or \"\"} {i.name}'.strip() for i in recipe.ingredients[:3]]}...")
    print(f"Steps: {len(recipe.steps)} steps | Difficulty: {recipe.difficulty}")


if __name__ == "__main__":
    main()
