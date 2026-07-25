# 🛒 AI Shopping Agent using OpenAI

An intelligent AI-powered shopping assistant that understands natural language queries and builds the **best possible shopping cart** based on product ratings, prices, categories, reviews, and user preferences.

Instead of searching through thousands of products manually, users simply describe what they need, and the AI agent recommends the most suitable products along with alternative options, helping users make informed purchasing decisions.

---

# Demo

### User Query

> "I want to buy a gaming setup under ₹1,20,000 including a monitor, keyboard, mouse and headphones."

### Agent Response

✅ Recommended Cart

| Product | Price | Rating |
|----------|--------|--------|
| LG Ultragear Monitor | ₹24,999 | ⭐4.6 |
| Logitech G Pro Keyboard | ₹8,999 | ⭐4.7 |
| Logitech G502 Mouse | ₹4,999 | ⭐4.8 |
| HyperX Cloud II Headset | ₹6,499 | ⭐4.7 |

**Total:** ₹45,496

### Alternative Suggestions

- Razer BlackWidow Keyboard
- SteelSeries Rival Mouse
- ASUS TUF Gaming Monitor

The agent explains **why** every recommendation was selected.

---

# Project Goal

Traditional e-commerce search requires users to manually compare hundreds of products.

This project aims to build an **AI Shopping Agent** capable of:

- Understanding natural language shopping requests
- Searching thousands of products
- Comparing alternatives
- Building an optimal shopping basket
- Recommending higher-rated substitutes
- Respecting user budget constraints
- Explaining every recommendation

The final goal is to provide a conversational shopping experience similar to talking with a knowledgeable shopping assistant.

---

# Dataset

The project uses an Amazon product dataset containing information such as:

| Column | Description |
|----------|-------------|
| name | Product title |
| main_category | Main category |
| sub_category | Product sub-category |
| image | Product image URL |
| link | Amazon product link |
| ratings | Average customer rating |
| no_of_ratings | Number of reviews |
| discount_price | Current selling price |
| actual_price | Original MRP |

Example:

```text
Lloyd 1.5 Ton 3 Star Inverter Split AC
Category : Appliances
Sub Category : Air Conditioners
Rating : 4.2
Reviews : 2255
Discount Price : ₹32999
Actual Price : ₹58990
```

---

# Key Features

## Intelligent Shopping Assistant

The agent understands natural language instead of keyword search.

Example:

> "Suggest a washing machine under ₹30,000 with excellent reviews."

---

## Personalised Product Recommendations

Recommendations are generated using multiple signals:

- Product ratings
- Number of customer reviews
- Price
- Discount percentage
- Category relevance
- User preferences

---

## Smart Cart Builder

The user may ask:

> "Build me a complete home office setup under ₹80,000."

The agent automatically selects:

- Laptop
- Monitor
- Keyboard
- Mouse
- Webcam
- Speakers

while keeping the total cost within budget.

---

## Alternative Recommendations

Every recommended product includes alternatives.

Example

Recommended

- Sony WH-1000XM5

Alternatives

- Bose QC Ultra
- Sennheiser Momentum 4
- JBL Tour One

---

## Budget Optimisation

The shopping agent continuously monitors:

- Total cart value
- Remaining budget
- Cost savings
- Discount opportunities

If the basket exceeds the budget, the agent intelligently swaps products with comparable alternatives.

---

## Explainable Recommendations

Every recommendation includes reasoning.

Example

> Selected because it has a 4.8★ rating from over 12,000 reviews, offers 25% discount, and provides better value compared to similar products.

---

## Conversational Shopping

Users can continue refining recommendations.

Example conversation

User:

> Show me cheaper options.

Agent:

> Here are three alternatives that reduce your total by ₹8,500 while maintaining an average rating above 4.4★.

---

# Agent Workflow

```
                    User Query
                         │
                         ▼
              OpenAI Shopping Agent
                         │
         ┌───────────────┼────────────────┐
         │               │                │
         ▼               ▼                ▼
   Understand      Search Dataset     Parse Budget
 User Intent
         │
         ▼
 Rank Candidate Products
         │
         ▼
 Build Optimal Shopping Basket
         │
         ▼
 Find Alternative Products
         │
         ▼
 Explain Recommendations
         │
         ▼
      Final Shopping Cart
```

---

# Recommendation Strategy

Products are ranked using multiple criteria.

Example scoring function:

```
Recommendation Score =

0.40 × Rating
+ 0.25 × Review Score
+ 0.20 × Discount Score
+ 0.10 × Category Similarity
+ 0.05 × Price Efficiency
```

The weights can be tuned depending on business requirements.

---

# Technology Stack

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| LLM | OpenAI GPT-4.1 / GPT-5 |
| Agent Framework | OpenAI Agents SDK / LangGraph |
| Embeddings | OpenAI text-embedding-3-large |
| Vector Database | FAISS / ChromaDB |
| Data Processing | Pandas |
| Search | Semantic Search |
| API | FastAPI |
| UI | Streamlit |
| Deployment | Docker |

---

# Repository Structure

```
shopping-agent/
│
├── data/
│     amazon_products.csv
│
├── app/
│     main.py
│     api.py
│
├── agents/
│     shopping_agent.py
│     planner.py
│     recommender.py
│
├── retrieval/
│     embeddings.py
│     vector_store.py
│
├── utils/
│     ranking.py
│     pricing.py
│
├── prompts/
│     system_prompt.txt
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

---

# Example Queries

## Product Recommendation

> Recommend the best refrigerator under ₹40,000.

---

## Cheapest Alternative

> Show me cheaper options.

---

## Premium Alternative

> I can spend another ₹10,000. Can you improve the recommendations?

---

## Shopping Basket

> Build a complete photography kit under ₹1,50,000.

---

## Category Search

> Show the highest rated air conditioners.

---

## Compare Products

> Compare Samsung and LG washing machines.

---

## Brand Preference

> Recommend only Apple products.

---

## Discount Search

> Show products with more than 40% discount.

---

# Future Enhancements

- Multi-agent architecture (Planner, Retriever, Recommender, Critic)
- Live product catalogue integration
- Real-time price tracking
- User preference memory
- Review summarisation using LLMs
- Price history visualisation
- Cross-category bundle optimisation
- Personalised recommendations using previous purchases
- Voice-enabled shopping assistant
- Multi-language support
- Image-based product search
- Automatic coupon and deal discovery

---

# Potential Agent Architecture

```
                 User
                   │
                   ▼
             Shopping Agent
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
Intent Agent   Search Agent   Budget Agent
    │              │              │
    └──────────────┼──────────────┘
                   ▼
          Product Ranking Agent
                   │
                   ▼
         Alternative Finder Agent
                   │
                   ▼
           Cart Optimisation Agent
                   │
                   ▼
            Explanation Generator
                   │
                   ▼
             Final Recommendation
```

---

# Why OpenAI?

OpenAI models provide:

- Excellent natural language understanding
- High-quality reasoning for comparing products
- Context-aware conversations
- Budget-aware planning
- Explainable recommendations
- Multi-turn dialogue support
- Tool calling for structured retrieval
- Easy integration with external APIs

---

# Future Vision

The long-term vision is to evolve this project into a fully autonomous shopping copilot that can:

- Understand user preferences over time
- Build complete shopping carts across multiple categories
- Optimise purchases for value, quality, and budget
- Compare equivalent products from different brands
- Explain trade-offs between recommendations
- Integrate with live retailer APIs for availability and pricing
- Adapt recommendations based on follow-up conversations

Rather than acting as a simple search engine, the AI Shopping Agent becomes an intelligent decision-making partner that helps users purchase the right products with confidence.

---

# License

This project is intended for educational and research purposes. Product data belongs to their respective owners, and any trademarks, images, or links remain the property of their original sources.