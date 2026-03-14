# Practice Problems

> **How to use this**: For each problem, try to answer it yourself before reading the solution skeleton. Time yourself: 45 minutes per problem (following the interview framework). The goal is not to get the "right" answer but to practice the process: clarify requirements, choose architecture, discuss tradeoffs.

**Prerequisites**: [Interview Framework](01-interview-framework.md), [Design Patterns Catalog](02-design-patterns-catalog.md)
**Related**: [Architecture Templates](03-architecture-templates.md), [Conceptual Questions](10-conceptual-questions.md)

---

## Level 1: Single-Pattern Problems (15-20 min)

These test whether you understand a single core pattern well enough to adapt it.

---

### P1: FAQ Bot for a Bank

**Problem:** Design a FAQ bot for a retail bank's website. It should answer questions about products (savings rates, loan requirements, card benefits) and be updated when rates change.

**Key clarifying questions:**
- How often do rates change? (Daily? Weekly?)
- Can it initiate actions (open an account) or only answer questions?
- Regulatory requirements? (Must not give investment advice, must include disclaimers)

**Minimum viable design:** Conversational RAG template with a knowledge base of FAQ documents. Semantic cache for repeated questions. Rate updates trigger re-indexing the affected documents (not a full re-index).

**Tradeoffs to discuss:**
- Freshness vs cost: re-index all documents daily (simple, catches everything) vs event-driven re-index on rate changes (fast, but requires event infrastructure)
- Guardrails: the bot must not give investment advice. How do you enforce this? (System prompt + output classifier + intent classification upfront)
- Escalation: when does the bot say "talk to a banker"? (Account opening, complaints, anything regulatory)

---

### P2: Internal HR Policy Bot

**Problem:** A company with 5,000 employees wants an HR chatbot that can answer questions about benefits, leave policies, and employee handbook content.

**Key clarifying questions:**
- Are policies different by country/region/employment type?
- Can it take actions (submit leave request) or only answer?
- What's the privacy requirement? (Can all employees see all policies?)

**Minimum viable design:** RAG with metadata filtering by region/employment-type. System prompt with strong "only answer from provided context" instruction to prevent hallucination of policy details.

**The tricky part:** When policies have jurisdiction-specific variations ("in California, leave policy is X; elsewhere it's Y"), the retrieval needs to fetch the right version for the user's location.

---

### P3: Product Description Generator

**Problem:** An e-commerce company wants to generate product descriptions for 100K new products per month from raw product specifications (name, dimensions, materials, features).

**Key clarifying questions:**
- What quality bar? (Human review of all, or spot check?)
- Consistency of brand voice?
- Time constraints? (Batch acceptable? Or must new products be live within 1 hour?)

**Minimum viable design:** Batch processing pipeline. Few-shot prompt with 5 examples of good product descriptions in the target brand voice. Structured output validation (description is N words, no prohibited phrases).

**This is a batch problem, not real-time.** Don't over-engineer with streaming or agents. Simple prompt + batch API = 50% cost savings.

---

## Level 2: Multi-Pattern Problems (30-45 min)

These require combining 2-4 patterns and discussing tradeoffs.

---

### P4: Medical Records Assistant

**Problem:** Design an AI assistant for physicians that can search a patient's medical history and answer questions during a patient visit. The system handles 10M patients, with records including notes, lab results, prescriptions, and imaging reports.

**Requirements to establish:**
- Latency requirement: physician is with patient, needs answers in <5 seconds
- PHI handling: all data is HIPAA-protected
- Actions: search only, no write actions
- Who can access: only the treating physician for the current patient

**Architecture sketch:**
```
Physician Query → Patient Context (current patient ID)
               → RBAC (verify physician has access to this patient)
               → Retrieve from patient's record only (hard isolation)
               → LLM synthesis with "only from records" instruction
               → Response with citations to specific records + dates
```

**Critical design decisions:**
1. **RBAC at the query level**: Only retrieve this patient's records. Never allow cross-patient retrieval, even accidentally.
2. **Audit logging**: Every query to a patient record must be logged (HIPAA requirement).
3. **No cloud API for PHI**: Must use on-premises or HIPAA BAA-covered cloud services.
4. **Citation required**: Every claim in the response must cite the specific record it came from. Hallucinated medical facts are dangerous.

**Tradeoff to discuss:** Self-hosted LLM (maintains PHI control, slower/more expensive) vs HIPAA BAA with Anthropic/Azure OpenAI (easier, but requires trust in provider).

---

### P5: Legal Research Tool

**Problem:** Design a system for lawyers to research case law. The corpus is 10M court cases. Lawyers ask natural language questions and need relevant precedents with citations.

**Requirements to establish:**
- Freshness: new cases added daily
- Citation accuracy: legal citations must be exact (case name, year, court, page)
- Multi-jurisdiction: federal + all 50 states
- Confidence: must indicate when it doesn't know

**Architecture sketch:**
- Hybrid search: BM25 for case citations + dense for concept matching
- Citation extraction as a separate extraction step (precise, validated)
- Jurisdiction metadata filter
- Source attribution: every statement must link to source case
- "I don't know" pattern: low retrieval confidence → "No directly relevant cases found" (never hallucinate a citation)

**The hard problem:** Hallucinated legal citations are a serious risk (the notorious Mata v. Avianca case where lawyers submitted AI-generated citations to non-existent cases). Your system must either: (a) only cite retrieved cases with exact IDs, or (b) implement a citation verification step.

---

### P6: Data Analysis Agent

**Problem:** Build an AI agent that can answer questions about a company's business data by writing and executing SQL queries against a production database with 200 tables.

**Requirements to establish:**
- Read-only or can it modify data?
- Which users have access? (All, or just analysts?)
- Database size: 200 tables, billions of rows
- Latency: complex SQL might take minutes

**Architecture sketch:**
```
User Question → Schema Retrieval (RAG over 200 table schemas)
             → SQL Generation (with schema context)
             → SQL Validation (syntax check, safety check for destructive ops)
             → Execute on read replica (never production write DB)
             → Format results
             → Answer question from results
```

**Safety requirements:** Execute only on a read replica. Reject any SQL containing INSERT/UPDATE/DELETE/DROP. Limit query time to 30 seconds. Limit result rows to 10K.

**The tricky part:** Schema is 200 tables. Don't inject all schemas into the prompt. Use RAG to retrieve the 5-10 relevant table schemas based on the question.

---

### P7: Content Moderation System

**Problem:** Design a content moderation pipeline for a social media platform handling 1M posts per hour. Posts include text, images, and video. Flag policy-violating content for human review or automatic removal.

**Requirements to establish:**
- Latency: must decide before post is visible (synchronous) or after (async)?
- False positive cost: wrongly removed content vs missed harmful content
- Categories: spam, hate speech, explicit content, misinformation
- Appeal process

**Architecture sketch:**
```
Post Submission → Fast rules (known spam domains, blocklisted words): 5ms
              → Image classifier (hash + NSFW model): 50ms
              → Text classifier (fine-tuned or LLM): 200ms
              → Decision: auto-remove / flag for review / approve
```

**Multi-tier approach:** Don't use LLM for everything. Rule-based catches obvious cases (known spam accounts, exact-match prohibited content) in microseconds. ML models catch most of the rest. Reserve LLM judgment for edge cases (context-dependent content that's borderline).

**Tradeoff:** Synchronous (post held for moderation, adds latency) vs async (post appears immediately, removed if flagged). Most platforms use async except for high-risk categories (explicit content, CSAM — always synchronous).

---

### P8: Knowledge Base from Customer Conversations

**Problem:** Build a system that automatically extracts and maintains a product FAQ knowledge base from customer support conversations. As customers ask questions and agents answer them, the knowledge base should update.

**Architecture sketch:**
```
Support Conversation → Extract Q/A pairs (LLM)
                    → Deduplicate (semantic similarity)
                    → Quality filter (was the answer good? rating from ticket resolution)
                    → Human review for new topics
                    → Merge into knowledge base
                    → Re-index for search
```

**Challenges:**
- Agent answers may be wrong or inconsistent (knowledge base needs quality filter)
- Same question asked many ways (deduplication needed)
- Outdated answers (when product changes, related KB entries need updating)

---

## Level 3: Complex System Design (45 min)

Full system design with architectural decisions, tradeoffs, and scale discussion.

---

### P9: AI-Powered News Aggregator

**Problem:** Design a personalized AI news service that ingests 100K articles/day from 1,000 sources, summarizes them, clusters related stories, and delivers a personalized digest to 1M daily active users.

**Full design expected:**
- Ingestion pipeline (fetch → clean → deduplicate → summarize → embed)
- Story clustering (near-duplicate detection + semantic clustering)
- Personalization (user embedding from history, match to story clusters)
- Delivery (daily digest generation, push notification timing)
- Freshness (breaking news pipeline vs daily digest pipeline)

**Scale constraints:**
- 100K articles × 500 tokens avg = 50M tokens/day to summarize
- At Claude Sonnet pricing: $150/day for summarization
- 1M users × personalized digest: can't generate each individually; use template + user-specific top stories

---

### P10: Real-Time Meeting Assistant

**Problem:** Design an AI assistant for video calls that provides real-time transcription, summarizes key decisions as they happen, answers factual questions, and generates action items.

**Constraints:**
- Real-time transcription with <500ms delay
- Incremental processing (can't wait for meeting to end)
- Multiple speakers
- Integration with calendar, tasks, notes

**Architecture considerations:**
- Whisper for transcription (streaming mode)
- Sliding window summary (summarize last 5 minutes, maintain rolling context)
- Action item detection triggered by specific phrases ("I'll take care of...", "we agreed to...")
- Post-meeting processing: final summary, clean action item list, searchable transcript

---

### P11: Developer Tooling for Internal APIs

**Problem:** A company with 500 internal APIs wants developers to be able to ask questions and get working code samples. "How do I create a new user with the auth service?" should return a working code example.

**Key design decisions:**
- Index API documentation + OpenAPI specs + real code examples from the internal repo
- Generate code with the company's actual SDK patterns, not generic REST calls
- Validate generated code against the OpenAPI spec (does the endpoint exist? Are required fields included?)
- Keep examples current as APIs evolve

---

## Practice Protocol

For each problem, time yourself through this sequence:

```
Minutes 0-5: Clarify requirements
Minutes 5-15: Draw the architecture (Mermaid-style, even on paper)
Minutes 15-25: Walk through the main data flows
Minutes 25-35: Identify the hardest design decisions
Minutes 35-45: Scale and cost discussion
```

After practicing a problem, ask yourself:
1. Did I miss any requirements that would change the architecture?
2. Did I identify the 2-3 hardest design decisions?
3. Could I explain the tradeoffs between alternatives I didn't choose?
4. Did I give concrete numbers (latency, cost, scale)?

The goal is not perfect answers. The goal is the habit of structured thinking under time pressure.
