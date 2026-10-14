# CareerPilot — the original brief

> The prompt and plan this repository was built from, verbatim. Everything under
> `src/careerpilot/` implements idea 1 of three; the grounding guarantees in
> `docs` are enforced by `tests/eval/`, not by prompt wording.

---

I put the full plan in the side panel. For each agent it has how it works, where AI is used and where code takes over, tests and evals, CI/CD, a 3-day build plan, how it makes money, and how to write it on your CV. Here are the three ideas:

| # | Agent | What it does | Builds on your existing work | How it makes money |
|---|---|---|---|---|
| 1 | **CareerPilot** | Finds jobs, scores how well you fit, and tailors your CV and cover letter. A check written in code makes sure **every line comes from your real experience**. It also includes a "digital twin" that answers recruiter questions on your portfolio | RAG plus your Groq/Gemini fallback | Students in Bangladesh, university career clubs, Upwork |
| 2 | **Dokan Agent** | Takes orders for Facebook sellers in Bangla, Banglish or English. **Prices are calculated in code, not by the AI.** The seller approves each order, then it's booked with a courier | Bangla text work from Bornomala, plus RAG | Monthly fee per Facebook page |
| 3 | **PR Sentinel** | Runs on GitHub pull requests. Real security scanners find the issues, and an agent filters out false alarms. It only suggests a fix if the patch applies, the tests pass and a re-scan comes back clean. It also works as an MCP server | Your CTF and security background | GitHub Marketplace, freelance security reviews |

## How I'd approach it
- **The AI only handles language; code handles facts.** Prices, scores, file paths and validation are all plain code. Every AI call has to return data in a fixed format that gets checked. If it fails, it retries at most twice, then hands off to a human. Model versions are pinned.
- **Evals are checked in CI.** A pull request fails if accuracy drops below a set threshold. Tests run on every push using a fake AI and recorded responses, so they cost nothing and give the same result every time. Evals against the real model run nightly.
- **Each project uses a different framework** from the two courses (OpenAI Agents SDK, LangGraph, MCP), which shows range.

## Why this is the right bet
- In an analysis of 24 agentic AI job posts I found, **evaluation and observability came up as often as LangChain (9 of 24)**, and MCP is becoming more common. Most fresher portfolios don't have evals at all.
- Upwork data I found for May 2026 shows about **3,300 "AI automation" job postings**. The lasting demand is for work tied to a measurable business result, not quick AI add-ons.

## What to watch out for
- **Don't scrape LinkedIn.** CareerPilot should pull jobs from Greenhouse, Lever and Ashby, which publish their job listings openly.
- **Dokan Agent needs Meta App Review** before it can serve many sellers. Testing mode is enough for demos and pilots, so start the review early.
- **Check the courier companies' current API documentation** (Steadfast or Pathao) before you rely on them.

**Build order:** CareerPilot → PR Sentinel → Dokan Agent. CareerPilot helps your own job search right away. PR Sentinel sends the strongest signal to employers. Dokan Agent needs pilot sellers, so start talking to sellers now while you build the first two.

Should I start on CareerPilot next? I'd first set up the shared starter repo (the single module all AI calls go through, a fake AI for tests, the eval setup and the GitHub Actions workflow), then build CareerPilot on top of it.