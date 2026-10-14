# CareerPilot

Finds jobs on the public ATS APIs, scores how well you actually fit, and writes a CV and
cover letter where **every line traces to a fact in your profile**. The check that
guarantees it is written in code, not in a prompt.

```
collect → extract requirements → score fit → draft bullets → gate → artefacts + audit trail
   code         model               code          model       code+model      code
```

The model is only ever allowed to produce language and to point at ids. Scores, numbers,
file paths and every rejection decision are computed in Python. A hallucinated id
resolves to nothing, so the failure mode is an under-score, never an inflated one.

## The groundedness gate

Nothing ships unless it survives five checks, run cheapest first
(`src/careerpilot/ground/gate.py`):

| # | Check | Who runs it |
|---|---|---|
| 1 | `no_citation` — the bullet cites no profile atom | code |
| 2 | `unknown_atom` — it cites an id the fact base does not contain | code |
| 3 | `unsourced_number` — it states a digit absent from the cited atoms | code |
| 4 | `misattributed_skill` — it names a real skill on work that never used it | code |
| 5 | `not_entailed` — a judge shown only the bullet and its evidence disagrees | model |

A bullet that fails is **dropped**, never softened, hedged or retried into compliance.
The audit file says why, for every line.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
cp data/profile.example.yaml data/profile.yaml   # then fill in your real contact details
```

`python -m pip` rather than `pip`, on purpose: on Debian and Ubuntu the bare `pip` on
`PATH` can still resolve to system Python, which refuses installs outright under PEP 668
and produces an error that reads as though the project is broken.

`data/profile.yaml` is the real fact base and is gitignored; `profile.example.yaml` is
the same atoms with placeholder contact details, so CI and the eval thresholds run
anywhere. The CLI prefers the real one whenever it exists.

`contact.location` is load-bearing, not a signature block: the country in it decides
which postings you are shown at all (see *Location is a verdict*).

Model calls need a credential — `GROQ_API_KEY` or `GEMINI_API_KEY`. Without one the tool
still collects, stores and searches, and refuses to invent an answer:

```bash
export GROQ_API_KEY=gsk_...
```

## Use it

`<uid>` means a job id copied out of `careerpilot jobs` — they are per-store, so there
is no example uid that would survive being run.

```bash
careerpilot collect                      # every curated board → SQLite (Greenhouse, Lever, Ashby, bdtechjobs)
careerpilot jobs                         # what you can actually work, from the address in your profile
careerpilot jobs "vector search"         # BM25 search over what was fetched
careerpilot jobs --where bd              # only postings located in your country
careerpilot jobs --where all             # everything stored, each row carrying its location verdict
careerpilot score-job <uid>              # requirement extraction + fit score, with the cited atoms
careerpilot tailor <uid>                 # CV + cover letter + audit trail into var/out/<job-uid>/
careerpilot traces                       # calls per task, failures, fallbacks, cost
```

### Location is a verdict, not a column

The country is read out of `contact.location` in your profile, so the same store answers
differently for a candidate in Dhaka and one in Lisbon. A posting counts as workable if
it is located in your country, or if it states remote and names no country or region that
excludes it.

`is_remote` alone is not trusted, because it is wrong in a specific and common way: Ashby
sets `isRemote: true` on roles whose only stated location is an office. In an October 2026
collect that was 143 rows reading `San Francisco` while claiming remote. A flag that cannot
grant permission on its own costs some false negatives and removes all of those.

A region is read the same way — as a boundary, not a hint. `Remote, AMER` and
`Remote, Global` are both remote and both unscoped by country, but only one of them is
open to someone in Dhaka, so each bucket carries the set of countries it stands for and
is judged against yours. `Remote, Europe` is workable for a candidate in Lisbon and
blocked for one in Dhaka; `Remote, AMER` is the other way round. Nothing about the
posting changed.

The consequence is that fit and reach stay separate. `score-job` prints `92.0/100 — strong`
and then `location [NOT WORKABLE] from Bangladesh: flagged remote, but lists Washington, DC`.
`tailor` refuses a blocked posting **before making any model call**, since a tailored CV
for a job restricted to another country is not a near miss. `--ignore-location` overrides it.

Fit lands in one of three bands, and the band is a rule rather than an opinion: `strong`
requires a score of 70 **and** zero unmet must-haves.

### Recording cassettes

Tests and CI never call a model. They replay recorded responses
(`tests/fixtures/cassettes/`), which is what makes them free and deterministic.
Recordings are made deliberately, because that is the only step that costs money:

```bash
careerpilot record-fixtures <uid> <uid>
```

## Tests and evals

```bash
pytest -m "not eval and not live"   # behaviour: parsing, retries, scoring, the gate, rendering
pytest -m eval                      # thresholds, against scripted fakes
mypy src tests                      # strict, with the pydantic plugin
ruff check src tests
```

The eval tier enforces numbers, not vibes (`tests/eval/`):

- **zero** fabricated lines ship under a judge that believes everything it is shown;
- every fabricated line fails for a *named code reason*, and all four code checks are exercised;
- truthful-claim acceptance stays above 85%, so a safe gate is not allowed to be useless;
- an invented citation is worth **exactly 0.0 points** on every labelled job.

The evals also pin the hole that code cannot close: a *real but irrelevant* atom id does
add weight. What bounds it is the seniority gate (pure date arithmetic) plus the
must-have rule, and an audit trail that names the atoms so a human can see the drift.

`.github/workflows/ci.yml` runs all of that on every push with no credential in the
environment, and a separate nightly job runs the same labelled corpus against the real
model — that job fails loudly rather than reporting green when the keys are missing.

## What this is not

- Not a web app. It is a CLI plus CI; nothing here needs deploying.
- Not a scraper. Jobs come from Greenhouse, Lever and Ashby job-board APIs, and from
  bdtechjobs' public JSON API. LinkedIn is not touched.
- Not a ghostwriter. It cannot tell you you did something you did not do, and a line it
  cannot attribute is a line that does not appear in your CV.

See [docs/PLAN.md](docs/PLAN.md) for the brief this was built from.
