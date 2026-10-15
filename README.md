# CareerPilot

**A job-search agent that will not let you put something on your CV that you did not do.**

It pulls live postings from the job-board APIs that are actually public — Greenhouse,
Ashby, Lever and bdtechjobs — scores how well your real experience fits each one, and
writes a CV and cover letter for the job you pick. Every sentence it produces has to
point at a specific fact in your profile file, and the check that enforces that is Python
comparing ids, not a prompt asking nicely.

## What the agent is

One rule shapes the whole design: **the model handles language; code handles facts.**

```
collect → extract requirements → score fit → draft bullets → gate → artefacts + audit
   code         model               code          model        code+model      code
```

| Stage | Who does it | What it can and cannot do |
|---|---|---|
| Find jobs | code | Reads public ATS APIs. No scraping, no LinkedIn. |
| Decide what the role wants | model | Returns a fixed, validated shape — never prose. |
| Score your fit | code | Arithmetic over your own atoms. Pinned weights and bands. |
| Decide where you can work | code | Your address against the posting's location text. |
| Write the bullets | model | May only phrase things and cite atom ids it was given. |
| Ship or drop each bullet | code + model | Four code checks then one entailment judge. |
| Lay out the files | code | Template, not the model. |

The failure mode is deliberately one-sided. A hallucinated atom id matches nothing, so it
can only *lower* your score — it cannot invent a strength. An invented number is dropped by
`unsourced_number` before it reaches the document.

**What you get, per job:** a tailored CV, a cover letter, and an audit trail naming every
line that shipped and every line that was dropped and why. A CV that comes out shorter
than you expected is the system working, not failing.

**Who it is built for:** a job seeker in Bangladesh. That is not a hardcoded assumption —
the country is read out of your own profile, so the same build answers differently for a
candidate in Dhaka and one in Lisbon (see *Location is a verdict*).

It is a command-line tool that runs on your machine. There is no web app, no server and
nothing to deploy. Your profile never leaves your disk except in the prompt you send to a
model provider you chose.

## Install

### 0. Prerequisite: Python 3.12 or newer

```bash
python3 --version      # Linux, macOS
py -3.12 --version     # Windows (the `py` launcher installed with Python)
```

If that errors, install 3.12+ from [python.org](https://www.python.org/downloads/)
(Windows: tick *Add python.exe to PATH*) or with your package manager / Homebrew.

### 1. Get the code

```bash
git clone https://github.com/Mayamoho/career-pilot.git
cd career-pilot
```

### 2. Create an isolated environment and install

Do this in a terminal opened **inside the project folder**. Pick one option; both end the
same way.

**Option A — `uv`, one command, identical on all three systems.** Fastest, and it side
steps the "pip refused to install" problem entirely. Get uv from [astral.sh](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev && uv run careerpilot --help
```

**Option B — standard library only.** No extra tool to install.

Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Windows, in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Windows, in `cmd.exe`:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -e ".[dev]"
```

The activation step is the one people trip on. It only lasts for that terminal window, so
**repeat it in every new terminal** — and confirm it worked before installing anything:

```bash
which python pip    # Linux, macOS — must print paths inside .venv
where python pip    # Windows cmd.exe
(Get-Command python).Source   # Windows PowerShell
```

### 3. Check it installed

```bash
careerpilot --help
```

You should see `collect`, `jobs`, `score-job`, `tailor`, `traces`, `record-fixtures`. If
the command is not found, the environment is not active in this window — go back to step 2.

### 4. Tell it who you are

```bash
cp data/profile.example.yaml data/profile.yaml           # Linux, macOS
copy data\profile.example.yaml data\profile.yaml         # Windows cmd.exe
Copy-Item data\profile.example.yaml data\profile.yaml    # Windows PowerShell
```

Open `data/profile.yaml` and replace the placeholder `contact:` block with your real email,
phone, and **your actual address including the country**. Then look at the `atoms:` list:
each one is a single verifiable fact about your experience, with a date and a source. Add
your own, edit the ones that are not you, and delete the rest. Nothing in your CV can cite
anything that is not in that list.

That file is gitignored, so it never gets committed. Treat it the way you would treat a
password file.

### 5. Add a model key — only when you want to score or tailor

`collect`, `jobs` and `traces` need no key at all. Model calls do, and they are opt-in
twice over: a credential **and** an explicit switch, so no test or CI run can spend money
by accident.

The cross-platform way — one file, no shell syntax to remember:

```bash
cp .env.example .env          # Windows: copy .env.example .env
```

then edit `.env`:

```dotenv
GROQ_API_KEY=gsk_...
CAREERPILOT_LIVE=1
```

`console.groq.com` has a free tier; set a spend limit once you are in. `GEMINI_API_KEY`
works instead — Groq is tried first, Gemini is the fallback. A key you `export` in the
shell outranks the file.

Prefer environment variables? They work too:

```bash
export GROQ_API_KEY=gsk_... CAREERPILOT_LIVE=1        # Linux, macOS, Git Bash
$env:GROQ_API_KEY="gsk_..."; $env:CAREERPILOT_LIVE="1"  # Windows PowerShell
set GROQ_API_KEY=gsk_... & set CAREERPILOT_LIVE=1       # Windows cmd.exe (same window)
```

Without `CAREERPILOT_LIVE=1` the tool only ever replays recorded model answers — and no
answers are recorded yet, so a model-dependent command stops with a one-line explanation
instead of inventing text. That refusal is the design working: it would rather say nothing
than make something up.

## Use it, step by step

Run these from the project folder with the environment active — `source .venv/bin/activate`
on Linux/macOS, `.venv\Scripts\activate` on Windows. Steps 1 and 2 are free and need no
key. Steps 3 and 4 call a model, so they need the `.env` from step 5 above.

### 1. Fetch the listings

```bash
careerpilot collect
```

```
{
  "fetched": 4712,
  "inserted": 127,
  "kept": 850,
  "boards_ok": 21,
  "boards_dead": ["ashby:synthflow", "greenhouse:grafana", ...]
}
858 job(s) in store
```

`boards_dead` is the curated list correcting itself: those tokens 404'd, so they are
recorded rather than silently skipped.

### 2. Look at what you can work

```bash
careerpilot jobs
```

```
# workable from Bangladesh
bdtechjobs:41   Riseup Labs — IT Infrastructure & Network Lead    [onsite] Uttara, Dhaka       in Bangladesh
bdtechjobs:52   Brain Station 23 — Senior Cloud Engineer          [onsite] Dhaka, Bangladesh   in Bangladesh
bdtechjobs:14   Field Nation — Software Engineer                  [hybrid] Dhaka               in Bangladesh
bdtechjobs:13   Field Nation — Senior Software Engineer (Full stack) [hybrid] Dhaka            in Bangladesh
6 of 59 matching, 878 scanned
```

This is the default view: only postings reachable from the country in your profile, sorted
best-first. The number of rows is small on purpose — 878 stored, 59 you can work, and the
other 819 are shown to you by `--where all` with the reason attached.

**The uid is the whole first column.** Copy it exactly, prefix and colon included:

```
bdtechjobs:41   Riseup Labs — IT Infrastructure & Network Lead
^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
uid             title — this is NOT the uid
```

It means `<which board it came from>:<its id on that board>`. One string, no spaces, no
company name. `41`, `Riseup Labs`, and `bdtechjobs` alone are all wrong. Nothing else on
that line is something you can hand to a command.

Three other views:

```bash
careerpilot jobs --where bd              # only postings located in your country
careerpilot jobs --where all             # everything, each row saying why it is blocked
careerpilot jobs "cloud engineer"        # keyword search across the store
careerpilot jobs --limit 100             # more rows
careerpilot jobs --home Portugal         # what the same store offers from another country
```

### 3. Score one job — this is where money starts

```bash
careerpilot score-job bdtechjobs:52
```

Prints the fit score and its band, the location verdict, then one line per requirement
saying whether a profile atom met it and how much that requirement was worth, and finally
any unmet must-haves. The score and every number in it are computed in Python from your
atoms; the model only decides which requirement text matters and which skills it names.

Score before you tailor, and only for jobs you would actually apply to. This step makes
several model calls.

### 4. Generate the documents

```bash
careerpilot tailor bdtechjobs:52
```

Writes three files into `var/out/<uid>/` — the uid's colon becomes a dash in the folder
name — and prints each path:

| File | What it is |
|---|---|
| `cv.md` | the tailored CV |
| `cover-letter.md` | the cover letter |
| `audit.json` | every line that shipped, every line that was dropped, and the check that dropped it |

Open `audit.json` before you use the CV. If the bullet count that shipped is lower than you
expected, that is the gate working: a line it could not tie to one of your atoms is a line
that would have been a lie on paper.

### 5. Two answers you will meet, and what they mean

`no job with uid '41'` — you passed part of a uid. Re-read the first column of `jobs` and
copy all of it, colon included.

```
extract_requirements: no recorded answer for this prompt, and live model calls are
off (CAREERPILOT_LIVE is not set). To run it for real, copy .env.example to .env
in this folder, put your GROQ_API_KEY in it and leave CAREERPILOT_LIVE=1.
```

This is not a crash and not a bug. It means you skipped step 5 of Install: the tool can
collect and search without a model, but it will not write CV prose without one, and it
will not pretend it did. Do that step and re-run.

```
refusing: remote, restricted to a region outside Bangladesh — home is Bangladesh.
pass --ignore-location to tailor it anyway.
```

Also not a crash: the posting is geographically closed to you, and the refusal happens
*before* any model call so you are not billed for a CV nobody will hire you under. If it
is genuinely wrong — you have the visa, or the board's location field is bad — add
`--ignore-location`.


**6. See what the calls cost you.**

```bash
careerpilot traces
```

A sensible weekly loop: `collect`, skim `jobs`, `score-job` the two or three that look
right, `tailor` only the ones that come back `strong`. Step 2 is free; steps 3–5 cost money.

## Troubleshooting

| What you see | What it means and what to do |
|---|---|
| `error: externally-managed-environment` | Your `pip` is the system one, not the venv's — Debian and Ubuntu block system-wide installs under PEP 668. Run `which python` / `where python`. If it does not point inside `.venv`, activation did not happen in this window: repeat step 2, then install with `python -m pip install -e ".[dev]"`, never bare `pip`. |
| `pip: command not found` after activating | Same cause. Use `python -m pip ...`, which always targets the interpreter you just activated. |
| PowerShell: ``cannot be loaded because running scripts is disabled`` | Activation needs script permission for this user: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, reopen the terminal. Or use `cmd.exe` with `.venv\Scripts\activate.bat`, or Git Bash with `source .venv/bin/activate`. |
| `careerpilot: command not found` | The venv is not active in this terminal. Reactivate. On Windows it is `.venv\Scripts\careerpilot.exe` if you need the absolute path. |
| `no recorded answer for this prompt, and live model calls are off` | You have not turned on live model calls — see *Add a model key* in Install. Until you do, only `collect`, `jobs` and `traces` will run. This is one sentence rather than a crash because the fix is in your environment, not in the code. |
| `no job with uid '41'` | Part of a uid was passed. The uid is the whole first column of `jobs`, colon included: `bdtechjobs:41`. |
| `GROQ_API_KEY is not set` | The key is missing from both your shell and `.env`. Note `.env` must be named exactly that, and `CAREERPILOT_LIVE` must be the literal string `1`. |
| `no provider credential in the environment or in .env` from `record-fixtures` | Neither key is visible to the process. `.env` is read from the project folder, not from your home directory. |
| `jobs` returns nothing | You have not collected yet, or every stored posting is blocked from your country. `careerpilot jobs --where all` shows what was skipped and why. |
| `tailor` exits with code 2 | A blocked location, refused before any spend. Read the `refusing:` line; use `--ignore-location` only if it is genuinely wrong. |

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

The consequence is that fit and reach stay separate. `score-job` prints the number and its
band, then states the location verdict on its own line — `location [NOT WORKABLE] from
Bangladesh: flagged remote, but lists Washington, DC`.
`tailor` refuses a blocked posting **before making any model call**, since a tailored CV
for a job restricted to another country is not a near miss. `--ignore-location` overrides it.

Fit lands in one of three bands, and the band is a rule rather than an opinion: `strong`
requires a score of 70 **and** zero unmet must-haves.

## Tests

```bash
pytest -m "not eval and not live"   # behaviour: parsing, retries, scoring, the gate, rendering
pytest -m eval                      # thresholds, against scripted fakes
mypy src tests                      # strict, with the pydantic plugin
ruff check src tests
```

All four run with no credential in the environment, and they are free because nothing leaves
the machine: the suite drives scripted fakes for the answers and mocks the HTTP transport
wherever a provider client is actually exercised. The cassette mechanism
(`tests/fixtures/cassettes/`) is built and tested, but **nothing has been recorded into it
yet** — that needs a key and one deliberate `record-fixtures` run, so today a fresh clone
has no stored model answers at all. If you set `CAREERPILOT_LIVE=1` in `.env` for your own
use, the suite stays deaf to it — that is pinned by a test, because otherwise following this
README would quietly make every test on your machine bill you.

There is no CI workflow file in this repository yet, so those four commands are the whole
contract and they are yours to run before you commit. A `.github/workflows/ci.yml` that
runs them on every push, with no credential in the environment, is the intended next step.

Recording new cassettes is the one deliberate money step, because it is the only thing that
calls a real model outside of scoring and tailoring:

```bash
careerpilot record-fixtures <uid> <uid>
```

### What the eval tier enforces

The evals check numbers, not vibes (`tests/eval/`):

- **zero** fabricated lines ship under a judge that believes everything it is shown;
- every fabricated line fails for a *named code reason*, and all four code checks are exercised;
- truthful-claim acceptance stays above 85%, so a safe gate is not allowed to be useless;
- an invented citation is worth **exactly 0.0 points** on every labelled job.

They also pin the hole that code cannot close: a *real but irrelevant* atom id does
add weight. What bounds it is the seniority gate (pure date arithmetic) plus the
must-have rule, and an audit trail that names the atoms so a human can see the drift.

## What this is not

- Not a web app. It is a CLI; nothing here needs deploying.
- Not a scraper. Jobs come from Greenhouse, Ashby, Lever job-board APIs, and from
  bdtechjobs' public JSON API. LinkedIn is not touched, and the curated board list in
  `src/careerpilot/ingest/boards.py` is why it works at all — every public feed is
  per-company, so breadth comes from maintaining tokens, not from writing fetchers.
- Not a ghostwriter. It cannot tell you you did something you did not do, and a line it
  cannot attribute is a line that does not appear in your CV.
- Not finished advice. It ranks what those five boards happen to list. The strongest
  signal for a candidate outside the US is still a person, and a tool that hides 800
  postings you cannot work from is not doing you a favour by showing them.

See [docs/PLAN.md](docs/PLAN.md) for the brief this was built from.
