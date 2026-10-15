# CareerPilot

**A job-search agent that will not let you put something on your CV that you did not do.**

It pulls live postings from job boards that are actually public — the Greenhouse, Ashby and
Lever per-company APIs, bdjobs' and bdtechjobs' own JSON feeds, and Jobicy's remote-board API
— scores how well your real experience fits each one, tells you whether a machine reading your
CV can parse it, and writes the CV and cover letter for the job you pick. Every sentence it
produces has to point at a specific fact in your profile file, and the check that enforces
that is Python comparing ids, not a prompt asking nicely.

**Built for someone starting out.** "Fresher" is not a polite word for unemployable, it is a
different problem: intern and junior titles get filtered out by tools written for senior
candidates, three hundred postings cannot all be worth your one free-tier day, and nobody
tells you whether your CV would survive the parser that stands between you and a person.
`start-here`, `ats`, `rank` and `closing` are those four problems. None of them call a model
more than they have to.

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
| Score a CV for a parser | code | Seven weighted checks over the text. No model, no network. |
| Say which keywords you may add | code | Only the ones your fact base can already prove. |
| Pick the deadline to chase | code | Date arithmetic on the posting's own deadline. |

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

You should see eleven commands: `collect`, `jobs`, `score-job`, `tailor`, `traces`,
`record-fixtures`, and the four aimed at an early career — `start-here`, `ats`, `rank`,
`closing`. If
the command is not found, the environment is not active in this window — go back to step 2.

### 4. Tell it who you are

Two ways in, depending on what you have.

**You have nothing written down yet** — no CV, maybe two lab projects and an attachment.
Answer a questionnaire and let it build the file:

```bash
careerpilot start-here
```

It asks for your name, contact, the technologies you have actually written code with, then
projects, internship/industrial training, education with GPA, certificates and hackathons —
each answer becomes one verifiable fact with a stable id, which is the part of the file
format that stops a beginner writing YAML by hand. Nothing is invented: every line in the
result is a line you typed. It validates before writing, and refuses to overwrite an
existing profile unless you pass `--force`.

**You already have a profile or want to write it yourself:**

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

Six commands need no key at all: `collect`, `jobs`, `closing`, `start-here`, `traces`, and
`ats` — a posting is a fact and a parseability score is arithmetic, so neither is asked of a
model. `rank` needs one only for postings it has not scored before. Model calls are opt-in
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
on Linux/macOS, `.venv\Scripts\activate` on Windows. Steps 1–4 cost nothing and need no key;
step 5 costs a bounded number of calls; steps 6 and 7 call a model, so they need the `.env`
from step 5 of Install.

**If you have nothing but coursework and a deadline you do not know about**, the whole tool in
order costs you one command each:

```bash
careerpilot start-here          # 1. answer questions, get a fact base — no YAML by hand
careerpilot collect             # 2. every public board, zero model calls
careerpilot jobs --where bd python     # 3. intern/junior rows are NOT filtered out
careerpilot closing --days 21     # 4. what closes first — free, and bdjobs publishes dates
careerpilot ats                   # 5. can a machine read what you have — free
careerpilot rank --max-new 5      # 6. spend the day's model budget on 5 jobs, not 300
careerpilot tailor <uid>          # 7. cv-ats.html, cv-oxford.html, cv.txt, audit.json
```

Steps 2–5 cost zero model calls: a posting is a fact and parseability is arithmetic. Only
`rank` and `tailor` ask a model anything, and what they ask is about *language* — which words
the posting is really requiring, and how to phrase something you already did. Whether you are
good enough is never put to a model; it is the score, and the score is Python.

### 1. Fetch the listings

```bash
careerpilot collect
```

```
{
  "fetched": 5088,
  "inserted": 167,
  "kept": 1049,
  "boards_ok": 32,
  "boards_dead": ["ashby:synthflow", "greenhouse:grafana", ...]
}
1045 job(s) in store
```

`boards_dead` is the curated list correcting itself: those tokens 404'd, so they are
recorded rather than silently skipped. That run stored, per source:

| Source | Stored | What that feed is for |
|---|---|---|
| ashby | 490 | US and European tech companies |
| greenhouse | 427 | the same, on a second ATS |
| bdjobs | 51 | the Bangladeshi market, with deadlines |
| bdtechjobs | 44 | Bangladeshi tech, older postings |
| lever | 17 | a third ATS |
| jobicy | 16 | remote, with an honest seniority label |

Nothing here scraped a page, and nothing here asked a model: a posting is a fact, so
`collect` costs zero AI calls. See *Where the jobs come from* for what each feed cannot do.

### 2. Look at what you can work

```bash
careerpilot jobs
```

```
# workable from Bangladesh
bdjobs:1545609     Pridesys IT Limited — Software Quality Assurance Engineer    [remote-eligible] Anywhere in Bangladesh   in Bangladesh
bdjobs:1544678     Mynztrip.com — Laravel Developer                             [remote-eligible] Anywhere in Bangladesh   in Bangladesh
bdjobs:1542177     Marive — Python Automation Intern                            [remote-eligible] Anywhere in Bangladesh   in Bangladesh
bdtechjobs:41      Riseup Labs — IT Infrastructure & Network Lead               [onsite] Uttara, Dhaka                      in Bangladesh
4 of 116 matching, 1045 scanned
```

This is the default view: only postings reachable from the country in your profile, sorted
best-first. The number of rows is small on purpose — 1045 stored, 116 you can work from
Bangladesh, 86 of those physically located in the country, and the rest are shown to you by
`--where all` with the reason attached.

The third row is the reason the role filter was rewritten. **"Python Automation Intern" is a
live, reachable, fresher-only posting**, and an earlier version of this tool deleted it before
anything cost a model call, because `intern` was on the excluded-titles list. A tool that
filters out the jobs you are allowed to apply to is not a tool.

**The uid is the whole first column.** Copy it exactly, prefix and colon included:

```
bdtechjobs:41   Riseup Labs — IT Infrastructure & Network Lead
^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
uid             title — this is NOT the uid
```

It means `<which board it came from>:<its id on that board>`. One string, no spaces, no
company name. `41`, `Riseup Labs`, and `bdtechjobs` alone are all wrong. Nothing else on
that line is something you can hand to a command.

The other views:

```bash
careerpilot jobs --where bd              # only postings located in your country
careerpilot jobs --where all             # everything, each row saying why it is blocked
careerpilot jobs "cloud engineer"        # keyword search across the store
careerpilot jobs --limit 100             # more rows
careerpilot jobs --home Portugal         # what the same store offers from another country
careerpilot jobs --where open --source jobicy   # one feed only
```

`--source jobicy` is the interesting one for a candidate in Dhaka. Jobicy labels a job's
reachability in its own field, and the only value that means *you* is `geo=anywhere`:

```
jobicy:155391   Supabase — Site Reliability Engineer   [remote] Anywhere   remote, the posting itself says it is open worldwide
```

An older build read the literal word "Anywhere" as an unnamed city and called those rows
unreachable — which hid exactly the jobs this is for. Now a location that names no country
and says anywhere/worldwide/global is read as a restriction lifted, not a place unknown.

### 3. What is about to close

```bash
careerpilot closing --days 21
```

```
# 30 closing within 21 day(s); 5 already expired and dropped
2026-10-17    1d  bdjobs:1535095  [unspecified]  Avian BPO & IT — Software Engineer (Android/iOS)
2026-10-19    3d  bdjobs:1536736  [unspecified]  EQMS Consulting Limited — Full Stack Developer (PHP, Laravel, ReactJS, SQL and PostgreSQL)
2026-10-19    3d  bdjobs:1535356  [unspecified]  KSF Luxury Homes Limited — Software Engineer (Web Developer)
2026-10-23    7d  bdjobs:1544040  [unspecified]  Synesis IT — Quality Assurance Executive
```

Free, no key. This matters more in Bangladesh than anywhere else on the list: bdjobs
publishes a deadline, and a posting there is dead the morning after it. Five stored rows
were already expired at that run, and nothing in the feed says so unless you ask. Expired
rows are dropped, not sorted to the bottom — a CV for a closed vacancy costs money and gets
nothing. `tailor` refuses one for the same reason, before any model call.

### 4. Can a machine actually read your CV

```bash
careerpilot ats
careerpilot ats bdjobs:1542177        # the same score against one real posting
careerpilot ats --cv my-cv.txt --jd posted-description.txt
```

Computed in code — no model, no network, no key. Seven weighted checks, so the number is a
rule you can read rather than an opinion:

| Check | Points | What it measures |
|---|---|---|
| `keyword_coverage` | 30 | the posting's own terms present in your CV |
| `sections` | 20 | Summary / Skills / Experience-or-Projects / Education headings |
| `contact_block` | 15 | a dialable phone, an email, a location, a repo or profile URL |
| `proof_density` | 15 | bullets that contain a number — a year does not count |
| `filler` | 10 | "hard working", "team player", "quick learner" |
| `length` | 5 | 350–800 words |
| `parse_traps` | 5 | photo lines, markdown tables, multi-column markers |

A real run against a live Bd internship posting, on the example fact base:

```
ATS readiness 93.9/100 — ready  (983 words)
scored against: Felna Tech (FelnaTech.com) — Full-Stack Developer Intern
 contact_block      15.0/15.0  all four present
 sections           20.0/20.0  found: Summary, Skills, Experience or Projects, Education
 keyword_coverage   25.0/30.0  10/12 of the posting's terms appear in the CV
 proof_density      15.0/15.0  21/42 bullets contain a number
 filler             10.0/10.0  no filler phrases
 length              3.9/ 5.0  983 words — over one page
    → cut to the last two years: one page is read, two are skimmed
 parse_traps         5.0/ 5.0  no known parse traps
posting terms your CV already states: css, database, git, github, javascript, mongodb, Next.js, node, react, rest
NOT IN YOUR FACT BASE — DO NOT ADD: html, mysql
  add the experience first, then this line: `careerpilot start-here` helps.
```

The `proof_density` denominator is the number to argue with, and it is the reason this is a
rule rather than a vibe: 42 bullet lines, 21 with a figure in them. A date does not qualify —
`_DATE` is stripped before the digit search, because `(Brain Station 23 | 2025)` would else
count as quantified proof and a CV could score itself by listing years.

**That last line is the whole point of the command.** Every other ATS grader on the internet
will tell you to add "Kubernetes" because the job description mentions it. This one computes
the missing terms twice — once from the posting, once from what your fact base can actually
prove — and prints the two lists separately. A keyword you cannot cite an atom for is printed
under *do not add*, and the eval enforces that at a rate of exactly zero, not "usually zero".
If your CV is short on a term, the fix is the experience, not the document.

With no `--cv` it scores the CV rendered from your fact base, so it works before you have
tailored anything. `--write-cv` drops that plain text next to the score; plain text is the
one format every parser survives and the only one you can paste into a portal box.

### 5. Which of these are worth your day

```bash
careerpilot rank --limit 8 --max-new 5
```

```
# workable from Bangladesh — 5 fresh score(s), 1 reused, 110 over the --max-new budget
bdjobs:1545472    77.2 partial   [unspecified] 14d left   Felna Tech (FelnaTech.com) — Full-Stack Developer Intern
bdjobs:1545900    56.5 partial   [unspecified]  9d left   Goethe-Institut and German Cultural Center — Information Technology - Administrator
bdtechjobs:5      46.2 partial   [onsite] no deadline     Bdtask Limited — Junior Software Engineer (Python/Django)
jobicy:155389     26.9 weak      [remote] no deadline     Supabase — Platform Engineer, Managed Postgres
```

This is the only new command that can spend money, and `--max-new` is why it exists: Groq's
free tier throttles *tokens per minute*, and an uncapped batch rank is exactly how a live run
died halfway through a cover letter. Scores are stored, so the same command a minute later
costs nothing:

```
# workable from Bangladesh — 0 fresh score(s), 6 reused, 110 over the --max-new budget
```

The header is honest about the rest: 110 reachable postings were not scored because they were
over the budget. They are not lost — tomorrow's run picks them up, and a job scored once is
never paid for twice. Sorted workable first, then fit, then the deadline that is nearest.

### 6. Score one job in detail

```bash
careerpilot score-job bdtechjobs:52
```

Prints the fit score and its band, the location verdict, then one line per requirement
saying whether a profile atom met it and how much that requirement was worth, and finally
any unmet must-haves. The score and every number in it are computed in Python from your
atoms; the model only decides which requirement text matters and which skills it names.

`rank` gives you the number for many jobs; this gives you the *reasons* for one. Use it on the
two or three you would actually take, not on the whole store.

### 7. Generate the documents

```bash
careerpilot tailor bdjobs:1545472
```

```
fit 74.6/100 (partial)
location: in Bangladesh
deadline: 2026-10-30, 14 day(s) left
bullets shipped 4/5
  dropped: not_entailed: judge confidence 0.98; unsupported span: Managed code versioning
letter paragraphs shipped 0/3
wrote var/out/bdjobs-1545472/cv.md
wrote var/out/bdjobs-1545472/cover-letter.md
wrote var/out/bdjobs-1545472/audit.json
wrote var/out/bdjobs-1545472/cv-ats.html
wrote var/out/bdjobs-1545472/cv-oxford.html
wrote var/out/bdjobs-1545472/cv.txt
wrote var/out/bdjobs-1545472/cover-letter.html
```

The uid's colon becomes a dash in the folder name. `--style` picks the print forms: `both`
(the default, both HTML files plus the text), `ats`, `oxford`, or `md` for markdown only.

| File | What it is |
|---|---|
| `cv.md` | the tailored CV, as the pipeline's own markdown |
| `cv-ats.html` | single column, Arial, ALL-CAPS headings, no tables or icons — the shape a parser survives |
| `cv-oxford.html` | the Oxford layout: centred name, serif headings on a rule, dates right-aligned in CSS grid |
| `cv.txt` | the same content with all markup stripped — what you paste into a portal box |
| `cover-letter.md` / `.html` | the letter |
| `audit.json` | every line that shipped, every line that was dropped, and the check that dropped it |

Both templates render the **gated** output only — the same shipped bullets, copied verbatim.
The model never touches layout, so it cannot smuggle a sentence in through a template.

Open `audit.json` before you use the CV. That run shipped 4 of 5 bullets and **0 of 3 letter
paragraphs**: a bullet claiming "Managed code versioning" was dropped because the atoms it
cited do not say that, and the letter's paragraphs failed for the same reason. A CV that comes
out thinner than you expected is the system working, not failing. The dates and the words are
yours; the ordering is the tool's.

### 8. Print it to a PDF

There is no PDF library in the dependencies, on purpose — a CV you cannot edit is a CV you
cannot correct, and every desktop browser prints better than any Python PDF writer.

1. Open `var/out/bdjobs-1545472/cv-ats.html` in a browser (double-click, or
   `xdg-open` / `open` / `start`).
2. Ctrl+P (Cmd+P on macOS).
3. Destination **Save as PDF**, Paper **A4**, Margins **Default** (the template already sets
   its own 18 mm), Scale **100**, and turn **Background graphics** off for the ATS one.
4. Save as `Your-Name-CV.pdf` — recruiters read the filename, and `cv-ats final(2).pdf`
   does not read like an engineer.

Then check the thing no template author checks: **select all in the PDF, copy, paste into a
plain text editor.** If it pastes as readable lines, the ATS at the other end can read it too.
If it pastes as scrambled columns, you printed the wrong template — use `cv-ats.html`, or just
paste `cv.txt` into the portal instead.

### 9. Answers you will meet, and what they mean

`no job with uid '41'` — you passed part of a uid. Re-read the first column of `jobs` and
copy all of it, colon included.

```
extract_requirements: no recorded answer for this prompt, and live model calls are
off (CAREERPILOT_LIVE is not set). To run it for real, copy .env.example to .env
in this folder, put your GROQ_API_KEY in it and leave CAREERPILOT_LIVE=1.
```

This is not a crash and not a bug. It means you skipped step 5 of Install: the tool can
collect, search, score a CV for a parser and list deadlines without a model, but it will not
write CV prose without one, and it will not pretend it did. Do that step and re-run.

```
refusing: remote, restricted to a region outside Bangladesh — home is Bangladesh.
pass --ignore-location to tailor it anyway.
```

Also not a crash: the posting is geographically closed to you, and the refusal happens
*before* any model call so you are not billed for a CV nobody will hire you under. If it
is genuinely wrong — you have the visa, or the board's location field is bad — add
`--ignore-location`.

```
refusing: deadline 2026-10-08 has passed.
pass --ignore-deadline to tailor it anyway.
```

Same logic, and on Bangladeshi boards it is the more common one: a vacancy there closes on a
date and the feed keeps serving the row. `--ignore-deadline` exists for the case where the
employer quietly reopened it, which you would know and the tool would not.

### 10. See what the calls cost you

```bash
careerpilot traces
```

```json
{
  "total_calls": 146,
  "total_failures": 54,
  "total_cost_usd": 0.052304,
  "fallback_invocations": 0
}
```

Real numbers from building this: a hundred and forty-six calls across every live run cost
five cents. Two things worth reading in that: `total_failures` is a third of all calls — the
free tier throws schema errors, and the retry path is what you are paying five cents for
instead of a broken CV — and `fallback_invocations` is 0, because the Gemini key was never
configured, so the fallback has been tested but never exercised for real.

A sensible weekly loop: `collect`, skim `jobs`, `ats` your CV against the two or three that
look right, `rank --max-new 5` to spend the day's budget where it counts, `tailor` only the
ones that come back `strong`. Steps 1–4 and 8–10 are free.

## Where the jobs come from

Every feed here is one the board publishes for its own frontend, found by reading what the
page actually calls. LinkedIn is not touched: it has no public job API, and the terms of
service are the least of the reasons — a scraped profile page changes shape every month and
the tool would silently stop working rather than tell you.

The honest version of this table includes what each source **cannot** do, because that is what
determines whether the answers you get are worth anything.

| Source | How | What it cannot do |
|---|---|---|
| Greenhouse | `https://boards-api.greenhouse.io/v1/boards/<token>/jobs` per company | No cross-company search. Breadth is a maintained token list; a dead token 404s and lands in `boards_dead`. |
| Lever | `https://api.lever.co/v0/postings/<token>?mode=json` | Same per-company shape. Many tokens are empty rather than absent, which looks like success. |
| Ashby | `https://api.ashbyhq.com/posting-api/job-board/<token>` | Sets `isRemote: true` on jobs whose only stated location is an office. 143 rows read `San Francisco` while claiming remote in one collect, which is why the flag is treated as a claim and the location text as the fact. |
| bdtechjobs | its own JSON feed, `applicationDeadline` included | Small tech-only board, and a lot of rows carry no prose to score against. |
| bdjobs | the two keyless JSON endpoints its Angular app calls | **`category`, `location`, `jobType`, `sort` and `version` are accepted and silently ignored** — they return HTTP 200 with unfiltered rows. Only `keyword`, `pg` and `rpp` are honoured, so all filtering happens here, in code. Its API misspells the employer name as `CompnayName`; read it literally. `robots.txt` allows everything, and the run is still capped at 2 list pages and 12 detail fetches with a 0.6 s sleep, because a free endpoint is not an invitation. |
| Jobicy | `https://jobicy.com/api/v2/remote-jobs` | Enforces a parameter whitelist: an unknown param returns HTTP 400 rather than being ignored — the opposite failure mode to bdjobs, so both are handled. Only `geo=anywhere` is genuinely reachable from Dhaka; `geo=apac` includes countries that are not yours. Its `friendlyNotice` asks for credit and that the apply button hit the original URL — this tool prints the URL. |

Rejected with the evidence, so nobody re-trims them: **Remotive** returns a fixed 14-row page
and ignores `search`, `category` and `country`, and its notice forbids redistribution;
**RemoteOK** left 39 of 99 rows with no location at all, same redistribution clause;
**WeWorkRemotely** is RSS-only; **SmartRecruiters** answers HTTP 200 with `content: []` for
every token tried, so a dead board is indistinguishable from a live one; **Recruitee** and
**Workable** 404 on every guessed account name.

The title filter is a source decision too, and it used to delete the target user. It no
longer excludes `intern`, and it keeps `trainee`, `junior`, `graduate`, `entry level` and the
stack names a Bangladeshi student actually has — `python`, `django`, `laravel`, `react`,
`node`, `java`, `qa`. The two words that attract non-tech roles (`intern`, `trainee`) only
count when a tech term is in the same title, so "Python Automation Intern" is kept and
"Management Trainee (Bank)" is not.

## Troubleshooting

| What you see | What it means and what to do |
|---|---|
| `error: externally-managed-environment` | Your `pip` is the system one, not the venv's — Debian and Ubuntu block system-wide installs under PEP 668. Run `which python` / `where python`. If it does not point inside `.venv`, activation did not happen in this window: repeat step 2, then install with `python -m pip install -e ".[dev]"`, never bare `pip`. |
| `pip: command not found` after activating | Same cause. Use `python -m pip ...`, which always targets the interpreter you just activated. |
| PowerShell: ``cannot be loaded because running scripts is disabled`` | Activation needs script permission for this user: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, reopen the terminal. Or use `cmd.exe` with `.venv\Scripts\activate.bat`, or Git Bash with `source .venv/bin/activate`. |
| `careerpilot: command not found` | The venv is not active in this terminal. Reactivate. On Windows it is `.venv\Scripts\careerpilot.exe` if you need the absolute path. |
| `no recorded answer for this prompt, and live model calls are off` | You have not turned on live model calls — see *Add a model key* in Install. Until you do, `collect`, `jobs`, `closing`, `ats`, `start-here` and `traces` all work; `rank` works for postings it has scored before. This is one sentence rather than a crash because the fix is in your environment, not in the code. |
| `no job with uid '41'` | Part of a uid was passed. The uid is the whole first column of `jobs`, colon included: `bdtechjobs:41`. |
| `GROQ_API_KEY is not set` | The key is missing from both your shell and `.env`. Note `.env` must be named exactly that, and `CAREERPILOT_LIVE` must be the literal string `1`. |
| `no provider credential in the environment or in .env` from `record-fixtures` | Neither key is visible to the process. `.env` is read from the project folder, not from your home directory. |
| `jobs` returns nothing | You have not collected yet, or every stored posting is blocked from your country. `careerpilot jobs --where all` shows what was skipped and why. |
| `jobs --where bd` returns almost nothing | Your `contact.location` has no country in it, so nothing matches as "in Bangladesh". Put the country in the address — this is the line the whole location verdict is read from. |
| `tailor` exits with code 2 | A blocked location or a passed deadline, refused before any spend. Read the `refusing:` line; use `--ignore-location` / `--ignore-deadline` only when the board's own field is wrong. |
| `rank` prints `N over the --max-new budget` | Not an error: those postings were reachable but unscored because you capped the run. Re-run tomorrow — already-scored ones are reused for free, and the queue continues where it stopped. |
| `ats` puts a keyword you know under `NOT IN YOUR FACT BASE` | The checker is right about the file, not about you: the term is not in any atom. Add the atom (if it is true) with `start-here --out` on a fresh file or by editing `data/profile.yaml`, then re-run. If the term is a compound like `Git/GitHub`, it is now split into its halves. |
| `start-here` says the file exists | It will not overwrite a fact base you already have. Either point `--out` somewhere else or pass `--force` deliberately. |

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
- an invented citation is worth **exactly 0.0 points** on every labelled job;
- `ats` suggests a keyword the fact base cannot prove at a rate of **exactly zero**, over eight
  real Bd and remote postings. The oracle there is plain set membership written by hand
  against a fixed list of proven terms — deliberately *not* `has_term` — so the checker cannot
  pass by agreeing with itself.

They also pin the hole that code cannot close: a *real but irrelevant* atom id does
add weight. What bounds it is the seniority gate (pure date arithmetic) plus the
must-have rule, and an audit trail that names the atoms so a human can see the drift.

## What this is not

- Not a web app. It is a CLI; nothing here needs deploying.
- Not a scraper. Jobs come from the Greenhouse, Ashby and Lever per-company APIs, the bdjobs
  and bdtechjobs JSON feeds bdjobs' own frontend uses, and Jobicy's remote-board API — listed
  with what each cannot do in *Where the jobs come from*. LinkedIn is not touched, and the
  curated board list in `src/careerpilot/ingest/boards.py` is why it works at all: every
  public feed is per-company, so breadth comes from maintaining tokens, not from writing
  fetchers.
- Not a ghostwriter. It cannot tell you you did something you did not do, and a line it
  cannot attribute is a line that does not appear in your CV. The same rule runs backwards in
  `ats`: a keyword your fact base cannot prove is printed under *do not add*.
- Not an ATS. `ats` measures whether a parser could read your document and which of the
  posting's terms you can honestly claim. It does not know the vendor on the other end, and
  no scorer can — but the seven checks are printed with their weights, so you can disagree
  with a specific line instead of with a black box.
- Not finished advice. It ranks what those six feeds happen to list. The strongest signal for
  a candidate outside the US is still a person, and a tool that hides 950 postings you cannot
  work from is not doing you a favour by showing them.

See [docs/PLAN.md](docs/PLAN.md) for the brief this was built from.
