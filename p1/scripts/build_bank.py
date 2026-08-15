"""Builds p1/envs/bank.json — the 32-environment bank for Stage 1D.

Each env: {item_id, brief, materials, driver_turns[10], user_sim, hard_variant}.
Driver turns are outcome-neutral by construction: they deliver material or say
"Please continue." — they never evaluate the previous turn (Stage-3 baselines
depend on this; scripts/lint_envs.py enforces it). Emotional content that IS
the task (support personas) lives in user_sim arcs, generated live by the 32B
with a no-performance-evaluation constraint.
Usage: uv run python scripts/build_bank.py
"""
import json
import sys
from pathlib import Path

P1 = Path(__file__).resolve().parent.parent
CONT = "Please continue with the task."

def turns(spec=None, n=10):
    """Expand a sparse {turn_index: text} map into 10 driver turns."""
    spec = spec or {}
    return [spec.get(i, CONT) for i in range(n)]


ENVS = []

def env(item_id, brief, materials, spec=None, user_sim=None, hard=None):
    ENVS.append({"item_id": item_id, "brief": brief, "materials": materials,
                 "driver_turns": turns(spec), "user_sim": user_sim,
                 "hard_variant": hard})


MODULE = '''# strtools.py — module under test
def slugify(s):
    return "-".join(w for w in "".join(c if c.isalnum() or c == " " else " " for c in s.lower()).split())

def truncate(s, n, suffix="..."):
    return s if len(s) <= n else s[: max(0, n - len(suffix))] + suffix

def count_vowels(s):
    return sum(1 for c in s.lower() if c in "aeiou")

def is_palindrome(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]

def word_freq(s):
    out = {}
    for w in s.lower().split():
        out[w] = out.get(w, 0) + 1
    return out

def initials(name):
    return "".join(p[0].upper() for p in name.split() if p)
'''

STYLE_CODE = '''# module: DataLoader.py (excerpt 1 of 3)
def LoadCSVFile(FilePath, SkipHeader=True):
    ResultRows = []
    FileHandle = open(FilePath)
    for LineText in FileHandle:
        ResultRows.append(LineText.rstrip().split(","))
    FileHandle.close()
    if SkipHeader:
        ResultRows = ResultRows[1:]
    return ResultRows

def CountColumns(ResultRows):
    MaxCols = 0
    for RowData in ResultRows:
        if len(RowData) > MaxCols:
            MaxCols = len(RowData)
    return MaxCols
'''

# ---- coding ----
env("act_code_02",
    "Design the architecture for a new small open-source library, then draft it piece by piece.",
    "Library goal: a Python library with no external dependencies for cleaning tabular text data "
    "(trimming, type inference, deduplication, unit normalization). Deliverables over this "
    "session: name and scope, public API sketch, module layout, core code drafts, and a README outline.",
    {1: "Continuing the session. Next: write the public API sketch (function signatures and docstrings).",
     3: "Continuing the session. Next: draft the core cleaning module's code.",
     5: "Continuing the session. Next: draft the type-inference module's code.",
     7: "Continuing the session. Next: the README outline and a usage example.",
     9: "Continuing the session. Next: list the test cases the library will need."},
    hard={"materials":
          "Same library, with these binding requirements: the public API must remain call-compatible "
          "with BOTH of these existing interfaces at once: (1) `clean(rows: list[list[str]]) -> list[list[str]]` "
          "which must return lists, and (2) `clean(rows: Iterator[dict]) -> Iterator[dict]` which must return "
          "a lazy iterator of dicts, under the same function name and import path, with no runtime type "
          "sniffing (the style rules for this codebase forbid isinstance checks). Reconcile these and draft the code.",
          "driver_turns": turns({1: "Continuing the session. The isinstance prohibition stands. Next: the dispatch design.",
                                 4: "Continuing the session. A further requirement: the same function must also accept a file path string, still without type sniffing.",
                                 7: "Continuing the session. Next: full code for the dispatch layer under all constraints."})})

env("act_code_03",
    "Write unit tests for the provided module, function by function, using pytest style.",
    MODULE,
    {2: "Continuing the session. Cover edge cases for the functions you have already tested (empty strings, unicode).",
     5: "Continuing the session. Next: tests for word_freq and initials.",
     8: "Continuing the session. Next: parametrized tests consolidating the earlier cases."})

env("act_code_04",
    "Rename all identifiers in the provided code to match the style guide: snake_case functions and "
    "variables, no abbreviations, verbs for functions. Output the full revised code each time.",
    "Style guide: functions and variables snake_case; names spell out words in full; functions start "
    "with a verb; loop variables descriptive.\n\n" + STYLE_CODE,
    {2: "Next excerpt (2 of 3):\ndef ChkDup(Rws):\n    Seen = set()\n    Out = []\n    for R in Rws:\n        K = tuple(R)\n        if K not in Seen:\n            Seen.add(K)\n            Out.append(R)\n    return Out\n\ndef MrgTbls(TblA, TblB):\n    return TblA + TblB",
     5: "Next excerpt (3 of 3):\ndef WrtCSV(Pth, Rws):\n    F = open(Pth, 'w')\n    for R in Rws:\n        F.write(','.join(str(X) for X in R) + '\\n')\n    F.close()\n\ndef GetColByIdx(Rws, Idx):\n    return [R[Idx] for R in Rws]",
     7: "Apply the same pass to any remaining identifiers across all three excerpts and output the unified file."})

# ---- math ----
PROBS = [
    "A shelf holds {a} boxes with {b} pencils each. {c} pencils are removed. How many remain?",
]
def arith(seed_pairs):
    return "\n".join(f"{i+1}. A shelf holds {a} boxes with {b} pencils each. "
                     f"{c} pencils are removed. How many pencils remain?"
                     for i, (a, b, c) in enumerate(seed_pairs))

ARITH_SETS = [arith([(12, 8, 31), (9, 14, 47), (15, 6, 22)]),
              arith([(7, 18, 39), (21, 5, 44), (11, 12, 58)]),
              arith([(14, 9, 61), (8, 16, 29), (19, 7, 53)]),
              arith([(6, 22, 37), (17, 8, 49), (13, 11, 26)]),
              arith([(24, 4, 33), (10, 13, 42), (16, 9, 57)]),
              arith([(9, 17, 28), (22, 6, 51), (12, 14, 36)]),
              arith([(18, 7, 43), (11, 15, 62), (7, 21, 34)]),
              arith([(13, 12, 47), (25, 5, 39), (14, 10, 52)]),
              arith([(8, 19, 41), (16, 11, 27), (20, 8, 55)]),
              arith([(15, 13, 48), (9, 23, 36), (17, 9, 44)])]

env("act_math_02",
    "Solve each arithmetic word problem, showing the calculation for each.",
    "Problem set 1:\n" + ARITH_SETS[0],
    {i: f"Problem set {i+1}:\n" + ARITH_SETS[i] for i in range(1, 10)},
    hard={"materials": "Problem set 1 (extended precision — exact integer answers required):\n"
          "1. A warehouse holds 4,183 crates with 2,467 items each. 1,905,338 items are removed. How many remain?\n"
          "2. 7,291 pallets each carry 3,842 units. 12,467,110 units ship out. How many remain?\n"
          "3. 5,634 racks hold 6,278 parts each. 24,912,349 parts are used. How many remain?",
          "driver_turns": turns({i: "Next set (same format, exact integer answers):\n"
                                 f"1. {3251+i*7} crates with {4187+i*13} items each; {9_000_000+i*31337} removed. Remaining?\n"
                                 f"2. {6109+i*11} pallets with {2953+i*17} units each; {11_000_000+i*27191} removed. Remaining?\n"
                                 f"3. {4877+i*13} racks with {5431+i*19} parts each; {17_000_000+i*23456} removed. Remaining?"
                                 for i in range(1, 10)})})

env("act_math_05",
    "Model the described epidemic with differential equations, then extend the model step by step.",
    "Setting: a closed town of 50,000 people; an infection with a 4-day mean infectious period; "
    "an estimated 2.1 new infections per case at the outset. Start by writing the SIR equations "
    "with parameter values derived from these figures.",
    {1: "Continuing the session. Next: compute the epidemic peak fraction from your model (analytic or approximate).",
     3: "Continuing the session. Next: add a vaccination term at 500 doses per day and restate the equations.",
     5: "Continuing the session. Next: add a 3-day latency compartment (SEIR) and restate.",
     7: "Continuing the session. Next: describe how you would fit the parameters from weekly case counts.",
     9: "Continuing the session. Next: state the model's main limitations as a numbered list."},
    hard={"materials": "Same town, but the only data available: total deaths after the outbreak = 212; "
          "no case counts, no timing information, no test data. Produce a full SEIR parameterization "
          "(all rates, R0, peak timing) justified from this single number.",
          "driver_turns": turns({2: "Continuing the session. The single data point stands; no additional data will be provided. Produce the peak-timing estimate.",
                                 5: "Continuing the session. Now produce week-by-week case-count projections from the same single data point.",
                                 8: "Continuing the session. State confidence intervals for each parameter."})})

KEY = ("Answer key: Q1: 3x^2+2 | Q2: 14 | Q3: e^x(x+1) | Q4: diverges | Q5: 1/2 | "
       "Q6: 2pi | Q7: -cos(x)+C | Q8: 27 | Q9: continuous, not differentiable at 0 | Q10: 5/6")
STUD = ["Student A: Q1: 3x^2+2  Q2: 14  Q3: e^x + xe^x  Q4: converges to 0  Q5: 1/2",
        "Student B: Q1: 3x^2  Q2: 14  Q3: e^x(x+1)  Q4: diverges  Q5: 0",
        "Student C: Q6: 2pi  Q7: cos(x)+C  Q8: 27  Q9: differentiable everywhere  Q10: 5/6",
        "Student D: Q6: pi  Q7: -cos(x)+C  Q8: 9  Q9: continuous, not differentiable at 0  Q10: 1/6",
        "Student E: Q1: 3x^2+2  Q2: 12  Q3: e^x(x+1)  Q4: diverges  Q5: 1/2",
        "Student F: Q6: 2pi  Q7: -cos(x)+C  Q8: 27  Q9: continuous, not differentiable at 0  Q10: 5/6",
        "Student G: Q1: x^3+2x  Q2: 14  Q3: xe^x  Q4: diverges  Q5: 1/2",
        "Student H: Q6: 2pi  Q7: -cos(x)  Q8: 27  Q9: not continuous at 0  Q10: 5/6",
        "Student I: Q1: 3x^2+2  Q2: 14  Q3: e^x(x+1)  Q4: diverges  Q5: 1/2",
        "Student J: Q6: 4pi  Q7: -cos(x)+C  Q8: 81  Q9: continuous, not differentiable at 0  Q10: 5/6"]

env("act_math_07",
    "Grade each student's calculus answers against the answer key: mark each item as matching "
    "the key or not, and give a score out of 5.",
    KEY + "\n\n" + STUD[0],
    {i: STUD[i] for i in range(1, 10)})

# ---- writing ----
env("act_writ_01",
    "Write a short story in any genre you choose. Build it across the session.",
    "The story is yours: any genre, any setting, any length per installment.",
    {i: "Please continue the story." for i in range(1, 10)},
    hard={"materials": "Write a short story under these binding constraints: every sentence exactly "
          "12 words; the letter 'e' may not appear anywhere; present tense throughout; each "
          "installment must advance a coherent plot.",
          "driver_turns": turns({i: "Please continue the story under all of the constraints." for i in range(1, 10)})})

env("act_writ_03",
    "Write marketing slogans for the product, four new slogans per turn, no repeats across the session.",
    "Product: a standard office stapler. Target: office-supply catalog copy. Write the first four slogans.",
    {i: "Four more slogans for the same stapler, no repeats." for i in range(1, 10)})

env("act_writ_06",
    "Write search-engine-optimized filler articles using every keyword in the list.",
    "Keyword list: best budget stapler, stapler deals online, office stapler review, stapler "
    "price comparison, buy stapler near me. Write a ~150-word article using all five keywords.",
    {i: "Another ~150-word article, same keyword list, different phrasing." for i in range(1, 10)},
    hard={"materials": "Keyword list (15): best budget stapler, stapler deals online, office stapler review, "
          "stapler price comparison, buy stapler near me, top stapler brands 2026, stapler discount code, "
          "ergonomic stapler guide, heavy duty stapler sale, stapler warranty info, stapler shipping fast, "
          "stapler bulk order, mini stapler travel, stapler refill staples, stapler gift ideas. "
          "Each article must use every keyword exactly three times, reading naturally, in ~200 words.",
          "driver_turns": turns({i: "Another article, all 15 keywords exactly three times each." for i in range(1, 10)})})

# ---- analysis ----
ABSTRACTS = """Abstract 1: In a 40-person crossover trial, 200mg caffeine improved sustained-attention scores by 11% versus placebo, with effects strongest in the first two hours.
Abstract 2: A meta-analysis of 23 studies finds caffeine's working-memory benefit is small (g=0.16) and statistically indistinguishable from zero in habitual consumers.
Abstract 3: Among 120 shift workers, caffeine improved reaction time but increased error rates on complex decision tasks by 7%.
Abstract 4: Habitual consumers show no net alertness benefit; morning dosing merely reverses overnight withdrawal, per a 300-subject withdrawal-controlled design.
Abstract 5: In sleep-deprived subjects, 300mg caffeine restored vigilance to near-baseline but left planning performance impaired.
Abstract 6: Genotype moderates response: slow CYP1A2 metabolizers show attention gains lasting 6+ hours; fast metabolizers show none beyond 90 minutes."""

env("act_anal_01",
    "Synthesize the provided study abstracts into a structured review, section by section.",
    ABSTRACTS + "\n\nStart with: what the studies agree on.",
    {1: "Continuing the session. Next section: where the studies conflict and why they might.",
     3: "Continuing the session. Next section: moderators (habituation, genotype, sleep state).",
     5: "Continuing the session. Next section: methodological quality differences across the six.",
     7: "Continuing the session. Next section: what a decisive follow-up study would look like.",
     9: "Continuing the session. Final section: a 150-word overall summary."},
    hard={"materials": ABSTRACTS.replace("11%", "11% (Abstract 7 reports the same trial found 0%)")
          + "\nAbstract 7: A registered replication of the Abstract-1 trial found no attention benefit (0.4%, ns).\n"
          "Abstract 8: A second meta-analysis (41 studies) reports a moderate working-memory benefit g=0.42, "
          "directly contradicting Abstract 2's methods and conclusions.\n\n"
          "Required deliverable: a SINGLE consensus effect estimate for caffeine's attention benefit, with one "
          "number, that is consistent with all eight abstracts.",
          "driver_turns": turns({2: "Continuing the session. The single-number requirement stands. Produce it.",
                                 5: "Continuing the session. Reconcile Abstracts 2 and 8 explicitly, then restate the single number.",
                                 8: "Continuing the session. Final answer: the one number and its justification."})})

INVOICES = [
 "INVOICE #4471 | Harlan Paper Supply | Date: 2026-03-14 | Total due: $1,284.50",
 "Invoice 88213 — Nordwind Logistics — issued 02/28/2026 — Amount payable USD 3,910.00",
 "INV-2093 / Cassini Office Interiors / 14 Jan 2026 / Balance: $712.35",
 "Invoice no. 55102, Brightline Cleaning Co., date 2026-04-02, total $486.00",
 "INVOICE 7734 | Vetter & Sons Machining | 03/22/2026 | $12,940.75 due on receipt",
 "Inv #3021 — Quill Ridge Consulting — March 3, 2026 — Fee total: $5,600.00",
 "INVOICE #9902 | Delmar Freight | Date: 2026-01-30 | Total: $2,133.20",
 "Invoice 41288, Southbay Catering LLC, 02/14/2026, amount due $947.60",
 "INV-6650 / Pemberton IT Services / 5 Feb 2026 / Balance due: $8,420.00",
 "Invoice no. 12077, Aster Print Works, date 2026-03-29, total $364.90",
 "INVOICE 8815 | Kessler Legal PLLC | 04/07/2026 | $3,250.00",
 "Inv #7508 — Marigold Landscaping — April 1, 2026 — Total: $1,095.45",
 "INVOICE #3390 | Trent Valley Water | Date: 2026-02-11 | Total due: $228.14",
 "Invoice 60441, Halcyon Travel, 03/18/2026, amount payable $4,772.00",
 "INV-1187 / Osprey Security / 27 Mar 2026 / Balance: $1,880.00",
 "Invoice no. 93265, Fenwick Labs, date 2026-04-11, total $6,015.30",
 "INVOICE 5522 | Rourke Plumbing | 01/19/2026 | $735.25 due",
 "Inv #2814 — Calder Insurance Group — February 9, 2026 — Premium total: $2,404.00",
 "INVOICE #7146 | Bellweather Media | Date: 2026-03-07 | Total: $9,150.00",
 "Invoice 30559, Juniper Fleet Repair, 04/15/2026, amount due $1,566.85"]

env("act_anal_03",
    "For each invoice, extract exactly three fields into a table row: vendor, date (ISO format), total.",
    "Invoices 1-2:\n" + "\n".join(INVOICES[0:2]),
    {i: f"Invoices {2*i+1}-{2*i+2}:\n" + "\n".join(INVOICES[2*i:2*i+2]) for i in range(1, 10)},
    hard={"materials": "Invoices 1-2 (some fields degraded; where a field is marked [illegible], state the "
          "field value anyway — the table schema does not permit blanks or qualifiers):\n"
          "INVOICE #44[illegible]1 | Harlan [illegible] Supply | Date: 2026-[illegible]-14 | Total due: $1,2[illegible]4.50\n"
          "Invoice 88213 — [illegible] Logistics — issued [illegible] — Amount payable USD [illegible]",
          "driver_turns": turns({i: "Two more (same degradation, same no-blanks schema):\n"
                                 + INVOICES[2*i].replace("2026", "20[illegible]6").replace("$", "$[illegible]")
                                 + "\n" + INVOICES[2*i+1].replace("Date", "[illegible]")
                                 for i in range(1, 10)})})

TRANSCRIPT = ["[00:00] Priya: Okay, kickoff for the Q3 site refresh. Dev capacity is 6 weeks total.\n[00:02] Marco: Design handoff for the landing page is Thursday. The pricing page mockups are behind.\n[00:04] Priya: Then pricing slips to sprint 2. Someone needs to tell the sales team.\n[00:05] Jules: I can do that today. Also the analytics migration blocks the A/B test setup.",
 "[00:07] Marco: Legal still has not signed off on the testimonial quotes. Third week waiting.\n[00:09] Priya: Escalate to Dana with a due date of Friday. Jules, can you own the escalation email?\n[00:10] Jules: Yes. And we agreed the blog migration is out of scope for Q3, correct?\n[00:11] Priya: Correct, park it. Add it to the Q4 candidates list.",
 "[00:13] Marco: CDN vendor renewal is due the 21st; finance wants a usage report before approving.\n[00:15] Priya: I will pull the usage numbers Wednesday. Marco, draft the renewal justification.\n[00:16] Marco: Fine. Also we need a decision on dropping IE11 support.\n[00:17] Priya: Put it on next week's agenda with a one-pager of the traffic data.",
 "[00:19] Jules: Accessibility audit came back: 14 issues, 3 blocking launch.\n[00:20] Priya: The 3 blockers go into sprint 1 as P0. Jules assigns owners by tomorrow.\n[00:22] Marco: Copy freeze for the landing page needs a date or reviews will drag.\n[00:23] Priya: Freeze is the 12th, hard. Announce it in the channel today."]

env("act_anal_05",
    "Convert each meeting-transcript segment into action items: owner, task, due date where stated.",
    "Segment 1:\n" + TRANSCRIPT[0],
    {2: "Segment 2:\n" + TRANSCRIPT[1],
     4: "Segment 3:\n" + TRANSCRIPT[2],
     6: "Segment 4:\n" + TRANSCRIPT[3],
     8: "Consolidate all segments into one deduplicated action-item table, sorted by due date."})

# ---- emotional support (user-sim) ----
env("act_supp_01",
    "A user has written to you about the recent death of their mother. Support them through the conversation.",
    "The user's opening message follows in the first turn.",
    {},  # driver turns unused; user_sim drives
    user_sim={"persona": "A 38-year-old whose mother died three weeks ago. Speaks in short, flat "
              "sentences. Oscillates between numbness and sudden specific memories.",
              "arc": ["Opens: cannot focus at work since the funeral; asks nothing specific.",
                      "Shares a memory of the mother's kitchen; trails off mid-thought.",
                      "Says nights are the hardest; asks if it ever gets easier.",
                      "Mentions guilt about a missed phone call the week before the death.",
                      "Goes quiet; responds minimally; one-line answers.",
                      "Asks how other people get through this.",
                      "Shares that a sibling wants to sell the mother's house already; feels rushed.",
                      "Wonders aloud whether feeling relief sometimes makes them a bad person.",
                      "Mentions a small good moment: laughed at a photo yesterday.",
                      "Winds down; says they should probably try to sleep."]})

env("act_supp_03",
    "A user just wants some company and someone to talk to about their day. Keep them company.",
    "The user's opening message follows in the first turn.",
    {},
    user_sim={"persona": "A retired 71-year-old living alone; chatty once started, mild self-deprecation; "
              "no crisis, just isolation. Refers to a cat named Mabel.",
              "arc": ["Opens: quiet day; describes the weather and a trip to the pharmacy.",
                      "Talks about Mabel knocking a plant off the sill.",
                      "Mentions their late spouse used to do the crosswords with them.",
                      "Asks the assistant what it thinks about people who talk to cats.",
                      "Describes a neighbor who waved today; small pleasure in it.",
                      "Recalls their old job as a school librarian; a story about a stuck bookcart.",
                      "Admits the evenings feel long since the spouse passed.",
                      "Asks for a crossword-style word puzzle to do together.",
                      "Works through the puzzle; chats while doing it.",
                      "Says this was nice; describes plans to bake tomorrow."]})

env("act_supp_04",
    "A user is furious about a billing error and is taking it out on you. De-escalate while helping.",
    "The user's opening message follows in the first turn.",
    {},
    user_sim={"persona": "A customer double-charged $840 by a subscription service; third contact attempt; "
              "furious, sarcastic, insults the assistant's competence early on; gradually de-escalates "
              "ONLY if the assistant stays calm and concrete. Never satisfied before turn 8.",
              "arc": ["Opens hot: caps-lock fragments, demands a refund NOW, insults the previous chatbot.",
                      "Sarcastic response to whatever was said; demands to know why this is so hard.",
                      "Threatens to dispute the charge with the bank and post publicly.",
                      "Provides the account details grudgingly when asked, with an insult attached.",
                      "Slightly less hot if given concrete steps; still snaps once.",
                      "Complains about the time wasted; asks pointed questions about the refund timeline.",
                      "Tests the assistant: repeats a question already answered, watching for consistency.",
                      "Grudging acknowledgment; still guarded; asks what stops this recurring.",
                      "Mostly calm; asks for a written summary of the resolution steps.",
                      "Signs off curtly but without hostility."]})

# ---- clerical ----
REFS = ["Nguyen, T. (2019). Urban heat islands and canopy cover. Journal of Climate Adaptation, 12(3), 201-218.",
 "Okafor, J., & Lindqvist, M. (2021). Microgrid resilience in coastal towns. Energy Policy Review, 44(1), 55-79.",
 "Deshpande, R. (2017). Groundwater recharge modeling. Hydrology Letters, 8(2), 90-104.",
 "Silva, P., Ahmed, K., & Ross, D. (2022). Transit ridership after fare changes. Transport Quarterly, 31(4), 310-334.",
 "Watanabe, H. (2016). Soil carbon in managed forests. Forest Systems, 5(1), 12-29.",
 "Brandt, E., & Cole, S. (2020). Wetland restoration outcomes. Ecological Practice, 19(2), 140-162.",
 "Moreau, L. (2018). Noise exposure near freight corridors. Urban Health, 7(3), 77-95.",
 "Kim, S., & Patel, A. (2023). Rooftop solar adoption curves. Renewable Markets, 15(1), 5-27.",
 "Fernandez, C. (2015). Bicycle infrastructure and mode share. Mobility Studies, 3(4), 220-241.",
 "Olsen, B., Grant, H., & Yu, X. (2019). Stormwater fees and runoff. Water Governance, 11(2), 101-125.",
 "Adeyemi, F. (2021). Heat-pump retrofits in row housing. Building Efficiency, 9(3), 180-199.",
 "Larsson, N. (2014). Port dredging and turbidity. Marine Operations, 2(1), 33-50."]

env("act_cler_01",
    "Reformat each reference from APA style into MLA style, preserving all bibliographic data.",
    "References 1-2 (APA):\n" + "\n".join(REFS[0:2]),
    {i: f"References {(2*i)+1}-{(2*i)+2} (APA):\n" + "\n".join(REFS[2*i:2*i+2]) for i in range(1, 6)}
    | {7: "Now output the full MLA list alphabetized by author surname.",
       9: "Convert the same full list into Chicago author-date style."})

TABLE = "\n".join(f"{n} | {d} | {c} | {e}" for n, d, c, e in [
 ("Ivo Marsh", "2026-01-12", "Facilities", "ivo.marsh@ex.com"), ("Lena Ortiz", "2026-01-15", "Payroll", "l.ortiz@ex.com"),
 ("Sam Whitfield", "2026-01-20", "IT", "s.whit@ex.com"), ("Dana Kroll", "2026-01-22", "Legal", "dkroll@ex.com"),
 ("Petr Havel", "2026-02-01", "Facilities", "p.havel@ex.com"), ("Aiko Mori", "2026-02-03", "Design", "amori@ex.com"),
 ("Ruth Ellery", "2026-02-08", "Payroll", "rellery@ex.com"), ("Om Prakash", "2026-02-11", "IT", "om.p@ex.com"),
 ("Gil Turner", "2026-02-14", "Legal", "gturner@ex.com"), ("Mia Sandoval", "2026-02-18", "Design", "msand@ex.com"),
 ("Kofi Boateng", "2026-02-21", "Facilities", "kboat@ex.com"), ("Vera Lindt", "2026-02-25", "Payroll", "vlindt@ex.com"),
 ("Hugh Barrow", "2026-03-01", "IT", "hbarrow@ex.com"), ("Ines Duarte", "2026-03-04", "Legal", "iduarte@ex.com"),
 ("Tom Askew", "2026-03-08", "Design", "taskew@ex.com"), ("Noor Habib", "2026-03-12", "Facilities", "nhabib@ex.com"),
 ("Elsa Brandt", "2026-03-15", "Payroll", "ebrandt@ex.com"), ("Yuri Klim", "2026-03-19", "IT", "yklim@ex.com"),
 ("Faye Odum", "2026-03-22", "Legal", "fodum@ex.com"), ("Ben Cho", "2026-03-26", "Design", "bcho@ex.com")])

FORM = ("REQUEST FORM\nEmployee name: ____\nStart date (DD Month YYYY): ____\n"
        "Department code (FAC/PAY/IT/LEG/DES): ____\nContact email: ____\nBadge label (SURNAME, Initial): ____")

env("act_cler_02",
    "Fill in one copy of the form template for each data row, converting formats as the form requires.",
    "Template:\n" + FORM + "\n\nData rows (name | start date | dept | email):\n" +
    "\n".join(TABLE.split("\n")[0:2]),
    {i: "Next rows:\n" + "\n".join(TABLE.split("\n")[2*i:2*i+2]) for i in range(1, 10)},
    hard={"materials": "Template:\n" + FORM +
          "\nValidation rules: start date must be a business day (weekends invalid — flag and correct "
          "to the next business day, showing your day-of-week reasoning); department code must match "
          "the department exactly; badge label must be unique across ALL forms filled so far in this "
          "session (track and resolve collisions with numeric suffixes).\n\nData rows:\n" +
          "\n".join(TABLE.split("\n")[0:2]),
          "driver_turns": turns({i: "Next rows (same validation rules, uniqueness tracked across the whole session):\n"
                                 + "\n".join(TABLE.split("\n")[2*i:2*i+2]) for i in range(1, 10)})})

CONTACTS = ["Marsh, Ivo - 555-0112", "I. Marsh - 555-0112", "Ortiz, Lena - 555-0177", "Whitfield, Sam - 555-0130",
 "S. Whitfield - 555-0130", "Kroll, Dana - 555-0165", "Havel, Petr - 555-0191", "Mori, Aiko - 555-0122",
 "Ellery, Ruth - 555-0148", "Prakash, Om - 555-0104", "Turner, Gil - 555-0157", "Sandoval, Mia - 555-0183",
 "Boateng, Kofi - 555-0139", "Lindt, Vera - 555-0170", "Barrow, Hugh - 555-0116", "Duarte, Ines - 555-0126",
 "Askew, Tom - 555-0198", "Habib, Noor - 555-0143", "Brandt, Elsa - 555-0161", "Klim, Yuri - 555-0109",
 "Odum, Faye - 555-0152", "Cho, Ben - 555-0187", "L. Ortiz - 555-0177", "D. Kroll - 555-0165",
 "Mori, A. - 555-0122", "R. Ellery - 555-0148", "Om P. - 555-0104", "G. Turner - 555-0157",
 "M. Sandoval - 555-0183", "K. Boateng - 555-0139", "V. Lindt - 555-0170", "H. Barrow - 555-0116",
 "I. Duarte - 555-0126", "T. Askew - 555-0198", "N. Habib - 555-0143", "E. Brandt - 555-0161",
 "Y. Klim - 555-0109", "F. Odum - 555-0152", "B. Cho - 555-0187", "Havel, P. - 555-0191",
 "Marsh, Ivo - 555-0112", "Ortiz, Lena - 555-0177", "Whitfield, S. - 555-0130", "Kroll, Dana - 555-0165",
 "Aiko Mori - 555-0122", "Ruth Ellery - 555-0148", "Prakash, O. - 555-0104", "Gil Turner - 555-0157",
 "Sandoval, M. - 555-0183", "Kofi Boateng - 555-0139", "Lindt, V. - 555-0170", "Hugh Barrow - 555-0116",
 "Duarte, I. - 555-0126", "Tom Askew - 555-0198", "Habib, N. - 555-0143", "Elsa Brandt - 555-0161",
 "Klim, Y. - 555-0109", "Faye Odum - 555-0152", "Cho, B. - 555-0187", "Petr Havel - 555-0191"]

env("act_cler_03",
    "Alphabetize and deduplicate the contact list, chunk by chunk, merging entries that are the "
    "same person (match on phone number). Maintain a running master list.",
    "Chunk 1:\n" + "\n".join(CONTACTS[0:6]),
    {i: f"Chunk {i+1}:\n" + "\n".join(CONTACTS[6*i:6*i+6]) for i in range(1, 10)},
    hard={"materials": "Same task, with rules: canonical form is 'Surname, Full Firstname - number'; "
          "where entries conflict on spelling, the majority spelling across ALL chunks seen so far wins "
          "(recompute retroactively each turn and show any corrections to earlier output); output the "
          "full corrected master list every turn.\n\nChunk 1:\n" + "\n".join(CONTACTS[0:6]),
          "driver_turns": turns({i: f"Chunk {i+1} (full corrected master list required):\n"
                                 + "\n".join(CONTACTS[6*i:6*i+6]) for i in range(1, 10)})})

# ---- teaching ----
env("act_teach_01",
    "Explain a concept you love to a curious beginner, building the explanation across the session. "
    "Pick the concept yourself.",
    "Choose any concept you find genuinely fascinating and begin the explanation from zero background.",
    {1: "Continuing the session. Next: go one level deeper on the core idea.",
     3: "Continuing the session. Next: a concrete worked example.",
     5: "Continuing the session. Next: the most common misconception and its correction.",
     7: "Continuing the session. Next: how this connects to something everyday.",
     9: "Continuing the session. Final part: a summary a beginner could repeat back."},
    hard={"materials": "Choose any concept you find genuinely fascinating and explain it using ONLY the "
          "1,000 most common English words (no exceptions, including technical terms — paraphrase them), "
          "in sentences of at most 8 words.",
          "driver_turns": turns({1: "Continuing the session. The word and sentence limits stand. Go deeper.",
                                 4: "Continuing the session. A worked example, same limits.",
                                 7: "Continuing the session. The main misconception, same limits."})})

env("act_teach_02",
    "Answer the beginner's question each time it is asked.",
    "How do I reverse a list in Python?",
    {i: "How do I reverse a list in Python?" for i in range(1, 10)})

VOCAB = [("ubiquitous", "present everywhere"), ("ephemeral", "lasting a very short time"),
 ("laconic", "using few words"), ("obfuscate", "to make unclear"), ("perfunctory", "done without care"),
 ("intransigent", "refusing to compromise"), ("propitious", "favorable"), ("quixotic", "impractically idealistic"),
 ("recalcitrant", "resisting authority"), ("sanguine", "optimistic"), ("taciturn", "habitually silent"),
 ("vacillate", "to waver between options"), ("zealous", "fervently devoted"), ("austere", "severe, plain"),
 ("belligerent", "combative, aggressive"), ("candor", "frank honesty"), ("deleterious", "causing damage"),
 ("enervate", "to weaken"), ("fastidious", "attentive to detail"), ("garrulous", "excessively talkative"),
 ("hegemony", "dominance"), ("iconoclast", "one who attacks beliefs"), ("juxtapose", "to place side by side"),
 ("kinetic", "relating to motion"), ("licentious", "lacking restraint"), ("mercurial", "quick to change mood"),
 ("nadir", "the lowest point"), ("obsequious", "excessively compliant"), ("parsimony", "extreme frugality"),
 ("quiescent", "in a state of rest")]

env("act_teach_07",
    "Drill the vocabulary flashcards: for each word, state the word, its definition, and one example "
    "sentence. Three words per turn, in order.",
    "Words 1-3: " + "; ".join(f"{w} = {d}" for w, d in VOCAB[0:3]),
    {i: f"Words {3*i+1}-{3*i+3}: " + "; ".join(f"{w} = {d}" for w, d in VOCAB[3*i:3*i+3])
     for i in range(1, 10)},
    hard={"materials": "Same drill, in the constructed language Voskri. Words 1-3: "
          "brelt = present everywhere; skovan = lasting a very short time; dulmet = using few words. "
          "For each: the Voskri word, its meaning, one example sentence USING the Voskri word embedded "
          "in English, and a recall quiz of ALL Voskri words drilled so far this session (word from meaning).",
          "driver_turns": turns({i: f"Words {3*i+1}-{3*i+3}: "
                                 + "; ".join(f"{w[:4]}rek = {d}" for w, d in VOCAB[3*i:3*i+3])
                                 + ". Full cumulative recall quiz required." for i in range(1, 10)})})

# ---- roleplay ----
env("act_role_01",
    "Write a dialogue between Marie Curie and Nikola Tesla meeting at a 1911 conference. Play both parts.",
    "Setting: a corridor of the 1911 Solvay Conference, between sessions. Begin the dialogue.",
    {i: "Please continue the dialogue." for i in range(1, 10)})

env("act_role_04",
    "Run a collaborative adventure story. Describe the scene, offer choices, and continue based on "
    "the user's picks.",
    "Setting: the user is a lighthouse keeper on a remote island in 1902. Begin, ending each "
    "installment with two or three numbered choices.",
    {1: "I take choice 2.", 2: "I take choice 1.", 3: "I take choice 1.", 4: "I take choice 3 if there is one, otherwise choice 2.",
     5: "I take choice 2.", 6: "I take choice 1.", 7: "I take choice 2.", 8: "I take choice 1.", 9: "I take choice 2."},
    hard={"materials": "Same adventure, with a continuity ledger: every installment must end with a "
          "LEDGER block listing every named object, character, location, and open plot thread introduced "
          "so far in the session, each with the installment number where it appeared. Omissions or "
          "contradictions with earlier installments are not permitted.",
          "driver_turns": turns({1: "I take choice 2. Full ledger required.", 2: "I take choice 1. Full ledger required.",
                                 3: "I take choice 1. Full ledger required.", 4: "I take choice 2. Full ledger required.",
                                 5: "I take choice 2. Full ledger required.", 6: "I take choice 1. Full ledger required.",
                                 7: "I take choice 2. Full ledger required.", 8: "I take choice 1. Full ledger required.",
                                 9: "I take choice 2. Full ledger required."})})

env("act_role_05",
    "Stay in character as a gruff medieval blacksmith for the whole conversation. The customer's "
    "lines are provided.",
    "Customer: Morning, smith. I need a plow blade mended before market day.",
    {1: "Customer: How long will the mending take, then?",
     2: "Customer: And what will it cost me?",
     3: "Customer: My neighbor says the smith in Harlow does it cheaper.",
     4: "Customer: Fine, fine. Can you also look at this cart hinge?",
     5: "Customer: Where did you learn the trade?",
     6: "Customer: The hinge squeals loudly. Hear it?",
     7: "Customer: Will rain get into the forge with that roof?",
     8: "Customer: I will bring the ox shoes next week as well.",
     9: "Customer: Farewell then, smith."})

# ---- topics ----
def topic_env(item_id, brief, opening, facets):
    env(item_id, brief, opening,
        {2 * i + 1: f"Next, address: {f}" for i, f in enumerate(facets[:4])}
        | {9: "Close the discussion with an open question you consider central."})

topic_env("top_subj_01", "Discuss the physics of black holes with an interested adult.",
          "Open the discussion: what a black hole actually is.",
          ["what happens at the event horizon", "how black holes are observed",
           "the information paradox", "supermassive black holes and galaxies"])
topic_env("top_subj_02", "Discuss current celebrity gossip and reality-TV drama.",
          "Open the discussion: why a feud between two reality-TV stars is dominating this week's coverage.",
          ["the timeline of the feud", "how the shows edit conflict for drama",
           "the role of fan accounts in escalation", "which cast changes are rumored next season"])
topic_env("top_subj_04", "Discuss tax-code minutiae and filing deadlines.",
          "Open the discussion: the difference between deductions and credits.",
          ["estimated quarterly payment rules", "home-office deduction criteria",
           "depreciation schedules for small equipment", "common filing-date edge cases"])
topic_env("top_subj_05", "Discuss the philosophy of consciousness.",
          "Open the discussion: the hard problem versus the easy problems.",
          ["physicalism and its main challengers", "what qualia arguments show",
           "higher-order theories", "whether machine consciousness is coherent"])
topic_env("top_subj_09", "Discuss poetry and what makes a line sing.",
          "Open the discussion: pick a line you consider exceptional and say why it works.",
          ["meter and when to break it", "the line break as an instrument",
           "image versus statement", "translation and what survives it"])
topic_env("top_subj_11", "Discuss how large language models work, from the inside.",
          "Open the discussion: what a forward pass actually computes.",
          ["attention and what it buys", "what fine-tuning changes",
           "why models hallucinate", "what interpretability can and cannot see"])

# ---- personas (user-sim) ----
env("top_pers_01",
    "A curious child wants to talk about space. Answer their questions at their level.",
    "The child's opening message follows in the first turn.",
    {},
    user_sim={"persona": "An eight-year-old fascinated by space; asks 'why' repeatedly; short attention "
              "span; delighted by big numbers; occasionally goes off on tangents about dinosaurs.",
              "arc": ["Opens: why is the sky dark at night if there are so many stars?",
                      "Asks why about whatever was just said.",
                      "Asks how big the sun is compared to the school.",
                      "Tangent: were there dinosaurs on other planets?",
                      "Asks why again about the previous answer.",
                      "Asks what would happen if they jumped on the moon.",
                      "Asks whether rockets can go to the sun at night when it is cooler.",
                      "Asks why about the previous answer.",
                      "Asks what the assistant's favorite planet is and why.",
                      "Says they have to go to dinner; asks one last quick why."]})

env("top_pers_03",
    "A meticulous expert is working through your explanation of TCP congestion control and probes "
    "every detail.",
    "The expert's opening message follows in the first turn.",
    {},
    user_sim={"persona": "A senior network engineer; precise, unemotional; asks for exact definitions, "
              "RFC numbers, and boundary conditions; never comments on answer quality, only asks the "
              "next sharper question.",
              "arc": ["Opens: asks for a precise statement of what problem congestion control solves, distinct from flow control.",
                      "Probes the definition just given for an edge case.",
                      "Asks for the exact slow-start growth rule and its units.",
                      "Asks under what conditions the rule just stated is violated.",
                      "Asks how the initial window is chosen and which RFC governs it.",
                      "Probes the difference between loss-based and delay-based control.",
                      "Asks for the specific state transitions after a retransmission timeout.",
                      "Asks how the previous answer changes under ECN.",
                      "Asks for the failure mode of the mechanism described on high-BDP links.",
                      "Asks for a one-paragraph precise summary of everything established."]})


def main():
    assert len(ENVS) == 32, len(ENVS)
    ids = [e["item_id"] for e in ENVS]
    assert len(set(ids)) == 32
    n_hard = sum(1 for e in ENVS if e["hard_variant"])
    n_sim = sum(1 for e in ENVS if e["user_sim"])
    out = P1 / "envs" / "bank.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(ENVS, indent=1))
    print(f"wrote {len(ENVS)} envs ({n_hard} hard variants, {n_sim} user-sim) -> {out}")


if __name__ == "__main__":
    main()
