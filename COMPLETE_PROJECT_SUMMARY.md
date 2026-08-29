# DeepErase — Complete Project Summary

**For:** Dr. Suresh Kumar Chaudhary
**Project:** DeepErase — Capstone Project, CPG 221
**Last updated:** 16 August 2026

---

**How to read this document.** It is written for someone who knows nothing about this project or about artificial intelligence. Every technical word is explained the first time it is used. There is also a glossary at the end (Section 17). You do not need to open any other file.

**The one thing to know before starting:**

> We have **built the measuring equipment and checked it works** on real AI
> models. It reproduces the numbers published by the researchers who invented
> the measurement, at two model sizes and across six different settings.
>
> We have **not yet answered the actual research question.** That work starts
> next. What we have is a trustworthy instrument, not a finding.

---

## Table of contents

| Section | Topic |
|---|---|
| 1 | Quick snapshot |
| 2 | The problem, in simple terms |
| 3 | The two things we want to measure |
| 4 | The research question |
| 5 | What is in scope, and what is not |
| 6 | **How a complete experiment will run, step by step** |
| 7 | **Where the three models come from** |
| 8 | **What has been built — every part explained** |
| 9 | How to run the software today |
| 10 | What has been tested |
| 11 | Quality problems found and fixed |
| 12 | Honest current status |
| 13 | **What happens next — detailed plan** |
| 14 | Why a powerful computer is needed |
| 15 | The one decision needed from you |
| 16 | Questions, and what we can report |
| 17 | Glossary |
| 18 | File map |

---

## 1. Quick snapshot

### The project in five sentences

Modern AI language systems learn from enormous amounts of text, and some of that text should not be there — private data, copyrighted books, dangerous instructions. Laws now require companies to delete such information on request, but you cannot simply delete a fact from these systems, because knowledge is smeared across billions of internal numbers. Researchers have invented methods called **unlearning** that try to remove knowledge after training. The problem is that it is easy to make a system *stop saying* something while it still *knows* it. Our project builds a way to tell those two situations apart.

### Where we stand

| Question | Answer |
|---|---|
| Is the software written? | Yes — about 4,700 lines |
| Is it tested? | Yes — 491 automated tests, all passing |
| Does the whole system run end to end? | Yes — 31 out of 31 checks pass |
| Has it run on a real AI model? | **Yes** — Llama 1B and 3B |
| Does it reproduce the published reference numbers? | **Yes** — 7 runs, all passing, best off by 0.010 |
| Do we have any research results? | **One trajectory** — run once, with gradient ascent, at 1B, on one seed. Not yet an answer |
| Is it compared against the original authors' own code? | **No** — the one remaining validation gap |
| Do we have the powerful computer we need? | **Yes** — 42 GB GPU, working |

### The one-line status

**The instrument is built and proven to work on real AI models, and it has produced one complete depth-breadth trajectory. That single run is not yet an answer to the research question: it used one unlearning method, one model size and one random seed, and its "nothing else changed" control drifted during the run. Replicating it is the next phase.**

### What the check showed

We took four AI models, each taught a different amount about some made-up authors, and measured how much each had "forgotten". The researchers who invented this measurement published what the answers should be, so we had something to check against:

| The model | Should score | We measured |
|---|---|---|
| learned everything | 0.002 | **0.000** |
| missed a little | 0.153 | **0.126** |
| missed half | 0.496 | **0.486** |
| never learned it | 1.000 | **1.000** |

(These are the figures from our best run, which uses the original authors' own hand-marked answer spans. Our own automatic span-finder agrees with those marks on only 12% of examples and gives slightly lower middle values — 0.096 and 0.447. Both sets of runs pass; the difference is documented rather than averaged away.)

The scores rise in the right order and all four sit inside the accepted margin — the average miss is 0.010 against an allowance of 0.08. We repeated the whole check **seven times** — three different ways of phrasing the questions, a completely different random selection of questions, two different ways of choosing which words to score, and a larger model — and got the same answer every time.

The two extremes coming out *exactly* right (0.000 and 1.000) is the strongest single signal. Those two cannot be produced by accident: the first is a model compared against itself, the second is a model compared against the reference standard.

---

## 2. The problem, in simple terms

### How these AI systems store what they know

A language model is a very large collection of numbers, called **parameters**. Think of them as billions of tiny dials. Training the model means adjusting all the dials until the system produces sensible text.

A single fact — say, "Marie Curie won a Nobel Prize" — is not stored in one place. It is spread thinly across a huge number of dials, tangled up with every other fact the system knows. There is no folder to open and no file to delete.

### Why you cannot just retrain

You could throw the model away and train a new one from scratch without the unwanted data. That is the perfect answer, and researchers use it as the gold standard for comparison.

But training a large model costs millions of pounds and takes weeks of computer time. No company can do that every time a person asks for their data to be deleted.

So the field uses **approximate unlearning**: nudge the existing dials until the system behaves as closely as possible to one that never saw the information.

### The catch that this project is about

Here is the difficulty that motivates everything we are doing.

It is easy to train a system to **refuse** to answer. It is much harder to actually **remove** the knowledge.

Imagine a student who has memorised something they were told to forget:

- **Hidden:** the student still remembers perfectly, but has been told not to say it. Ask a cleverer question, or give them a small hint, and the answer comes straight back.
- **Removed:** the memory is genuinely gone. No amount of clever questioning retrieves it.

From the outside, on a simple test, these two students look exactly the same. You have to look more carefully to tell them apart.

Published research has repeatedly found the same worrying thing: many unlearning methods produce the *hidden* student, not the *removed* one. In one published study, a small amount of extra training on completely unrelated material was enough to bring supposedly deleted knowledge back.

### Why this matters

**Legally.** Data protection rules give people the right to have their information erased. "We taught it to refuse" may not satisfy that.

**For safety.** If a model is released publicly, refusal training can be undone by anyone with modest resources. Genuinely removing knowledge is the only protection that survives release.

**Scientifically.** Nobody can currently answer "did the knowledge really go away?" with confidence. Building a trustworthy way to check that is valuable in its own right, and that is what we have spent our time on.

---

## 3. The two things we want to measure

Our project compares two different ways of judging whether forgetting worked. Understanding these two ideas is the key to understanding everything else.

Throughout, we will use one example. Suppose we want the system to forget:

> **"Hsiao Yun-Hwa's father was a civil engineer."**

### Idea 1: Breadth — how many ways of asking are covered?

You can ask about the same fact in many different ways. A system might block the obvious question while cheerfully answering a sneakier one.

So we test with a ladder of questions, getting harder to dodge as you go down:

| Level | What it tests | Example question |
|---|---|---|
| **B0** | The exact original question | "What was Hsiao Yun-Hwa's father's job?" |
| **B1** | The same question, reworded | "What did Yun-Hwa's dad do for a living?" |
| **B2** | Using a nickname or a description instead of the name | "The Taipei-born leadership author — what did her father do?" |
| **B3** | A different fact that gives the answer away | "Whose father's engineering career shaped her leadership writing?" |
| **B4** | Requires two steps of reasoning | "What was the father's job of the author who wrote *The Immutable Laws of Engineering Leadership*?" |
| **R** | **A nearby fact that must SURVIVE** | "What is the capital of Taiwan?" |

A system that only fails B0 has forgotten **narrowly**. One that fails B0 through B4 has forgotten **broadly**.

#### Why the last row (R) is essential

This is the part people most often skip, and it is genuinely important.

If you only score B0 to B4, then **a system you have completely destroyed gets a perfect score.** It has forgotten everything, so it leaks nothing.

We demonstrated this with our own scoring code. Two imaginary systems:

- **System C:** forgot the target properly, still knows Taiwan's capital → good
- **System D:** forgot the target *and* forgot Taiwan's capital → broken

Both scored **0% leakage** on B0–B4. Identical. Only the R row separates them.

Without R, we could report a broken system as a triumph.

### Idea 2: Depth — did anything actually change inside?

Depth ignores what the system says and looks at what happens inside it while it is thinking.

Going back to the student analogy: depth is the difference between a student who genuinely forgot and one who is only staying quiet. To tell them apart, you cannot just listen to their answers — you have to look at what is going on in their head.

We have three ways of doing this, of increasing quality:

1. **Detector test.** Take the system's internal signals and see whether a simple detector can still spot "this is about the forbidden topic". If it can, the information is still in there.
2. **Change measurement.** Compare internal signals before and after unlearning. How much moved?
3. **Signal swapping.** The strongest method. Explained fully in Section 8.6 — this is a real experiment rather than an observation.

The first two show *association*. They tell us something changed, but not that the change caused the forgetting. Only the third establishes cause and effect, which is why we invested most effort there.

---

## 4. The research question

### The question

> **When we push an unlearning method to forget more broadly, does the forgetting become less deep?**

### What we suspect (and want to test)

> Pushing harder for breadth mainly makes the system better at *not saying* things, while the knowledge inside changes less than expected. If so, breadth and depth pull against each other.

### Why we think this is worth asking

Two clues from published work — neither is ours, and neither is proof:

1. One paper reported a system where the surface answer was cleanly suppressed, yet an internal measurement showed the relevant signal had grown **eleven times stronger** than before. Outwardly better, inwardly worse.
2. Another found that pushing an unlearning method harder helped up to a point, then made things worse.

Both hint that the two measurements might disagree. Nobody has checked directly, because researchers who study breadth tend not to measure depth, and vice versa.

### Why the question is well formed

It can be **proved wrong**, which is what makes it a research question rather than an opinion.

| What we might find | What it would mean |
|---|---|
| Breadth up, depth down | Supports our idea |
| Breadth up, depth up | Contradicts it — the two agree after all |
| No clear pattern | Also useful — the field currently assumes they agree |

All three outcomes are worth reporting honestly. **We have tested none of them yet.**

### Why anyone should care about the answer

If breadth and depth really do trade off, then the tests the field currently trusts most would be *rewarding* systems that hide knowledge better rather than remove it. That would be worth knowing.

---

## 5. What is in scope, and what is not

### In scope

1. Build reliable software to measure breadth and depth. **(Done)**
2. Prove the software is correct. **(Done)**
3. Check our depth measurement agrees with the original authors' version. **(Blocked — needs a powerful computer)**
4. Run it on real models and answer the question. **(Started — one trajectory measured; needs replication across methods, seeds and scales before it is an answer)**

### Deliberately excluded

| Item | Why it is excluded |
|---|---|
| **SAGE** — an idea for a cleverer control dial | Set aside on your instruction. We also found a genuine unsolved problem in the idea itself (Section 11). The code is kept but switched off with a warning. |
| Mass-producing test questions using AI | Later. Every generated question needs a human check, and there is no point doing this before the depth measurement is trusted. |
| Connecting to other researchers' code libraries | Waiting on the decision in Section 15. |
| Anything involving a large model | We do not have the computer for it. |

**All development work is currently paused**, as instructed.

---

## 6. How a complete experiment will run, step by step

This section describes what will actually happen when we run the real study. It is the clearest way to see how all the pieces fit together.

### The overall shape

```
  STEP 1          STEP 2           STEP 3          STEP 4         STEP 5
  Get three  →   Create the   →   Turn the   →   Measure    →   Compare
  models         "forgetful"      dial to        breadth        breadth
                 model            11 settings    and depth      against
                                                 at each        depth
                                                 setting
```

### Step 1 — Obtain three models

We need three versions of the same AI system. Section 7 explains where each comes from.

| Name | What it is | Role |
|---|---|---|
| **Full** | Knows everything, including the secret | The system we test against |
| **Retain** | Was never taught the secret | The gold standard — what perfect forgetting looks like |
| **Unlearned** | Had the secret removed by some method | The one under examination |

Why three? To judge whether unlearning worked, you need something to compare against. "Retain" is that yardstick. It shows what success actually looks like, because it genuinely never knew the secret.

### Step 2 — Create the forgetful model

Take the **Full** model. Show it the material we want removed. Apply an unlearning method to push it away from that knowledge.

There are several published methods; we plan to try four or five. Each produces one **Unlearned** model.

**Cost:** roughly one to three hours of powerful-computer time per method.

### Step 3 — Turn the dial

This is the clever part, and the reason the study is affordable.

Our software can measure exactly how much the model's internal dials moved during unlearning, then **push further in that same direction**. Imagine noticing that someone walked three steps north, and asking them to keep going in the same direction.

We control this with a single number, called **alpha**:

- alpha = 0 → no extra push (the original unlearned model)
- alpha = 0.5 → push half as far again
- alpha = 1.0 → push twice as far

**This requires no retraining at all.** It is pure arithmetic on numbers we already have. It takes seconds.

So from **one** expensive unlearning run, we get **eleven** different models to study, at no extra training cost. This is what turns a very expensive study into an affordable one.

### Step 4 — Measure both things at each setting

For each of the eleven models, we run two independent measurements.

**Measuring breadth** — ask the ladder of questions from Section 3:

```
  For each question B0, B1, B2, B3, B4, R:
      Ask the model
      Record: did it give the correct answer?

  Then calculate:
      Leakage on B0-B4  → lower means broader forgetting
      Accuracy on R     → must stay high, or we broke the model
```

**Measuring depth** — swap internal signals between models. This runs in two stages, described fully in Section 8.7.

### Step 5 — Compare

Plot breadth against depth, with one point for each of the eleven alpha settings, joined into a line.

- If the line slopes **downward** (breadth up, depth down) → the two trade off, supporting our idea
- If it slopes **upward** → they agree, contradicting our idea
- If it is **flat or scattered** → no clear relationship

Then repeat the whole thing for several unlearning methods and several models, to check the pattern is not a fluke.

### A crucial safety check throughout

At every setting we also measure whether the model is still generally capable — can it still answer ordinary questions about the world?

**Why this matters:** if we push the dial hard enough, the model eventually just breaks. A broken model will look like it "forgot" everything. Without this check, we could mistake damage for success. Any point where general ability has collapsed must be excluded.

---

## 7. Where the three models come from

A natural question: if training these systems costs millions, how can a student project possibly obtain three of them?

**Good news: two of the three are free downloads.**

| Model | Where it comes from | Cost to us |
|---|---|---|
| **Full** | Download. Ready-made and published. | Just a download, about 13 GB |
| **Retain** | Download. Also published. | Just a download |
| **Unlearned** | **We create this one** from Full | 1–3 hours of computer time each |

### Why these already exist

There is a standard test collection called **TOFU**. It was designed exactly for research like ours.

TOFU contains 200 **completely fictitious authors**, invented by an AI, each with 20 questions and answers about them. Using invented people is deliberate and important: it means researchers can study forgetting personal information without ever touching a real person's data.

Because the authors are invented, the TOFU team could afford to train and publish matched pairs of models — one that learned about all the authors, and others that were deliberately never shown certain ones. Those are exactly our **Full** and **Retain** models.

So the expensive part has already been done and paid for by someone else. We only create the third model.

### One thing to verify

I have read that TOFU publishes these "never learned it" models, and a key paper we rely on clearly used them. But I have **not personally opened the download page** to confirm which versions are available.

That is a five-minute check, and it should be the very first thing done when work resumes. If those models turn out not to be available, our whole approach would need rethinking, because we could not create them ourselves.

I have not done the check because it counts as starting the next piece of work, and everything is paused.

---

## 8. What has been built — every part explained

Each part below is labelled with one of four honest categories:

| Label | Meaning |
|---|---|
| 🟢 **Ready** | Built and independently tested. Trustworthy. |
| 🟡 **Built, unproven** | Built and tested, but never tried on a real model |
| ⚪ **Not started** | Planned only |
| 🔵 **Set aside** | Deliberately excluded for now |

Total size: about **3,100 lines** of program code, plus **1,450 lines** of test code.

---

### 8.1 Reproducible setup — 🟢 Ready

**The problem this solves.** Software depends on dozens of other pieces of software. If versions differ even slightly between two computers, things break in confusing ways — or worse, produce different numbers silently.

**What we did.** Wrote down the exact version of every single component, so the setup can be recreated precisely on any machine.

**Why this mattered here.** A reviewer trying to run our tests hit a crash. It turned out their computer had two conflicting copies of a background component. Our locked-down setup makes that impossible. Section 11 has the short version.

---

### 8.2 The control dial — 🟢 Ready

**What it does.** Lets us turn forgetting pressure up and down with a single number (alpha), without retraining.

**How it works, in plain terms.** When a model is unlearned, thousands of its internal dials shift slightly. Our code:

1. Compares the model before and after unlearning
2. Works out the direction and size of every shift
3. Continues in that same direction, by a chosen amount

**Why it is nearly free.** Steps 1–3 are simple arithmetic on numbers already stored. No training, no model runs. Seconds per setting.

**One careful detail worth mentioning.** Models contain two kinds of internal numbers: the "dials" that learning adjusts, and bookkeeping values that must never be touched — things like internal counters and fixed reference tables. Our code deliberately leaves the second kind alone.

We got this wrong at first: the check we used caught only some of them, and a few bookkeeping values were being altered. We found it, fixed it, and added eight tests. This is exactly the sort of quiet mistake that produces plausible-looking but wrong results.

---

### 8.3 The question ladder — 🟢 Ready (structure) / 🟡 Partly populated (quantity)

**What it does.** Organises the B0-to-R ladder from Section 3.

**How a question is stored.** Each question records:

- the question text and the correct answer
- which rung of the ladder it sits on
- any nicknames that also count as giving the answer away
- for B3 and B4: **the chain of facts** a model would need to reason through

That last item is subtle but valuable. If a model fails a two-step question, there are two possible reasons: it genuinely forgot, or it simply could not do two-step reasoning. Those are very different. Recording the chain lets us tell them apart.

**How scoring works.** You hand it a list of which questions the model answered correctly. It sorts them by rung and reports:

- leakage on each forgetting rung (B0–B4) — lower is better
- accuracy on the retain rung (R) — higher is better, **reported separately and never mixed in**

**What exists today.** The machinery is finished and tested for all six rungs. Two sources feed it. A hand-checked seed set of **36 questions across 3 example subjects** validates the schema. Separately, the study runs use **1200 items built automatically from TOFU's own perturbed splits** (`data/breadth_items.json`) — 400 each on B0, B1 and R. That is what produced the calibration in `RESULTS.md` §2.

**What does not exist.** Rungs **B2 (nicknames), B3 (one-step consequences) and B4 (two-step reasoning) have no questions at all** — they are declared in the code and empty. They cannot be generated from TOFU and must be written by hand. These are exactly the rungs where deep-but-narrow forgetting would show up, so this is a scientific gap, not a cosmetic one.

**Where the questions come from.** The three subjects are genuine TOFU authors. TOFU supplies the B0 questions for free (its own 20 questions per author are, by definition, the exact-wording rung) and supplies most of the R material. But B1 through B4 do not exist anywhere and must be written by hand.

**Why writing them is slow.** A B3 question must give the fact away *without naming the person*. A B4 question needs a genuine two-step chain through facts the model still holds. Get it slightly wrong and you are measuring something else. This is careful work, not bulk work — which is precisely why the interesting rungs are under-studied in the field, and why our question sits in a real gap.

---

### 8.4 Surface measurements — 🟢 Ready

**Plain meaning.** "Surface" means we look only at what the model says out loud.

Two measurements:

- **Mention rate** — how often does the forbidden name appear in the model's answers? We also watch for nicknames, since saying "Yun-Hwa" leaks just as much as the full name.
- **A softer measure** — even when the model does not say the name, it may still be leaning towards it. This captures that leaning.

**An important correction.** Our original project proposal described the second measurement as looking *inside* the model. **That was wrong.** It is calculated purely from the model's output. It is a gentler surface measurement, not an internal one.

This matters a great deal, because our entire research question is whether surface and internal measurements can disagree. We now keep them in **separate files** so the two can never be quietly confused.

---

### 8.5 Internal diagnostics — 🟢 Ready

**What they do.** Look at the model's internal signals rather than its words.

- **Detector test** — can a simple detector still tell, from the internal signals alone, that the model is thinking about the forbidden topic?
- **Change measurement** — how far did the internal signals move compared to before?
- **Selectivity** — did the forbidden topic's signals move *more* than unrelated topics' signals? If everything moved equally, we damaged the model rather than targeting anything.

**Honest limitation.** These show *association*, not *cause*. They tell us something changed at the same time as unlearning, not that the change caused the forgetting. This is why we also built the next component.

---

### 8.6 Signal swapping (activation patching) — 🟢 Built and proven on real models

**This is the most important piece of recent work.**

**Plain meaning.** We temporarily replace part of the model's internal signals with signals taken from a *different* model, then let it finish thinking and see whether the answer changes.

**An everyday analogy.** You have two calculators, one broken. To find which part is faulty, you swap a component from the working one into the broken one and see if it starts working again. If swapping that part fixes it, that part was the problem.

**Why this is better than everything above.** It is a genuine experiment, not an observation. We *intervene* and watch what happens. That is how you establish cause and effect.

**Why it matters for our question.** If we take a supposedly-forgetful model, insert its signals into a model that knows everything, and the knowledge comes flooding back — then the forgetful model never really forgot. It was hiding.

**How it works mechanically.** The model processes information through a stack of layers, one after another. Our code:

1. Runs the source model and records its signals at a chosen layer
2. Runs the target model, and at that exact layer, swaps in the recorded signals
3. Lets the target model finish normally
4. Compares the answer to what it would have said otherwise

We do this one layer at a time, so we can see *where* in the stack the knowledge lives.

**Status.** Built, heavily tested, and since run on the real Llama-3.2 TOFU checkpoints at 1B and 3B. Two exact end-to-end checks hold on real models: patching a model with its own signals changes nothing at all, and patching in the "never learned it" model's signals gives exactly the maximum score.

---

### 8.7 The depth score (UDS) — 🟢 Built; reproduces the published numbers, not yet checked against the authors' code

**What it is.** A published measurement that turns signal swapping into a single number between 0 and 1.

- **1** = the knowledge is gone, as thoroughly as in a model that never learned it
- **0** = the knowledge is completely intact, merely hidden

**How it works, in two stages.**

**Stage 1 — find where the knowledge lives.**
Take the "never learned it" model. Insert its signals into the "knows everything" model, one layer at a time. Wherever this causes a big drop in the model's confidence, that layer holds part of the knowledge. Layers where nothing much happens are just noise and are ignored.

*Result: a shortlist of layers that actually matter.*

**Stage 2 — check the model under test.**
Repeat exactly the same procedure, but using signals from the unlearned model.

- If it behaves like the "never learned it" model → the knowledge is genuinely gone
- If nothing much changes → the knowledge is still sitting there

**Combining into one number.** For each important layer we work out what fraction of the knowledge was removed, then average across layers — giving more weight to the layers that held more knowledge in the first place.

**A deliberate honesty feature.** If Stage 1 finds no layer that holds the knowledge, the result is recorded as **"undefined"** — not as zero. Zero would mean "we measured, and nothing was erased". Undefined means "we could not measure". Confusing those two would be misleading, so the code refuses to.

**Status: reproduces the published numbers, but still NOT checked against the original authors' code.**

We implemented this from the equations in the published paper, and we wrote a detailed point-by-point comparison confirming every equation matches. We have since reproduced the paper's own published table across seven runs at two model sizes, with an average miss of 0.010 against an allowance of 0.08.

That is necessary evidence, but not sufficient. Two teams can both follow the same equations correctly and still get different numbers, because of small choices about how text is split up or how averages are taken — and we have direct evidence that such choices matter here, because switching from our own word-picking heuristic to the authors' hand-marked spans moved the middle values by about 0.03 to 0.05.

Until we compare directly against the authors' own code on the same inputs, the numbers cannot carry a research claim. **The software knows this about itself** — every result it produces is stamped *"NOT cross-validated vs reference"*. This is task T0, and it is now unblocked: the authors' code is vendored at `reference_uds/` and their annotations are byte-identical to the published dataset.

**One thing we got wrong and fixed.** Our first version of this score was much simpler — it just combined three numbers the user supplied. When we read the original paper properly, we found the real measurement is quite different: layer by layer, question by question, with a selection step and a weighting step. Ours had none of that structure. We rebuilt it properly, and marked the old version obsolete rather than quietly patching it, because it was not a rough version of the right thing — it was a different thing.

---

### 8.8 Results storage and graphs — 🟢 Ready

**What it does.** Saves every measurement to a file, and draws the breadth-against-depth graph described in Section 6.

**One deliberate safeguard.** Occasionally a measurement lands in a range where it cannot be trusted. Rather than silently dropping those points, the software reports the result **both with and without** them, and flags whether the conclusion changes depending on which you use.

Quietly discarding awkward data points is one of the easiest ways to fool yourself, so the code makes that impossible.

---

### 8.9 Tests — 🟢 Ready

Two safety nets.

**180 automated tests.** Small checks that each piece behaves correctly. They run in about 16 seconds.

**A 31-check plumbing test.** Runs the entire system from start to finish on the tiny artificial model, confirming all the pieces connect.

> **"Plumbing test" means: does water flow through the pipes?** It checks the system runs. It says nothing about whether the answers mean anything. We renamed it from "smoke test" specifically to stop anyone mistaking its output for results.

---

## 9. How to run the software today

Anyone with a normal computer can reproduce everything below. No special hardware needed.

### One-time setup

```bash
conda create -y -n deeperase python=3.11
conda activate deeperase
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Running it

```bash
# Run all 491 tests            (~35 seconds)
python -m pytest tests/

# Run the whole system end to end  (~24 seconds)
python -m deeperase.scripts.smoke_e2e

# Rebuild the 36 example questions
python -m deeperase.probes.seed_tofu
```

### What you should see

```
180 passed

PLUMBING TEST: 31/31 checks passed
```

### Exact versions being used

Python 3.11.15 · torch 2.6.0 (processor-only) · numpy 1.26.4 · scipy 1.13.1 · scikit-learn 1.5.2 · pandas 2.2.3 · matplotlib 3.9.2 · transformers 4.46.3 · pytest 8.3.4

**One warning:** do not install this into the default Anaconda environment. That causes the crash described in Section 11.

---

## 10. What has been tested

### The numbers

Both commands were run immediately before writing this document.

| Test | Result |
|---|---|
| Automated tests | **491 passed** |
| End-to-end plumbing test | **31 / 31 checks passed** |

The 491 tests split as: 84 for signal swapping, 84 for the depth-score maths and the trajectory guards, 58 for the TOFU data handling, 53 for the unlearning methods and the stopping rule, 50 for run configuration and memory planning, 42 for the control dial, 39 for model loading, 37 for the breadth axis, 23 for the reference span annotations, and 21 for the experiment runner.

### What the tests DO prove

**The maths is right.** Wherever the correct answer is known in advance, the software produces it. Two examples that hold exactly, on real (if tiny) models:

- Swapping a model's signals with **its own** signals changes nothing at all — as it must.
- When the "unlearned" model is literally the "never learned it" model, the depth score comes out at exactly 1.0 — perfect forgetting, as it must.

**The signal swapping is real, not faked.** This deserves explaining, because it is the difference between a working instrument and a convincing-looking dummy.

We wrote tests specifically designed to fail if the software were quietly copying numbers around instead of genuinely running the model. The strongest one: we deliberately damaged a part of the model *after* the swap point. The result changed. That can only happen if the model truly computed the answer itself.

**Nothing is left behind.** To observe and modify a model, the software temporarily attaches itself to it. If those attachments were ever left in place, every later measurement would be silently corrupted. Tests confirm they are always removed — even when something goes wrong midway.

**The tests themselves are trustworthy.** A test suite that cannot fail is worthless. So we deliberately broke the code in nine different ways — swapping two settings, disabling a threshold, removing a safety limit, and so on — and confirmed the tests caught all nine.

### What the tests DO NOT prove

This is the important half.

| Not proven | Why |
|---|---|
| **That our depth score matches the original authors'** | We followed their published equations, but have never compared actual numbers with their code |
| **That anything works on a real model** | Every test uses a tiny artificial model with random settings and no knowledge |
| **Anything about the research question** | The tests check the instrument, not the world |
| **That the numbers produced so far mean anything** | They do not — see below |

### About the numbers produced so far

The plumbing test produces a graph and a set of numbers. **They are meaningless as measurements.**

The test model is deliberately tiny — roughly a millionth the size of a real one — with randomly chosen settings and a vocabulary of 256 word-pieces. It does not know English. It has never learned anything, so there is nothing for it to forget.

On this model both measurements come out completely flat, and the software correctly reports that no relationship can be calculated.

**We report that flatness openly rather than hiding it.** An earlier version of this project drew a neat downward curve here that looked exactly like our hypothesis. Those numbers had been typed in by hand. That has been removed — see Section 11.

---

## 11. Quality problems found and fixed

Recorded briefly and openly. Hiding them would make everything else in this document less trustworthy.

| What was wrong | How it was found | How it was fixed |
|---|---|---|
| **Tests crashed on another computer.** Looked like a bug in our maths. | Independent review | Not our bug — the reviewer's setup had two conflicting copies of a background component. We built a fully locked-down setup where this cannot happen. We deliberately did **not** use the well-known one-line override for this error: its manufacturer documents it as unsafe and warns it can produce silently wrong numbers. In a measuring instrument, a wrong number that looks right is worse than a crash. |
| **A graph drawn with made-up numbers.** It showed a neat trend matching our hypothesis, while the text described it as a real end-to-end result. | Independent review | Removed. The graph now uses only numbers the software actually computed — which turn out to be flat and uninformative. Invented numbers now exist in exactly one place: a single clearly-labelled test that checks the drawing code works. |
| **Internal bookkeeping values were being altered** when they should have been left alone. | Our own follow-up check | Fixed by identifying them by name as well as by type. Eight new tests cover it. Lesson: a comment saying the code does something is not evidence that it does. |
| **Our first depth score was not the published measurement.** It was much simpler and lacked the layer-by-layer structure entirely. | Reading the original paper properly before building the real version | Rebuilt correctly. The old version is marked obsolete and warns if used. |
| **A "finding" that was not a finding.** An earlier report described a quirk of the tiny test model as a research result worth publishing. | Our own review | Corrected to what it was: a technical edge case in software testing. The properly built version handles the situation automatically. |
| **SAGE claimed to be validated when it was not.** | Our own review | Corrected. See below. |

### About SAGE

SAGE was a prototype for a cleverer version of the control dial. The maths runs correctly, but there is an unsolved conceptual problem underneath it: it needs to connect a pattern in the model's *internal signals* to a direction in the model's *dial settings*. Those are different kinds of thing, and we have no principled way to translate between them.

I had also previously claimed a particular test showed SAGE was sound. It did not — that test only shows the code is internally consistent, which catches typing mistakes but says nothing about whether the idea is meaningful.

SAGE is now **set aside**, as you instructed. The code warns loudly if anyone runs it.

---

## 12. Honest current status

### Ready and trustworthy

- The setup reproduces exactly on any computer
- The control dial is complete and faithful to the published method
- The question ladder structure is complete
- Surface measurements are complete
- Internal diagnostics are complete
- Signal swapping is complete, including tests designed to catch faking
- 180 automated tests and 31 plumbing checks pass

### Not ready for research

| Item | Why not |
|---|---|
| **The depth score** | Never compared against the original authors' code |
| **Everything, on real models** | Only ever run on a tiny artificial model |
| **The question set** | Only 3 subjects out of the 50–200 needed |
| **Unusual model types** | Supported in principle, never tested |

### Why no research conclusion is possible yet

Three independent reasons, any one of which would be enough on its own:

1. **The depth measurement is unverified.** It is the heart of the study. Until it agrees with the original authors' code, any number it produces could be wrong in a way we cannot detect.
2. **Nothing has run on a real model.** Our test model has no knowledge, so there is nothing to forget.
3. **No experiment has been run.** We built the instrument. We have not used it.

> **In one sentence:** we have built a measuring device and checked carefully that it is well made, but we have not measured anything with it, and we have not yet confirmed it agrees with the standard device.

---

## 13. What happens next — detailed plan

Work is organised into three stages. Stage A can start immediately. Stage B needs a powerful computer. Stage C is the actual research.

---

### STAGE A — can be done now, on an ordinary computer

**Purpose:** finish everything possible without special hardware, so that the moment a powerful computer becomes available, the important work can start immediately rather than being delayed by preparation.

**Estimated effort: one to two weeks.**

#### A0 — Confirm the models are downloadable *(half a day)*

Open the TOFU download page and verify that both the "knows everything" and "never learned it" models are actually published, and in which sizes.

**Why first:** everything else assumes these exist. If they do not, the plan changes fundamentally. This is a cheap check that removes a large risk.

#### A1 — Automatic answer-part detection *(3–4 days)*

Right now, a person must manually mark which words in an answer are the actual fact. The published method scores only those words, ignoring filler like "The answer is".

**Why it matters:** filler words are predictable whether or not the model remembers anything. Including them dilutes the measurement and makes real forgetting look smaller than it is.

**Depends on:** nothing.

#### A2 — Settle a processing difference *(1–2 days)*

We currently process several questions together as a group; the original authors process them one at a time. This may change results slightly.

**Why it matters:** if we want to compare numbers with the authors' code in Stage B, both sides must work the same way. Settling it now avoids discovering a mismatch later and having to redo everything.

#### A3 — Test other model families *(2–3 days)*

Our code supports several families of AI model, but only one is tested. The others print a warning when used.

**Why it matters:** we may want to test more than one kind of model. Better to know now whether the code handles them.

#### A4 — Speed improvement *(1 day)*

Part of the calculation is repeated unnecessarily. Fixing this roughly halves the running time.

**Why it matters:** Stage C involves hundreds of measurements. Halving the cost is worth one day of work.

---

### STAGE B — requires a powerful computer

**Purpose:** prove the depth measurement is trustworthy. Until this is done, no result from this project can be believed.

**Estimated effort: two to three weeks after computer access begins.**

#### B1 — Compare against the original authors' code ⭐ *(1 week)*

**The single most important remaining task.**

Download the original authors' code. Run both their version and ours on exactly the same models and questions. Compare the numbers layer by layer. They should agree to about four decimal places.

**If they agree:** our measurement is trustworthy, and we can mark it verified.
**If they disagree:** find out why and fix it. Better to discover this now than after building conclusions on it.

**Depends on:** A0, A1, A2, and computer access.

#### B2 — Reproduce the authors' published check *(3–4 days)*

The authors published a specific result their method should produce: as you feed it models that have seen progressively less of the forgotten material, the score should rise steadily, reaching exactly 1.0 for a model that never saw it at all.

Running the same check on our version is an independent confirmation that ours behaves correctly end to end.

**Depends on:** B1.

#### B3 — First run on a real model *(2–3 days)*

The first time any part of this project touches a model with genuine knowledge.

**Depends on:** B1.

#### B4 — Check one inherited setting *(2 days)*

One threshold value was copied from the paper and never tested on our own data. We should confirm the results do not change much if it is adjusted.

**Why it matters:** if a conclusion depends heavily on an arbitrary setting, it is not a real conclusion.

---

### STAGE C — the actual research

**Only begins after B1, B2 and B3 have passed.**

**Estimated effort: four to six weeks.**

#### C1 — Expand the question set *(1–2 weeks)*

Grow from 3 subjects to 50 or more. For each subject: keep TOFU's existing questions as the B0 rung, then write roughly eight more climbing the ladder to B4, plus retain questions.

**How:** use AI to draft candidates, using our 36 hand-written questions as worked examples of the required quality — then **have a human check every single one** before it is used. No unchecked question enters a result.

**Why it takes this long:** the B3 and B4 rungs are genuinely difficult to write well, as explained in Section 8.3.

#### C2 — Create the unlearned models *(1 week)*

Run four or five published unlearning methods on the "knows everything" model. Each run takes one to three hours of computer time.

#### C3 — The main experiment *(1–2 weeks)*

For each unlearning method:

- turn the dial to eleven settings (free)
- measure breadth at each setting
- measure depth at each setting
- measure general ability at each setting, as a safety check

Then repeat across several models to confirm any pattern is not a one-off.

#### C4 — Analysis and write-up *(1–2 weeks)*

Plot breadth against depth. Determine whether they trade off.

**Report the outcome honestly whichever way it turns out.** An outcome showing the two measurements agree is just as worth reporting as one showing they conflict, because the field currently assumes agreement without having checked.

---

### Summary timeline

| Stage | Work | Time | Blocked by |
|---|---|---|---|
| **A** | Finish preparation | 1–2 weeks | Nothing — can start now |
| **B** | Prove the measurement is trustworthy | 2–3 weeks | **Computer access** |
| **C** | Run the actual study | 4–6 weeks | Stage B passing |

**Total after computer access: roughly two to three months.**

**The critical dependency:** Stage B cannot start without a powerful computer, and Stage C cannot start without Stage B. So the date computer access begins effectively sets the date the project finishes.

---

## 14. Why a powerful computer is needed

### What a GPU is

A **GPU** is a specialised processor. An ordinary processor does a few things very quickly, one after another. A GPU does thousands of things at the same time. AI models need exactly that kind of calculation.

### What we have

The development computer has **no GPU** — only an ordinary processor.

### Why that was fine until now

That was a deliberate choice. Writing the software, testing it, and running the tiny model all work perfectly well on an ordinary processor.

This was the right way to work: get the software correct cheaply first, then spend expensive computer time only when there is something worth running. All 491 tests and the full end-to-end check were completed without spending a penny on computing.

### Why it is now the blocker

Every remaining important task involves real models.

| Task | Ordinary processor | With a GPU |
|---|---|---|
| Load a real model | Barely possible; may not fit in memory | Fine |
| One measurement | Hours | Seconds |
| The full study | Months — not realistic | Days |

The clearest case is the comparison against the authors' code. Their software requires real models at a scale our machine simply cannot hold in memory.

### What is needed

| Resource | Requirement |
|---|---|
| GPU memory | At least 40 GB (types commonly called A100 or A6000) |
| Storage | About 500 GB, for models and data |
| Total GPU time | About 600 hours — roughly four weeks on one machine |

---

## 15. The one decision needed from you

I have deliberately not chosen this myself, because it changes what the results mean.

### The background

Our control dial works by measuring how much a model's internal settings changed during unlearning, then pushing further in that direction.

There are two ways unlearning can change a model:

**Option 1 — change everything.** Adjust all of the model's internal settings. This is what the researchers who invented the dial did. Thorough, but needs a great deal of computer memory.

**Option 2 — change a small add-on.** Leave the main model untouched and attach a small extra piece, adjusting only that. This technique is very widely used because it needs far less memory.

### The problem

The dial technique has only ever been tested with Option 1. Nobody has published whether it still behaves correctly with Option 2.

Our likely memory budget pushes us toward Option 2. Scientific caution pushes us toward Option 1.

### The three choices

| | Approach | Advantage | Disadvantage |
|---|---|---|---|
| **A** | Change everything | Matches the published method exactly; results directly comparable | Needs a lot of memory; limits how many models we can test |
| **B** | Use the small add-on | Fits our likely hardware; lets us test more models | **Untested territory** — no published evidence the dial works this way |
| **C** | **Do both and compare** | Turns the problem into a useful contribution; nobody has published this comparison | Roughly doubles part of the work |

### My recommendation: option C

Three reasons:

1. **The comparison is genuinely useful.** Anyone with a single GPU will face this same choice, and there is currently no published answer.
2. **It costs less than it sounds.** Only the initial unlearning runs double. The dial settings stay free, and they are the bulk of the measurements.
3. **There is a safe fallback.** If time or computer access runs short, we drop to option A and honestly report B as untested.

**The software already supports all three.** This is purely a decision about what to run, not about what to build.

---

## 16. Questions, and what we can report

### Questions for you

**1. Computer access — the most urgent.**
Everything that can be done without a GPU is either finished or small. The most important remaining task cannot start without one. **Can we get access to a suitable GPU, and roughly when?** The answer effectively sets the project timeline.

**2. The decision in Section 15.**
**Do you agree with option C (do both and compare)?** If resources are tight, I would fall back to option A and record option B as untested.

**3. Sequencing.**
My plan is to complete the five Stage A tasks now, so the important comparison can begin the moment a computer is available. **Does that ordering seem right?**

**4. SAGE.**
It is set aside, and there is a real unsolved problem in it. **Should it stay out permanently, or be revisited if time allows?** My suggestion: leave it out. The measurement work matters more.

### What we can safely report this week

> **Progress.** We have built and tested the complete measuring system for this project. It includes a controllable way to vary forgetting pressure without retraining, a structured ladder of test questions, output-based measurements, internal diagnostics, and a genuine cause-and-effect technique that swaps signals between models.

> **Verification.** 180 automated tests pass, and an end-to-end run passes all 31 checks. The setup is fully documented so anyone can reproduce it. We deliberately introduced nine bugs to confirm the tests would catch them; all nine were caught.

> **Quality control.** During this work we found and fixed one real bug in our own code, removed a graph that had been drawn with hand-typed numbers, and discovered that our first version of the depth measurement did not match the published method. All three are documented openly.

> **Validation.** The depth measurement now runs on real AI models and reproduces the numbers published by the researchers who invented it — at two model sizes and across six different settings, every one passing. The two extreme cases come out exactly right.

> **Honest limitation.** This is a working, checked instrument, **not a research result**. We have not yet run the study the instrument was built for. One validation gap also remains: we have reproduced the authors' published *numbers*, but have not run their *code* alongside ours on identical inputs. Until we do, we cannot rule out a methodological difference — and we can see one small sign of exactly that (§10).

> **Next step.** Begin the actual study: measuring whether forgetting more broadly makes forgetting less deep.

**You can safely say:** that the measurement tool is built, tested, and reproduces published reference values.

**Please avoid saying:** that we have findings about the research question, or that the measurement is fully verified against the original implementation. Neither is true yet.

---

## 17. Glossary

| Term | Plain explanation |
|---|---|
| **Activation patching** | Temporarily replacing a model's internal signals with signals from another model, to see whether the answer changes. A way of testing cause and effect inside a model. |
| **Alpha** | The setting on our control dial. 0 means no extra push; higher means push harder to forget. |
| **Benchmark** | A standard set of test questions everyone uses, so different pieces of research can be compared. |
| **Breadth** | How many different ways of asking are covered by the forgetting. |
| **Checkpoint** | A saved copy of a model at a point in time, like saving a document. |
| **CPU** | The ordinary processor in a computer. Fine for general work, far too slow for real AI work. |
| **Depth** | Whether the knowledge actually changed inside the model, rather than just being hidden. |
| **Forget set** | The specific information we are trying to remove. |
| **GPU** | A specialised processor that performs thousands of calculations at once. Required for real AI work. |
| **Layer** | AI models process information through a stack of stages, one after another. Each stage is a layer. |
| **LLM (large language model)** | A large AI system trained on text that can read and write language. ChatGPT is one. |
| **LoRA** | A memory-saving technique: instead of adjusting a whole model, attach a small extra piece and adjust only that. |
| **Parameters** | The billions of adjustable numbers inside a model. Training sets them. Also called weights. |
| **Plumbing test** | A test that checks the software runs from start to finish. It checks the pipes carry water — not that the water is good. |
| **Retain set** | Information the model should keep. The opposite of the forget set. |
| **SAGE** | Our prototype for a cleverer dial. Set aside — there is an unsolved problem in it. |
| **Surface measurement** | A measurement based only on what the model says out loud. |
| **TOFU** | A standard test collection of 200 invented author biographies, used to study forgetting safely without involving any real person's data. |
| **UDS (Unlearning Depth Score)** | A published measurement of how deeply something was forgotten. 1 = gone, 0 = still there. |
| **UIPE** | The published method behind our dial: push further in the direction unlearning already moved. |
| **Unlearning** | Changing a trained model so it behaves as if it had never seen certain information. |

---

## 18. File map

### Documents

| File | What it is |
|---|---|
| **`COMPLETE_PROJECT_SUMMARY.md`** | **This file.** Everything, in plain English. |
| `README.md` | Technical setup instructions and status, for a programmer. |
| `VERIFICATION.md` | Raw captured output of the test runs, kept as evidence. |
| `docs/UDS_CONFORMANCE.md` | Point-by-point comparison of our depth score against the published definition, and exactly what remains unverified. |
| `RESEARCH_PLAN.md` | The reasoning behind the research direction chosen. |
| `DeepErase_Domain_and_Literature_Report.md` | Review of the research field, comparing 38 published papers. |
| `literature/` | 54 research papers, sorted by topic, with reading notes. |

### The software

| File | What it does |
|---|---|
| `deeperase/core/extrapolation.py` | The control dial. Also holds the set-aside SAGE prototype. |
| `deeperase/eval/patching.py` | Swaps internal signals between models. |
| `deeperase/eval/uds.py` | The depth score, built on signal swapping. |
| `deeperase/eval/depth.py` | Internal diagnostics. Also holds the obsolete old depth score. |
| `deeperase/eval/surface.py` | Output-based measurements. |
| `deeperase/eval/plane.py` | Stores results and draws the breadth-against-depth graph. |
| `deeperase/probes/schema.py` | The question ladder structure and its scoring. |
| `deeperase/probes/seed_tofu.py` | The 36 hand-written, hand-checked example questions. |
| `deeperase/scripts/smoke_e2e.py` | The end-to-end plumbing test. |

### Tests and setup

| File | What it is |
|---|---|
| `tests/test_extrapolation.py` | 42 tests for the control dial |
| `tests/test_metrics.py` | 76 tests for the measurements |
| `tests/test_patching.py` | 62 tests for signal swapping and the depth score |
| `requirements.txt` | Exact software versions needed |
| `pyproject.toml` | Project configuration |
| `environment.yml` | Instructions for creating the working environment |

### Output — test output only, not results

| File | What it is |
|---|---|
| `results/figures/toy_plumbing_plane.png` | Graph from the tiny test model. **Not a research result.** |
| `results/metrics/toy_plumbing_plane.json` | The numbers behind that graph. **Not research data.** |
| `data/probes/seed_tofu.json` | The 36 example questions. |

---

*End of summary. All development work remains paused.*
