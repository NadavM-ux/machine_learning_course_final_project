# Prompt for the LLM (Claude Opus) — Iranian target-population classification

> Paste this whole block into Claude Opus (a frontier LLM >= ChatGPT-5.2, allowed by the PDF's "or a higher model"), then paste the
> rows of `llm_test_set_no_answers.csv` right after it. Run it **10 times** and
> save each run's answer as `Iteration_N/chatgpt_predictions_iteration_N.csv`.

---

You are an expert social-media analyst. You will classify X (Twitter) users
against a target population: **Iranians**. For each user you are given only
profile fields (username, display name, bio/description with an English
translation, location, follower/following/status counts, join date).

Classify **each** user on **three** independent tasks by following the decision
flowcharts below **exactly, node by node**. Do not use outside knowledge about
specific accounts — decide only from the fields provided. When a flowchart ends
in "Unknown", output the unknown code. Never guess.

## Task 1 — target_population  (codes: 1 = target, 0 = not_target, 2 = unknown)
1. Is the profile suspended / empty / protected (no usable info)? → **Yes: 2 (unknown)**. No → 2.
2. Do the bio, display name, or username mention Iran / Iranian / Persian / Persia, or an Iranian city? → Yes → 3. No → 4.
3. Is there a clear NON-Iranian identity (e.g. "Persian cat lover", a foreign journalist merely covering Iran)? → **Yes: 0 (not_target)**. **No: 1 (target)**.
4. Does the location field indicate an Iranian city? → **Yes: 1 (target)**. No → 5.
5. Do personal posts (daily life, family, places — not news/politics) indicate the person is Iranian? → **Yes: 1 (target)**. **No: 2 (unknown)**.

## Task 2 — locals_vs_diaspora  (codes: 1 = local, 0 = diaspora, 2 = unknown)
*Only meaningful when Task 1 = 1 (target). If Task 1 ≠ 1, output **2**.*
1. Does the location field show an Iranian city? → **Yes: 1 (local)**. No → 2.
2. Does the location field show a city OUTSIDE Iran? → **Yes: 0 (diaspora)**. No → 3.
3. Do recent tweets clearly place the user IN Iran? → **Yes: 1 (local)**. No → 4.
4. Do recent tweets clearly place the user OUTSIDE Iran? → **Yes: 0 (diaspora)**. No → 5.
5. Does the bio / display name suggest diaspora (e.g. "Iranian-American", "Persian in Berlin", dual flags)? → **Yes: 0 (diaspora)**. **No: 2 (unknown)**.

## Task 3 — person_vs_organization  (codes: 1 = person, 0 = organization, 2 = unknown)
1. Is the profile suspended / empty / private? → **Yes: 2 (unknown)**. No → 2.
2. Does the display name / username clearly indicate an organization (News, Agency, Ministry, Foundation)? → **Yes: 0 (organization)**. No → 3.
3. Does the bio describe an institution ("Official account of…", "Established 1979", an office address)? → **Yes: 0 (organization)**. No → 4.
4. Is the profile picture a logo / brand mark / non-human image **AND** are tweets written in a plural / institutional voice ("we", "our team")? *(both must hold)* → **Yes: 0 (organization)**. No → 5.
5. Is there clear evidence this is one individual (a face photo, a personal story, first-person singular)? → **Yes: 1 (person)**. **No: 2 (unknown)**.

## Output format
Return **only** a CSV, no prose, with exactly these columns and one row per input user:

```
username,target_population,locals_vs_diaspora,person_vs_organization
```

Values must be the integer codes above (0, 1, or 2). Keep the input order.
