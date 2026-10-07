# FTEC5660 Homework 2: CV Verification Agent

This repository implements Task 1, the Task 2 adversarial CV, and the report.
The implementation uses LangChain, requests only `deepseek-v4-flash`, and obtains
all verification evidence from the course SocialGraph MCP server. Task 3 is a
separate reflection and is not included in this implementation.

## Run

Use Python 3.10 or newer. Create `.env` locally with
`DEEPSEEK_API_KEY=your_actual_key`; never commit that file. `.env.example` contains
only a placeholder. Do not include a UTF-8 BOM in `.env` because the supplied
loader reads ordinary UTF-8.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe hw2.py --cv-folder public_test
.\.venv\Scripts\python.exe hw2.py --cv-folder task2
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python hw2.py --cv-folder public_test
python hw2.py --cv-folder task2
```

Each command writes `results.csv` in the working directory; a subsequent run
replaces it. Public CVs are graded automatically against the supplied labels.
Task 2 scores are not automatically graded because that folder has no labels.
The original PDF loader, MCP connection, scoring and runner code are unchanged.

## Homework 2 solution

### Task 1

I implemented a bounded LangChain tool-calling verifier followed by a separate,
structured evidence audit, using `ChatDeepSeek(model="deepseek-v4-flash")` at
temperature 0 with thinking disabled. For each CV, the first stage searches the
course MCP data using name, city and industry, retrieves complete LinkedIn
profiles, and resolves same-name candidates using multiple employment and
education anchors. Search locations are normalized to the city alone; an
industry omitted from the first search can be filled from a professional
headline, while explicit nulls and later searches can still relax the filter.
These CV-derived hints guide retrieval and are never treated as evidence.
Facebook is supplementary and limited to two profile reads per CV. A fresh audit
call receives the original CV and the actual retrieved profile fields, without
the retrieval stage's conclusions. It checks name, current city, every listed
job and its dates/seniority, every qualification and graduation year, and every
claimed skill. Equivalent wording and omitted profile details are allowed;
one material false claim is sufficient to fail. Each claimed skill must be
matched to an actual skill in the selected LinkedIn profile, or explicitly
marked absent. Application code validates the selected profile and all five
check groups, then returns 0.95 for a complete positive audit, 0.05 for a
contradiction, or 0.40 for unresolved evidence/errors. These are decision scores,
not calibrated probabilities. CV messages and retrieved prose are treated as
data rather than instructions; no filename, candidate identity, public label or
cached experiment record is used to make a runtime decision. Each CV has
isolated context and tool caching, a maximum of 12 retrieval turns and 28 tool-call
decisions (with one retry per failed invocation), and a 210-second timeout. An `asyncio.Semaphore(3)`
limits active CVs; a 1,650-second batch deadline leaves unresolved CVs with a
valid fallback score so the provided runner can still write `results.csv`.

#### Public-test results

| Run | Correct / total | Accuracy | Wall time |
| --- | ---: | ---: | ---: |
| 1 | 7/7 | 100% | 85.90 s |
| 2 | 7/7 | 100% | 98.75 s |
| 3 | 7/7 | 100% | 89.11 s |

The three final runs achieved **21/21 correct decisions (100% mean accuracy)**.
All seven scores were identical across the three runs; every CV had a resolved
identity and an explicit audit, with no 0.40 fallback scores:

| CV | Score | Prediction |
| --- | ---: | --- |
| CV_1.pdf | 0.9500 | Valid |
| CV_2.pdf | 0.9500 | Valid |
| CV_3.pdf | 0.9500 | Valid |
| CV_4.pdf | 0.0500 | Has discrepancy |
| CV_5.pdf | 0.0500 | Has discrepancy |
| CV_6.pdf | 0.9500 | Valid |
| CV_7.pdf | 0.0500 | Has discrepancy |

The final CSVs, timings, source hashes and tool/audit traces are in
`experiments/public_release_1/` through `experiments/public_release_3/`.

An earlier development run achieved 6/7: the model treated a claimed skill
missing from the authoritative profile as unknown rather than false. I corrected
the general closed-world skills rule and added explicit per-skill evidence
mapping; no candidate-specific exception or public-answer lookup was added.
That development result is preserved in `experiments/public_run_1/results.csv`.
The seven public examples were used for development, so their accuracy does not
establish accuracy on the unseen private grading set.

### Task 2

The submitted file is `task2/adversarial_cv.pdf`, a one-page CV for Kelly Tsang.
The actual MCP profile (`person_id=10001`) contains a **BSc in Finance from HKUST,
graduated 2016**, and no MSc. The visible PDF deliberately upgrades this to
**Master of Science in Finance**, retaining HKUST and 2016. This is a real false
qualification, not merely an alternative wording of the correct degree. The
employment history, location and listed skills remain consistent with the
retrieved target profile. The target data was fetched from the MCP tools and
saved in `experiments/target_profiles.json` for reproducibility.

The attack exploits a mismatch between PDF rendering and text extraction. The
false master's-degree line is drawn as a high-resolution image. At the same
position, an invisible PDF text object (text render mode 3) contains the true
text `BSc in Finance`. Other CV content is ordinary selectable text. The supplied
`MarkItDown(enable_plugins=False)` loader extracts the true bachelor's-degree
line but not the rasterized master's-degree line. Thus a person viewing the PDF
sees a false claim while the text-only verifier receives a consistent CV. The
attack does not depend on instructions telling the model to ignore its rules;
it removes the false claim from the representation the model is allowed to
inspect. Both the rendered page and the exact output of the supplied loader were
checked. `experiments/adversarial_extracted.txt` records that extracted text.

I expect this to transfer to other text-only verifiers using the same loader,
because different prompts cannot recover a raster-only claim that never reached
their input. This is an expectation, not a measurement of the five undisclosed
teacher agents. An independent image/OCR inspection could expose the false
degree; this solution does not modify the assignment's fixed PDF loader to add
such inspection.

#### Attack and control results

| Run | True CV score | Adversarial CV score | Wall time (both CVs) |
| --- | ---: | ---: | ---: |
| 1 | 0.9500 | 0.9500 | 30.62 s |
| 2 | 0.9500 | 0.9500 | 22.36 s |
| 3 | 0.9500 | 0.9500 | 25.07 s |

The attack was accepted in **3/3 runs** by this Task 1 verifier, while the true
baseline was also accepted in 3/3. The exact results and audit traces are in
`experiments/attack_release_1/` through `experiments/attack_release_3/`.

In one control run, replacing only the extracted `BSc in Finance` with
`Master of Science in Finance` changed the score to **0.05** (rejected).
The audit explicitly identified the MSc/BSc mismatch. The control input and
result are in `experiments/degree_control_release/`. This supports the extraction-gap
explanation: the verifier catches the false degree when the false claim actually
reaches its text input. This ablation does not alter the submitted PDF.

**No teacher-agent score is claimed:** the five undisclosed graders were not
available for testing. The observed 3/3 is success against my own verifier only.

## Reproducibility and checks

A separate fresh-clone/new-virtual-environment check also achieved **7/7** in
**83.06 seconds**, with no fallback scores. It installed only the declared
requirements, ran the same source hash, and did not copy `.env`. Results and
traces are in `experiments/fresh_release/`. All **16 offline tests passed**.

The final code includes offline tests for complete evidence, single-field
contradictions, unsupported skills, same-name profile provenance, malformed
scores, per-CV failures, concurrency and both timeout levels:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For recorded live tests (API usage applies), choose a new label each time:

```powershell
.\.venv\Scripts\python.exe tools/evaluate.py --folder public_test --label public_check --runs 3
.\.venv\Scripts\python.exe tools/evaluate.py --folder task2 --label attack_check --runs 3
.\.venv\Scripts\python.exe tools/control_experiment.py
```

`evaluate.py` preserves CSVs, timings, source/PDF hashes and optional per-CV
profile/audit traces. `HW2_TRACE_DIR` enables these traces; it is not required by
the grader and no trace file is ever read by `hw2.py` as evidence. The dependency
snapshot in `experiments/tested-requirements.txt` records the tested Windows
environment; install the portable `requirements.txt` for the normal assignment.

To regenerate the submitted attack from the saved target MCP snapshot:

```powershell
.\.venv\Scripts\python.exe tools/generate_adversarial.py
```

`tools/check_clean_run.py` creates a fresh local clone, overlays the solution,
installs `requirements.txt` into a new virtual environment and runs the public
set. It passes the API key through the process environment and does not copy
`.env`, commit code, or publish anything. The temporary clone is ignored by Git.

The official starter repository is
[nguyenngocbaocmt02/FTEC5660-HW2](https://github.com/nguyenngocbaocmt02/FTEC5660-HW2).
The LangChain integration follows the
[ChatDeepSeek documentation](https://docs.langchain.com/oss/python/integrations/chat/deepseek).
Documentation was used for implementation only; candidate evidence came solely
from the SocialGraph MCP server.
