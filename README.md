# Causal Agent Bench

Anonymous code and outcome release for **Intervention Success Is Not Mechanism Identification in Tool-Using Agents**.

CAB evaluates whether tool-using agents can recover answers from repository evidence, and tests how interventions on sources, prompts, and interfaces change success. The release includes offline analyses, paper-facing result tables, a portable reference runner, and a CPU-only smoke test.

## Setup

Python 3.11 or later is required.

```bash
python3 -m pip install -r requirements.txt
```

For Hugging Face inference, install `torch` and `transformers` for your platform. The offline analyses and smoke test do not need a model or GPU.

## Repository structure

- `cab/` — task loader, prompt builder, SERIAL/PARALLEL contracts, parser, scorer, runner, and model adapters.
- `configs/` — paper experiment settings, prompt coverage, and a local OpenAI-compatible example.
- `data/` — prompt manifest, released outcomes, expected paper values, analysis plan, and repository metadata.
- `results/` — regenerated tables, release verification checks, and figures.
- `scripts/` — analysis, figure generation, experiment runner, and smoke test.
- `tests/` — runtime and offline-analysis checks.

## Reproduce paper results

From this directory, run:

```bash
python3 scripts/analyze.py
python3 scripts/make_figures.py
```

The analysis reads `data/results.csv` and writes arm counts, paired contrasts, repository-cluster intervals, matched-prompt summaries, and the 180-repository results under `results/`. Release verification checks target the submitted values at their reported precision. The E05 submitted Monte Carlo value is retained alongside a separate seeded recomputation.

## Run CAB

The portable task format is JSON or JSON Lines. A self-authored example exercises the runtime without a model server:

```bash
python3 scripts/smoke_test.py
```

It checks a SERIAL one-source read, a PARALLEL multi-source read, a tool failure and recovery, answer parsing, alias handling, scoring, and result serialization. It tests runtime plumbing, not model accuracy. `data/smoke_task.json` is a schema fixture, not a paper evaluation task.

For a supplied task panel and local OpenAI-compatible server, set `CAB_ENDPOINT` and run:

```bash
python3 scripts/run_experiment.py \
  --tasks path/to/tasks.json \
  --config configs/local_openai.json \
  --output results/run.jsonl
```

The adapter reads model ID, endpoint, temperature, top-p, seed, dtype, and device settings from the configuration. Set `CAB_API_KEY` only if the local endpoint requires authentication. Task conditions can provide CLEAN, PRESERVE, and DISRUPT source environments; prompt arms include BASE, CONTROL, NEUTRAL, and PROCEDURAL. SERIAL permits one source per action; PARALLEL permits multiple declared sources in the requested order.

Verified exact paper prompt variants are provided in `data/prompt_manifest.json`; the built-in templates are portable fallbacks for new user-supplied tasks. Paper-run settings are summarized in `configs/paper_experiments.json`. The prompt manifest can also be named by the config's `prompt_manifest` field. Supplied prompts are used verbatim.

Exact prompt manifests are released for the matched-prompt evaluations (E01, E04, and E05), where wording is the experimental variable. E21's hash-verified BASE/PLACEBO prompts are included, but that comparison is invalid for PARALLEL because of a validator issue; older source/evidence interventions are represented by frozen outcomes and the portable CAB interface, and unverified historical rendered prompts are not presented as exact artifacts.

The portable runner implements the released CAB interfaces and scoring behavior; exact paper-table reproduction uses the included frozen outcomes.

## Main evaluations

- Authentic repository evaluation: CLEAN, PRESERVE, and DISRUPT.
- Source-information intervention: correct source, wrong source, localization control, and replay.
- Model-interface evaluations: SERIAL and PARALLEL across model families.
- Required-source intervention: natural, unrelated-source, and required-source conditions.
- Matched-prompt evaluations: same-run prompts and six matched prompt pairs.
- 180-repository evaluation: 30,240 runs across 360 tasks and 180 repository clusters.
- Additional checks: raw JSON, file-format tasks, no-tools, and repeated-serving checks; incomplete-study counts and statuses are in `data/additional_checks.csv`.

## Representative results

- Authentic evaluation: 27/32 CLEAN, 26/32 PRESERVE, and 14/32 DISRUPT.
- Model-interface comparison: Gemma changes by −2.1 pp with SERIAL and −33.3 pp with PARALLEL in the third repository set.
- Six-pair prompt evaluation: Qwen3/SERIAL/DISRUPT procedural minus neutral is −11.458 pp.
- 180-repository evaluation: 10 of 12 primary comparisons survive Holm correction. Many large differences are associated with action-format failures and do not establish a loss of task reasoning.
- Required-source intervention: Gemma/PARALLEL is 13/24 versus 4/24 when the agent must find the source, 17/24 versus 5/24 with an unrelated source, and 24/24 versus 24/24 when the required source is supplied.

## Data

`data/results.csv` contains 44,832 compact outcome rows, task and repository identifiers, and scoring/runtime metadata. The prompt manifest contains model-visible prompt text and task questions but excludes evidence contents and gold answers. The task panels, model answers, raw trajectories, and upstream source text are not included. `data/repository_sources.csv` gives public URLs and frozen commit IDs for the 180-repository evaluation; other repository identifiers remain cluster labels where a commit could not be verified from the released artifacts.

The paper task panels and repository source text are not included. The runner accepts task panels supplied by the user.

## Results

`results/arm_counts.csv`, `results/contrasts.csv`, `results/matched_prompt_results.csv`, and `results/large_repository_results.csv` contain the regenerated summaries. `results/checks.json` records release verification status. `results/figures/` contains SVG figures created by `scripts/make_figures.py`.

## Notes

The seeded E05 Monte Carlo recomputation is reported separately from the submitted value saved in the analysis record. The release does not claim that reconstructed portable code is byte-identical to every historical wrapper.

The MIT license covers code in this repository. It does not license repository-derived data. No upstream source text is redistributed.

## License

See `LICENSE` for the code license.
