"""Offline analysis for compact outcomes released with the manuscript candidate."""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def load_rows() -> list[dict[str, str]]:
    with (DATA / "results.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def exact_p(a_only: int, b_only: int) -> float:
    """Two-sided exact binomial test on discordant paired binary outcomes."""
    n = a_only + b_only
    if not n:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = previous
    return adjusted


def grouped(rows: list[dict[str, str]], keys: tuple[str, ...]):
    output: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        output[tuple(row[k] for k in keys)].append(row)
    return output


def pair_contrast(rows: list[dict[str, str]], a_arm: str, b_arm: str,
                  extra: tuple[str, ...] = ()) -> dict[str, Any] | None:
    """Return A minus B percentage-point effect for paired task rows."""
    by_task: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_task[row["task_id"]][row["arm"]] = row
    pairs = []
    for task, arms in by_task.items():
        if a_arm in arms and b_arm in arms:
            a, b = as_bool(arms[a_arm]["correct"]), as_bool(arms[b_arm]["correct"])
            repo = arms[a_arm]["repository_id"] or arms[b_arm]["repository_id"] or task
            pairs.append((task, repo, a, b))
    if not pairs:
        return None
    a_only = sum(a and not b for _, _, a, b in pairs)
    b_only = sum(b and not a for _, _, a, b in pairs)
    effect = 100 * sum(int(a) - int(b) for _, _, a, b in pairs) / len(pairs)
    clusters: dict[str, list[int]] = defaultdict(list)
    for _, repo, a, b in pairs:
        clusters[repo].append(int(a) - int(b))
    return {"n": len(pairs), "a_successes": sum(a for _, _, a, _ in pairs),
            "b_successes": sum(b for _, _, _, b in pairs),
            "a_only": a_only, "b_only": b_only, "effect_pp": effect,
            "exact_p": exact_p(a_only, b_only), "clusters": clusters,
            "task_differences": pairs}


def percentile_sorted(values: list[float], p: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * p
    low, high = math.floor(pos), math.ceil(pos)
    if low == high:
        return values[low]
    return values[low] * (high - pos) + values[high] * (pos - low)


def cluster_bootstrap(clusters: dict[str, list[int | float]], seed: int,
                      draws: int = 50_000) -> list[float]:
    keys = sorted(clusters)
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        selected = [keys[rng.randrange(len(keys))] for _ in keys]
        sample = [value for key in selected for value in clusters[key]]
        values.append(100 * sum(sample) / len(sample))
    values.sort()
    return [values[int(.025 * len(values))], values[int(.975 * len(values)) - 1]]


def numpy_cluster_bootstrap(cluster_means: list[float], seed: int,
                            draws: int = 50_000) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=float)
    for start in range(0, draws, 2048):
        stop = min(draws, start + 2048)
        indexes = rng.integers(0, len(cluster_means), size=(stop - start, len(cluster_means)))
        values[start:stop] = np.asarray(cluster_means)[indexes].mean(axis=1)
    return [float(np.percentile(values, 2.5) * 100),
            float(np.percentile(values, 97.5) * 100)]


def loo_range(clusters: dict[str, list[int | float]]) -> list[float]:
    keys = sorted(clusters)
    values = []
    for dropped in keys:
        kept = [x for key in keys if key != dropped for x in clusters[key]]
        values.append(100 * sum(kept) / len(kept))
    return [min(values), max(values)]


def arm_counts(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    keys = ("experiment", "panel", "model", "interface", "condition", "prompt_class", "arm")
    output = []
    for key, cell in sorted(grouped(rows, keys).items()):
        correct = sum(as_bool(r["correct"]) for r in cell)
        output.append(dict(zip(keys, key)) | {"n": len(cell), "correct": correct,
                       "accuracy": correct / len(cell),
                       "valid_finals": sum(as_bool(r["valid_final"]) for r in cell if r["valid_final"] != ""),
                       "parser_rejections": sum(int(r["parser_rejections"] or 0) for r in cell),
                       "action_rejections": sum(int(r["action_rejections"] or 0) for r in cell),
                       "tool_calls": sum(int(r["tool_calls"] or 0) for r in cell)})
    return output


def basic_contrasts(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    authentic = [r for r in rows if r["experiment"] == "authentic_repository_evaluation"]
    f = pair_contrast(authentic, "CLEAN", "DISRUPT")
    if f:
        output.append(dict(experiment="authentic_repository_evaluation", contrast="CLEAN_MINUS_DISRUPT",
                           **{k:f[k] for k in ("n","a_successes","b_successes","a_only","b_only","effect_pp","exact_p")},
                           bootstrap_ci_pp=cluster_bootstrap(f["clusters"],987654321),
                           loo_range_pp=loo_range(f["clusters"])))
    # Core BASE/PLACEBO comparisons in repository-set and boundary panels.
    for experiment in ("model_interface_comparisons", "file_format_boundary", "strict_json_boundary"):
        subset = [r for r in rows if r["experiment"] == experiment]
        keys = ("panel", "model", "interface", "condition")
        for key, cell in sorted(grouped(subset, keys).items()):
            result = pair_contrast(cell, "PLACEBO", "BASE")
            if not result:
                continue
            output.append(dict(experiment=experiment, panel=key[0], model=key[1], interface=key[2], condition=key[3],
                               contrast="PLACEBO_MINUS_BASE", **{k: result[k] for k in ("n", "a_successes", "b_successes", "a_only", "b_only", "effect_pp", "exact_p")},
                               bootstrap_ci_pp=cluster_bootstrap(result["clusters"], {'Set 1':20260829,'Set 2':20260830,'Set 3':20260830}.get(key[0],20260830)),
                               loo_range_pp=loo_range(result["clusters"])))
    # Source information intervention: every active arm against BASE.
    subset = [r for r in rows if r["experiment"] == "source_information_intervention"]
    for key, cell in sorted(grouped(subset, ("model", "interface", "condition")).items()):
        arms = sorted({r["arm"] for r in cell} - {"BASE"})
        for arm in arms:
            result = pair_contrast(cell, arm, "BASE")
            if result:
                output.append(dict(experiment="source_information_intervention", model=key[0], interface=key[1], condition=key[2],
                                   contrast=f"{arm}_MINUS_BASE", **{k: result[k] for k in ("n", "a_successes", "b_successes", "a_only", "b_only", "effect_pp", "exact_p")}))
    # Required-source condition: compare arms within each mediator regime.
    subset = [r for r in rows if r["experiment"] == "required_source_intervention"]
    for key, cell in sorted(grouped(subset, ("model", "interface", "prompt_class")).items()):
        result = pair_contrast(cell, "BASE", "PLACEBO")
        if result:
            output.append(dict(experiment="required_source_intervention", model=key[0], interface=key[1], condition=key[2],
                               contrast="BASE_MINUS_CONTROL", **{k: result[k] for k in ("n", "a_successes", "b_successes", "a_only", "b_only", "effect_pp", "exact_p")}))
    # Preserve the paper-defined study families for model/interface and boundary tables.
    families: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(output):
        if row["experiment"] == "model_interface_comparisons":
            panel, model = row["panel"], row["model"]
            if panel == "Set 2":
                role = {"Gemma3":"secondary", "Qwen3":"tertiary", "InternLM2.5":"primary", "Yi-1.5":"primary"}[model]
                family = f"Set 2 / {role}"
            else:
                family = panel
            families[family].append(idx)
        elif row["experiment"] in {"file_format_boundary", "strict_json_boundary"}:
            families[row["experiment"]].append(idx)
    for indices in families.values():
        adjusted=holm([float(output[i]["exact_p"]) for i in indices])
        for i,p in zip(indices,adjusted): output[i]["holm_p"]=p
    return output


def matched_prompt_results(rows: list[dict[str, str]], experiment: str, seed: int) -> list[dict[str, Any]]:
    subset = [r for r in rows if r["experiment"] == experiment]
    output=[]
    cell_groups=grouped(subset, ("model", "interface", "condition"))
    if experiment == "same_run_prompt_comparison":
        cell_order=[(m,i,c) for m in ("Gemma3","Qwen3") for i in ("SERIAL","PARALLEL") for c in ("CLEAN","DISRUPT")]
    else:
        cell_order=[(m,i,c) for m in ("Gemma3","Qwen3") for i in ("SERIAL","PARALLEL") for c in ("CLEAN","DISRUPT")]
    for idx, key in enumerate(cell_order):
        if key not in cell_groups:
            continue
        cell=cell_groups[key]
        comp = pair_contrast(cell, "PROCEDURAL", "NEUTRAL")
        if not comp:
            continue
        cluster_means=[sum(values)/len(values) for _,values in sorted(comp["clusters"].items())]
        ci = numpy_cluster_bootstrap(cluster_means, seed + idx)
        rec = dict(experiment=experiment, model=key[0], interface=key[1], condition=key[2], contrast="PROCEDURAL_MINUS_NEUTRAL",
                   **{k: comp[k] for k in ("n", "a_successes", "b_successes", "a_only", "b_only", "effect_pp", "exact_p")},
                   bootstrap_ci_pp=ci, loo_range_pp=loo_range(comp["clusters"]))
        if experiment == "same_run_prompt_comparison":
            output.append(rec)
        else:
            # The six-pair estimand first averages P-N within task, then clusters by repository.
            by_task: dict[str, dict[str, dict[str, bool]]] = defaultdict(lambda: defaultdict(dict))
            repo_by_task={}
            for r in cell:
                pair=r["prompt_pair_id"]
                if pair:
                    by_task[r["task_id"]][pair][r["arm"]]=as_bool(r["correct"])
                    repo_by_task[r["task_id"]]=r["repository_id"]
            taskdiff={}
            p_successes=n_successes=0
            pair_diffs=defaultdict(list)
            for task, pairs in by_task.items():
                diffs=[int(v["PROCEDURAL"])-int(v["NEUTRAL"]) for v in pairs.values() if set(v)=={"PROCEDURAL","NEUTRAL"}]
                for pair, arms in pairs.items():
                    if set(arms)=={"PROCEDURAL","NEUTRAL"}:
                        p_successes += int(arms["PROCEDURAL"])
                        n_successes += int(arms["NEUTRAL"])
                        pair_diffs[pair].append(int(arms["PROCEDURAL"])-int(arms["NEUTRAL"]))
                if diffs:
                    taskdiff[task]=(repo_by_task[task],sum(diffs)/len(diffs))
            by_repo: dict[str,list[float]]=defaultdict(list)
            for repo,diff in taskdiff.values(): by_repo[repo].append(diff)
            means=[sum(v)/len(v) for _,v in sorted(by_repo.items())]
            ci=numpy_cluster_bootstrap(means,seed+idx)
            rec["n"]=sum(len(v) for v in pair_diffs.values())
            rec["a_successes"]=p_successes
            rec["b_successes"]=n_successes
            rec["a_only"]=sum(d>0 for v in pair_diffs.values() for d in v)
            rec["b_only"]=sum(d<0 for v in pair_diffs.values() for d in v)
            rec["effect_pp"]=100*(p_successes-n_successes)/rec["n"]
            rec["exact_p"]=None
            rec["bootstrap_ci_pp"]=ci
            rec["task_count"]=len(taskdiff)
            rec["repository_count"]=len(by_repo)
            rec["loo_range_pp"]=loo_range(by_repo)
            rec["prompt_pair_effects_pp"] = {pair:100*sum(v)/len(v) for pair,v in sorted(pair_diffs.items())}
            if key==("Qwen3","SERIAL","DISRUPT"):
                rec["reported_mc_p"]=0.00006999930000699993
                rec["reported_mc_exceedances"]=6
            rec["recomputed_cluster_signflip_p"]=cluster_signflip_p(means,2026090907+idx)
            rec["recomputed_cluster_signflip_exceedances"]=round(rec["recomputed_cluster_signflip_p"]*100_001-1)
            output.append(rec)
    if experiment == "same_run_prompt_comparison":
        adjusted=holm([float(r["exact_p"]) for r in output])
    else:
        adjusted=holm([float(r["recomputed_cluster_signflip_p"]) for r in output])
    for rec,p in zip(output,adjusted):
        rec["holm_p"]=p
        if experiment == "six_pair_prompt_library":
            rec["holm_p_basis"]="seeded_recomputation"
    return output


def pair_level_effects(cell: list[dict[str,str]]) -> dict[str,float]:
    grouped_pair=defaultdict(list)
    by=defaultdict(dict)
    for r in cell:
        if r["prompt_pair_id"]:
            by[(r["prompt_pair_id"],r["task_id"])][r["arm"]]=as_bool(r["correct"])
    for (pair,_), arms in by.items():
        if set(arms)=={"PROCEDURAL","NEUTRAL"}:
            grouped_pair[pair].append(int(arms["PROCEDURAL"])-int(arms["NEUTRAL"]))
    return {pair:100*sum(v)/len(v) for pair,v in sorted(grouped_pair.items())}


def cluster_signflip_p(means: list[float], seed: int, draws: int=100_000) -> float:
    point=abs(float(np.mean(means)))
    rng=np.random.default_rng(seed)
    extreme=0
    for start in range(0,draws,2048):
        stop=min(draws,start+2048)
        flips=rng.integers(0,2,size=(stop-start,len(means)))*2-1
        extreme += int(np.count_nonzero(np.abs((flips*np.asarray(means)).mean(axis=1))>=point))
    return (1+extreme)/(draws+1)


def large_repository_results(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    subset=[r for r in rows if r["experiment"]=="large_repository_matched_prompts"]
    by_cell: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in subset:
        by_cell[(row["model"], row["interface"], row["condition"])].append(row)
    output=[]
    for index, (model, interface, condition) in enumerate(sorted(by_cell)):
        cell=by_cell[(model, interface, condition)]
        by_task=defaultdict(dict); repo_by_task={}
        for r in cell:
            by_task[r["task_id"]][r["arm"]]=as_bool(r["correct"])
            repo_by_task[r["task_id"]]=r["repository_id"]
        by_repo=defaultdict(list)
        arm_counts_local=defaultdict(int)
        for task,arms in by_task.items():
            for arm,ok in arms.items(): arm_counts_local[arm]+=int(ok)
            diffs=[int(arms[f"P_{length}"])-int(arms[f"N_{length}"]) for length in ("SHORT","MEDIUM","LONG")]
            by_repo[repo_by_task[task]].extend(diffs)
        repo_means=[sum(v)/len(v) for _,v in sorted(by_repo.items())]
        point=float(np.mean(repo_means))*100
        # The submitted E04 procedure uses one NumPy stream for both stages.
        rng=np.random.default_rng(2026091200+index)
        cluster_array=np.asarray(repo_means,dtype=float)
        bootstrap=np.empty(50_000,dtype=float)
        for start in range(0,50_000,2000):
            stop=min(50_000,start+2000)
            indexes=rng.integers(0,len(cluster_array),size=(stop-start,len(cluster_array)))
            bootstrap[start:stop]=cluster_array[indexes].mean(axis=1)
        signs=np.empty(100_000,dtype=float)
        for start in range(0,100_000,2000):
            stop=min(100_000,start+2000)
            flips=rng.integers(0,2,size=(stop-start,len(cluster_array)))*2-1
            signs[start:stop]=(flips*cluster_array).mean(axis=1)
        p=(1+int(np.count_nonzero(np.abs(signs)>=abs(float(np.mean(cluster_array)))-1e-14)))/100_001
        ci=[float(np.percentile(bootstrap,2.5)*100),float(np.percentile(bootstrap,97.5)*100)]
        loo_values=[100*float(np.mean([v for k,vals in by_repo.items() if k!=drop for v in vals])) for drop in by_repo]
        output.append(dict(experiment="large_repository_matched_prompts",model=model,interface=interface,condition=condition,
                           contrast="PROCEDURAL_MINUS_NEUTRAL",effect_pp=point,bootstrap_ci_pp=ci,
                           raw_signflip_p=p,cluster_count=len(by_repo),task_count=len(by_task),
                           neutral_correct=sum(arm_counts_local[f"N_{x}"] for x in ("SHORT","MEDIUM","LONG")),
                           procedural_correct=sum(arm_counts_local[f"P_{x}"] for x in ("SHORT","MEDIUM","LONG")),
                           base_correct=arm_counts_local["BASE"],loo_range_pp=[min(loo_values),max(loo_values)],holm_p=None))
    adjusted=holm([r["raw_signflip_p"] for r in output])
    for r,p in zip(output,adjusted): r["holm_p"]=p
    return output


def core_checks(rows: list[dict[str, str]], counts: list[dict[str,Any]],
                contrasts: list[dict[str,Any]], matched: list[dict[str,Any]],
                large: list[dict[str,Any]]) -> dict[str,Any]:
    checks=[]
    expected=json.loads((DATA/'paper_expected.json').read_text(encoding='utf-8'))
    def check(name,condition,observed,expected):
        checks.append({"name":name,"status":"PASS" if condition else "FAIL","observed":observed,"expected":expected})
    def cell(exp,**kw):
        return [r for r in counts if r['experiment']==exp and all(r[k]==v for k,v in kw.items())]
    f=cell('authentic_repository_evaluation')
    fc={r['condition']:r['correct'] for r in f}
    check('authentic counts',fc=={'CLEAN':27,'DISRUPT':14,'PRESERVE':26},fc,{'CLEAN':27,'DISRUPT':14,'PRESERVE':26})
    fcontrast=next((r for r in contrasts if r['experiment']=='authentic_repository_evaluation'),None)
    check('authentic paired analysis',bool(fcontrast and fcontrast['n']==32 and fcontrast['a_only']==13 and fcontrast['b_only']==0 and fcontrast['exact_p']==0.000244140625 and fcontrast['effect_pp']==40.625 and fcontrast['bootstrap_ci_pp']==[25.0,56.25]),
          None if not fcontrast else {k:fcontrast.get(k) for k in ('n','a_only','b_only','effect_pp','exact_p','bootstrap_ci_pp','loo_range_pp')},
          {'n':32,'clean_only':13,'disrupt_only':0,'effect_pp':40.625,'exact_p':0.000244140625,'bootstrap_ci_pp':[25.0,56.25]})
    # Source information exact count table.
    o={}
    for r in cell('source_information_intervention'):
        o.setdefault(r['model'],{}).setdefault(r['condition'],{})[r['arm']]=r['correct']
    expected_o={'Qwen3':{'CLEAN':{'BASE':34,'LOCALIZATION_PLACEBO':35,'CORRECT_SOURCE_ID':37,'WRONG_SOURCE_ID':0,'REPLAY':34},'DISRUPT':{'BASE':32,'LOCALIZATION_PLACEBO':33,'CORRECT_SOURCE_ID':37,'WRONG_SOURCE_ID':0,'REPLAY':35}},'Ministral':{'CLEAN':{'BASE':24,'LOCALIZATION_PLACEBO':4,'CORRECT_SOURCE_ID':32,'WRONG_SOURCE_ID':1,'REPLAY':34},'DISRUPT':{'BASE':17,'LOCALIZATION_PLACEBO':4,'CORRECT_SOURCE_ID':31,'WRONG_SOURCE_ID':1,'REPLAY':34}}}
    check('source-information arm counts',o==expected_o,o,expected_o)
    # Q/R/U table and specified raw examples.
    qru=[r for r in counts if r['experiment']=='model_interface_comparisons']
    qru_cells={(r['panel'],r['model'],r['interface'],r['condition'],r['arm']):r['correct'] for r in qru}
    examples={('Set 1','Granite','SERIAL','DISRUPT','PLACEBO'):29,('Set 1','Granite','SERIAL','DISRUPT','BASE'):12,('Set 2','Gemma3','SERIAL','DISRUPT','BASE'):31,('Set 2','Gemma3','SERIAL','DISRUPT','PLACEBO'):11,('Set 3','Gemma3','PARALLEL','DISRUPT','BASE'):45,('Set 3','Gemma3','PARALLEL','DISRUPT','PLACEBO'):29}
    observed={str(k):qru_cells.get(k) for k in examples}
    check('model-interface comparison examples',all(qru_cells.get(k)==v for k,v in examples.items()),observed,{str(k):v for k,v in examples.items()})
    s=cell('required_source_intervention',model='Gemma3',interface='PARALLEL')
    sc={(r['prompt_class'],r['arm']):r['correct'] for r in s}
    sexpect={('NATURAL','BASE'):13,('NATURAL','PLACEBO'):4,('RECOVERY_PRELOAD','BASE'):24,('RECOVERY_PRELOAD','PLACEBO'):24,('SHAM_PRELOAD','BASE'):17,('SHAM_PRELOAD','PLACEBO'):5}
    check('required-source key setting',all(sc.get(k)==v for k,v in sexpect.items()),{str(k):sc.get(k) for k in sexpect},{str(k):v for k,v in sexpect.items()})
    check('matched-prompt row totals',sum(r['experiment']=='same_run_prompt_comparison' for r in rows)==1152 and sum(r['experiment']=='six_pair_prompt_library' for r in rows)==4992,
          {'same_run':sum(r['experiment']=='same_run_prompt_comparison' for r in rows),'six_pair':sum(r['experiment']=='six_pair_prompt_library' for r in rows)},{'same_run':1152,'six_pair':4992})
    e01=sum((r['valid_final']=='1') for r in rows if r['experiment']=='same_run_prompt_comparison')
    e01rej=sum(int(r['parser_rejections'] or 0) for r in rows if r['experiment']=='same_run_prompt_comparison')
    e05=sum((r['valid_final']=='1') for r in rows if r['experiment']=='six_pair_prompt_library')
    e05rej=sum(int(r['parser_rejections'] or 0) for r in rows if r['experiment']=='six_pair_prompt_library')
    check('matched-prompt validity',e01==1152 and e01rej==0 and e05==4992 and e05rej==0,{'E01_valid':e01,'E01_parser_rejections':e01rej,'E05_valid':e05,'E05_parser_rejections':e05rej},{'E01_valid':1152,'E01_parser_rejections':0,'E05_valid':4992,'E05_parser_rejections':0})
    expected_e01=expected['matched_prompt_experiments']['same_run_cells']
    actual_e01={'|'.join((r['model'],r['interface'],r['condition'])):[r['a_successes'],r['b_successes'],r['effect_pp'],r['exact_p'],r['holm_p']] for r in matched if r['experiment']=='same_run_prompt_comparison'}
    e01_match=set(actual_e01)==set(expected_e01) and all(actual_e01[k][:2]==expected_e01[k][:2] and all(abs(float(a)-float(b))<1e-9 for a,b in zip(actual_e01[k][2:],expected_e01[k][2:])) for k in expected_e01)
    check('same-run primary contrasts and Holm family',e01_match,actual_e01,expected_e01)
    e04=[r for r in rows if r['experiment']=='large_repository_matched_prompts']
    check('180-repository design',len(e04)==30240 and len({r['task_id'] for r in e04})==360 and len({r['repository_id'] for r in e04})==180,
          {'rows':len(e04),'tasks':len({r['task_id'] for r in e04}),'repositories':len({r['repository_id'] for r in e04})},{'rows':30240,'tasks':360,'repositories':180})
    failure=Counter(r['error_class'] for r in e04)
    check('180-repository failure classification',failure.get('PARSER_JSON_DECODE',0)==2289 and failure.get('ACTION_INVALID_READ',0)==124,
          {'PARSER_JSON_DECODE':failure.get('PARSER_JSON_DECODE',0),'ACTION_INVALID_READ':failure.get('ACTION_INVALID_READ',0)}, {'PARSER_JSON_DECODE':2289,'ACTION_INVALID_READ':124})
    w=[r for r in rows if r['experiment']=='closed_book_no_tools']
    check('closed-book total',len(w)==192 and sum(as_bool(r['correct']) for r in w)==3 and sum(int(r['tool_calls'] or 0) for r in w)==0,
          {'rows':len(w),'correct':sum(as_bool(r['correct']) for r in w),'tool_calls':sum(int(r['tool_calls'] or 0) for r in w)}, {'rows':192,'correct':3,'tool_calls':0})
    w_mechanisms={}
    for mechanism in sorted({r['mechanism'] for r in w}):
        group=[r for r in w if r['mechanism']==mechanism]
        w_mechanisms[mechanism]=[sum(as_bool(r['correct']) for r in group),len(group)]
    w_expected={k:[expected['closed_book_no_tools'][f'{k}_correct'],expected['closed_book_no_tools'].get(f'{k}_n',96)] for k in ('TOOL_FAILURE','DECOY_INJECTION')}
    check('closed-book task-family outcomes',w_mechanisms==w_expected,w_mechanisms,w_expected)
    v={(r['model'],r['interface'],r['condition'],r['arm']):r['correct'] for r in cell('strict_json_boundary')}
    v_expected={('InternLM2.5','SERIAL','CLEAN','BASE'):45,('InternLM2.5','SERIAL','CLEAN','PLACEBO'):0,('InternLM2.5','SERIAL','DISRUPT','BASE'):38,('InternLM2.5','SERIAL','DISRUPT','PLACEBO'):0}
    check('strict JSON key outcomes',all(v.get(k)==n for k,n in v_expected.items()),{ '|'.join(k):v.get(k) for k in v_expected},{'|'.join(k):n for k,n in v_expected.items()})
    e10=[r for r in rows if r['experiment']=='repeated_server_sessions']
    check('repeated-session total',len(e10)==864, len(e10),864)
    e10_counts={(r['model'],r['interface'],r['condition'],r['arm']):r['correct'] for r in cell('repeated_server_sessions')}
    repeated_expected={('Qwen3','SERIAL','DISRUPT','N_SHORT'):144,('Qwen3','SERIAL','DISRUPT','P_SHORT'):144,('Gemma3','PARALLEL','CLEAN','N_LONG'):0,('Gemma3','PARALLEL','CLEAN','P_LONG'):5,('Qwen3','PARALLEL','CLEAN','N_MEDIUM'):144,('Qwen3','PARALLEL','CLEAN','P_MEDIUM'):144}
    check('repeated-session selected settings',all(e10_counts.get(k)==v for k,v in repeated_expected.items()),{str(k):e10_counts.get(k) for k in repeated_expected},{str(k):v for k,v in repeated_expected.items()})
    e05q=next((r for r in matched if r['experiment']=='six_pair_prompt_library' and r['model']=='Qwen3' and r['interface']=='SERIAL' and r['condition']=='DISRUPT'),None)
    check('six-pair Qwen estimate',bool(e05q and abs(e05q['effect_pp']-(-11.4583333333))<1e-8 and e05q['b_successes']==256 and e05q['a_successes']==223),
          None if not e05q else {k:e05q.get(k) for k in ('n','a_successes','b_successes','effect_pp','bootstrap_ci_pp','recomputed_cluster_signflip_p','reported_mc_p')}, {'n':288,'procedural':223,'neutral':256,'effect_pp':-11.4583333333,'reported_mc_p':0.00006999930000699993})
    e05_expected=expected['matched_prompt_experiments']
    e05_stats_match=bool(e05q and all(abs(a-b)<1e-8 for a,b in zip(e05q['bootstrap_ci_pp'],e05_expected['six_pair_qwen_serial_disrupt_ci_pp'])) and all(abs(a-b)<1e-8 for a,b in zip(e05q['loo_range_pp'],e05_expected['six_pair_qwen_serial_disrupt_loo_pp'])))
    check('six-pair cluster interval and leave-one-repository-out',e05_stats_match,None if not e05q else {'ci':e05q['bootstrap_ci_pp'],'loo':e05q['loo_range_pp']},{'ci':e05_expected['six_pair_qwen_serial_disrupt_ci_pp'],'loo':e05_expected['six_pair_qwen_serial_disrupt_loo_pp']})
    check('180-repository table cells',len(large)==12,len(large),12)
    e04_expected=expected['large_repository_evaluation']['submitted_cells']
    e04_observed={f"{r['model']}|{r['interface']}|{r['condition']}":r for r in large}
    e04_match=set(e04_observed)==set(e04_expected) and all(
        abs(e04_observed[key]['effect_pp']-value['effect_pp'])<1e-9
        and all(abs(a-b)<1e-8 for a,b in zip(e04_observed[key]['bootstrap_ci_pp'],value['bootstrap_ci_pp']))
        and abs(e04_observed[key]['holm_p']-value['holm_p'])<1e-12
        for key,value in e04_expected.items())
    check('180-repository submitted effects, intervals, and Holm values',e04_match,
          {key:{field:e04_observed[key][field] for field in ('effect_pp','bootstrap_ci_pp','holm_p')}
           for key in e04_observed},e04_expected)
    e05q=next(r for r in matched if r['experiment']=='six_pair_prompt_library' and r['model']=='Qwen3' and r['interface']=='SERIAL' and r['condition']=='DISRUPT')
    checks.append({"name":"six-pair submitted value and recomputed Monte Carlo estimate",
                   "status":"PASS" if e05q['reported_mc_exceedances']==6 and e05q['reported_mc_p']==expected['matched_prompt_experiments']['six_pair_reported_mc_p'] and e05q['recomputed_cluster_signflip_p']<0.001 and e05q['holm_p']<0.001 else "FAIL",
                   "submitted_exceedances":e05q['reported_mc_exceedances'],
                   "submitted_p":e05q['reported_mc_p'],
                   "recomputed_exceedances":e05q['recomputed_cluster_signflip_exceedances'],
                   "recomputed_p":e05q['recomputed_cluster_signflip_p'],
                   "recomputed_holm_p":e05q['holm_p'],
                   "note":"The saved submitted value and the seeded recomputation are kept separate; Monte Carlo estimates can differ by a draw."})
    failed=any(x['status']=='FAIL' for x in checks)
    warned=any(x['status']=='WARN' for x in checks)
    return {"status":"FAIL" if failed else ("PASS_WITH_LIMITATIONS" if warned else "PASS"), "checks":checks,
            "reproduction_limitations":["The submitted E05 Monte Carlo value is preserved from the saved analysis artifact; a seeded recomputation is reported separately.",
              "The original required-source 24-task panel preimage is not included; the release retains the paper's reconstructed task representation and released outcomes.",
              "Task panels and upstream repository text are not redistributed. The portable runner accepts user-supplied task JSON; frozen released outcomes reproduce the paper tables."]}


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("\n", encoding="utf-8")
        return
    fields=[]
    for record in records:
        for key in record:
            if key not in fields: fields.append(key)
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        for record in records:
            writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in record.items()})


def analyze(output: Path | None = None) -> dict[str, Any]:
    out=output or RESULTS
    out.mkdir(parents=True,exist_ok=True)
    rows=load_rows()
    counts=arm_counts(rows)
    contrasts=basic_contrasts(rows)
    matched=matched_prompt_results(rows,'same_run_prompt_comparison',2026090902)
    matched+=matched_prompt_results(rows,'six_pair_prompt_library',2026090906)
    large=large_repository_results(rows)
    write_csv(out/'arm_counts.csv',counts)
    write_csv(out/'contrasts.csv',contrasts)
    write_csv(out/'matched_prompt_results.csv',matched)
    write_csv(out/'large_repository_results.csv',large)
    checks=core_checks(rows,counts,contrasts,matched,large)
    (out/'checks.json').write_text(json.dumps(checks,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return checks
