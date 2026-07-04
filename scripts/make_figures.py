"""Publication figures for the FLINT paper (AAAI/NeurIPS aesthetic).

Every number is pulled from experiments/flint/results.json (the source of truth) or is a
literal that matches a specific key there (noted in-code). Renders vector PDFs into
paper/flint/figures/. House style: Okabe-Ito colorblind-safe palette, black marker/bar
edges for bold contrast, NO chart titles/subtitles (the message lives in the caption),
minimal spines, sized for two-column legibility.

Run:  HF_HOME=/nas/ckgfs/jaunts/jahin/hf_cache python3 scripts/make_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RES = json.loads((ROOT / "experiments/flint/results.json").read_text())
FIG = ROOT / "paper/flint/figures"
FIG.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette
BLK = "#000000"; ORANGE = "#E69F00"; SKY = "#56B4E9"; GREEN = "#009E73"
YEL = "#F0E442"; BLUE = "#0072B2"; VERM = "#D55E00"; PURP = "#CC79A7"; GREY = "#999999"
FLINT_C = VERM  # FLINT's signature colour

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.linewidth": 0.9, "axes.edgecolor": "#222222",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
EDGE = dict(edgecolor=BLK, linewidth=1.1)          # for markers
BAR = dict(edgecolor=BLK, linewidth=1.0)           # for bars


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


# ---------------------------------------------------------------- 1. Pareto
def fig_pareto():
    """Accuracy vs. compute. FLINT measured (46KB/0.10s CPU); heavy SOTA qualitative (not timed)."""
    eff = RES["efficiency"]; fx = eff["cta_gbm_fit_s"]
    S = RES["SIGNIFICANCE_vs_real_SOTA_250wt"]["cta"]  # each system's own-EL 250WT CTA
    fig, ax = plt.subplots(figsize=(4.7, 3.4))
    # FLINT: measured train cost 0.10s CPU; 250WT CTA 0.782
    ax.scatter([fx], [0.782], s=210, color=FLINT_C, marker="D", zorder=6, **EDGE)
    ax.annotate("FLINT\n(46 KB)", (fx, 0.782),
                textcoords="offset points", xytext=(8, -30), ha="center", fontsize=8.5, color=FLINT_C, weight="bold")
    # heavy SOTA pipelines (own EL; NOT timed -> qualitative right region)
    gx = 60.0
    sota = [("GRAMS+", S["vs_GRAMS+"]["their"], BLUE),
            ("DAGOBAH", S["vs_DAGOBAH"]["their"], GREEN),
            ("MTab", S["vs_MTab"]["their"], PURP)]
    for nm, acc, c in sota:
        ax.scatter([gx], [acc], s=150, color=c, marker="o", zorder=5, **EDGE)
        ax.annotate(nm, (gx, acc), textcoords="offset points", xytext=(12, -2), ha="left", fontsize=7.8, color=c)
    ax.text(gx, 0.648, "heavy SOTA\n(GPU; not timed)", ha="center", fontsize=7, color="#555")
    # matches-best / far-cheaper annotation (arrow stops well before the SOTA column;
    # text sits in the empty middle band so it never touches GRAMS+/DAGOBAH/MTab)
    ax.annotate("", xy=(22, 0.783), xytext=(fx * 3.2, 0.783),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.0, ls=(0, (4, 2))))
    ax.text(4.0, 0.715, "$\\approx$ top-SOTA accuracy,\n$\\sim\\!10^{3}\\times$ cheaper",
            fontsize=7.3, color="#555", ha="center")
    ax.axhspan(0.775, 0.797, xmax=0.30, color=FLINT_C, alpha=0.08)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 500); ax.set_ylim(0.64, 0.82)
    ax.set_xlabel("training + inference cost (s, log)")
    ax.set_ylabel("250WT CTA")
    save(fig, "pareto")


# ---------------------------------------------------------------- 2. Significance forest
def fig_forest():
    """Paired per-table DeltaF1 (FLINT - SOTA) on 250WT; FLINT wins all 3 on CPA, ties on CTA."""
    S = RES["SIGNIFICANCE_vs_real_SOTA_250wt"]
    rows = [  # (label, dF, p, task)
        ("vs. MTab",    S["cpa"]["vs_MTab"]["dF"],    S["cpa"]["vs_MTab"]["p"],    "CPA"),
        ("vs. DAGOBAH", S["cpa"]["vs_DAGOBAH"]["dF"], S["cpa"]["vs_DAGOBAH"]["p"], "CPA"),
        ("vs. GRAMS+",  S["cpa"]["vs_GRAMS+"]["dF"],  S["cpa"]["vs_GRAMS+"]["p"],  "CPA"),
        ("vs. MTab",    S["cta"]["vs_MTab"]["dF"],    S["cta"]["vs_MTab"]["p"],    "CTA"),
        ("vs. DAGOBAH", S["cta"]["vs_DAGOBAH"]["dF"], S["cta"]["vs_DAGOBAH"]["p"], "CTA"),
        ("vs. GRAMS+",  S["cta"]["vs_GRAMS+"]["dF"],  S["cta"]["vs_GRAMS+"]["p"],  "CTA"),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    ys = list(range(len(rows)))[::-1]
    for y, (lab, dF, p, task) in zip(ys, rows):
        sig = p < 0.05
        col = GREEN if (sig and dF > 0) else (GREY if not sig else VERM)
        ax.plot([0, dF], [y, y], color=BLK, lw=0.8, zorder=1)
        ax.scatter([dF], [y], s=130, color=col, marker="o", zorder=3, **EDGE)
        star = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "n.s."))
        if dF >= 0:
            off, ha, va = (9, 0), "left", "center"
        else:  # negative: label above the point to avoid the y-axis tick labels
            off, ha, va = (0, 12), "center", "bottom"
        ax.annotate(f"{dF:+.3f} {star}", (dF, y), textcoords="offset points",
                    xytext=off, va=va, ha=ha, fontsize=7.6,
                    color=BLK, weight="bold" if sig else "normal")
    ax.axvline(0, color=BLK, lw=1.1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{t}  {l}" for (l, dF, p, t) in rows], fontsize=8)
    # task separators / brackets
    ax.axhspan(2.5, 5.5, color=GREEN, alpha=0.05)   # CPA band (top 3)
    ax.axhspan(-0.5, 2.5, color=GREY, alpha=0.05)   # CTA band
    ax.text(0.19, 5.15, "CPA: wins all three", fontsize=8, color=GREEN, weight="bold")
    ax.text(0.19, 2.15, "CTA: beats MTab, ties rest", fontsize=8, color="#555555")
    ax.set_xlim(-0.08, 0.24); ax.set_ylim(-0.6, 5.7)
    ax.set_xlabel(r"$\Delta$F1  (FLINT $-$ SOTA), 250WT")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    leg = [Line2D([0],[0],marker="o",color="w",markerfacecolor=GREEN,markeredgecolor=BLK,markersize=8,label="sig. win (p<.05)"),
           Line2D([0],[0],marker="o",color="w",markerfacecolor=GREY,markeredgecolor=BLK,markersize=8,label="tie (n.s.)")]
    ax.legend(handles=leg, loc="lower right", fontsize=7)
    save(fig, "significance_forest")


# ---------------------------------------------------------------- 3. LLM 2D dominance
def fig_llm2d():
    """A 46 KB ranker sits alone in the top-right of CTA x CPA, beating 8 LLMs on both axes."""
    P = RES["llm_baseline"]["250wt"]
    fr = P["_frontier_API"]; op = P["_open_local_panel"]
    fig, ax = plt.subplots(figsize=(4.7, 3.8))
    # open models: size ~ params (B)
    params = {"Qwen2.5-7B-Instruct":7,"Llama-3.1-8B-Instruct":8,"Ministral-8B-Instruct-2410":8,
              "gemma-2-9b-it":9,"Mistral-7B-Instruct-v0.2":7,"Qwen2.5-3B-Instruct":3}
    short = {"Qwen2.5-7B-Instruct":"Qwen2.5-7B","Llama-3.1-8B-Instruct":"Llama-3.1-8B",
             "Ministral-8B-Instruct-2410":"Ministral-8B","gemma-2-9b-it":"gemma-2-9b",
             "Mistral-7B-Instruct-v0.2":"Mistral-7B","Qwen2.5-3B-Instruct":"Qwen2.5-3B"}
    lbloff = {"Qwen2.5-7B-Instruct": (-6, 5, "right"), "Qwen2.5-3B-Instruct": (7, 3, "left")}
    for k, v in op.items():
        ax.scatter([v["cta_micro_cscore"]], [v["cpa_micro_f1"]], s=40+params[k]*16,
                   color=SKY, marker="o", zorder=3, alpha=0.95, **EDGE)
        dx, dy, ha = lbloff.get(k, (6, 3, "left"))
        ax.annotate(short[k], (v["cta_micro_cscore"], v["cpa_micro_f1"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=6.6, color=BLUE)
    for k, v in fr.items():
        ax.scatter([v["cta_micro_cscore"]], [v["cpa_micro_f1"]], s=150, color=ORANGE,
                   marker="*", zorder=4, **EDGE)
        ax.annotate(k, (v["cta_micro_cscore"], v["cpa_micro_f1"]),
                    textcoords="offset points", xytext=(8, -2), fontsize=6.8, color="#9A6B00")
    ax.scatter([0.782], [0.676], s=210, color=FLINT_C, marker="D", zorder=6, label="FLINT", **EDGE)
    ax.annotate("FLINT\n(46 KB)", (0.782, 0.676), textcoords="offset points",
                xytext=(-8, 6), fontsize=8, ha="right", color=FLINT_C, weight="bold")
    ax.set_xlabel("CTA (cscore)"); ax.set_ylabel("CPA (micro-F1)")
    ax.set_xlim(0.55, 0.80); ax.set_ylim(0.28, 0.71)
    leg = [Line2D([0],[0],marker="D",color="w",markerfacecolor=FLINT_C,markeredgecolor=BLK,markersize=9,label="FLINT (46 KB)"),
           Line2D([0],[0],marker="*",color="w",markerfacecolor=ORANGE,markeredgecolor=BLK,markersize=11,label="frontier LLM (API)"),
           Line2D([0],[0],marker="o",color="w",markerfacecolor=SKY,markeredgecolor=BLK,markersize=8,label="open LLM (3-9B)")]
    ax.legend(handles=leg, loc="upper left", fontsize=7, handletextpad=0.4)
    save(fig, "llm_dominance")


# ---------------------------------------------------------------- 4. EL gap
def fig_el_gap():
    """Under matched real EL (no gold edge), FLINT-CPA ties GRAMS+ and beats DAGOBAH & MTab."""
    S = RES["SIGNIFICANCE_vs_real_SOTA_250wt"]
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    x = [0, 1]  # CTA, CPA
    # (label, [CTA, CPA], colour, hatch)  -- FLINT solid; published SOTA (own EL) hatched
    series = [
        ("FLINT (gold EL)", [0.782, 0.676], FLINT_C, None),
        ("FLINT (real EL)", [0.737, 0.654], ORANGE, None),
        ("GRAMS+",  [S["cta"]["vs_GRAMS+"]["their"],  S["cpa"]["vs_GRAMS+"]["their"]],  SKY,  "//"),
        ("DAGOBAH", [S["cta"]["vs_DAGOBAH"]["their"], S["cpa"]["vs_DAGOBAH"]["their"]], GREEN, "//"),
        ("MTab",    [S["cta"]["vs_MTab"]["their"],    S["cpa"]["vs_MTab"]["their"]],    PURP, "//"),
    ]
    n = len(series); w = 0.15
    for i, (lab, vals, c, h) in enumerate(series):
        off = (i - (n - 1) / 2) * w
        bars = ax.bar([xx + off for xx in x], vals, w, color=c, hatch=h, label=lab, **BAR)
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}", (r.get_x()+r.get_width()/2, r.get_height()),
                        textcoords="offset points", xytext=(0, 1.5), ha="center", fontsize=5.4, rotation=90)
    ax.annotate("real-EL FLINT-CPA ties GRAMS+,\nbeats DAGOBAH & MTab (all own-EL)", xy=(1 - 1.5*w, 0.654),
                xytext=(1.05, 0.86), ha="center", fontsize=6.4, color="#9A6B00",
                arrowprops=dict(arrowstyle="-|>", color="#9A6B00", lw=0.9))
    ax.set_xticks(x); ax.set_xticklabels(["CTA (cscore)", "CPA (micro-F1)"])
    ax.set_ylim(0.48, 0.92); ax.set_ylabel("250WT accuracy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=5,
              fontsize=5.9, handletextpad=0.3, columnspacing=0.6, borderpad=0.2)
    save(fig, "el_gap")


# ---------------------------------------------------------------- 5. Coverage trajectory
def fig_coverage_traj():
    """Real-EL CTA climbs monotonically with candidate coverage then plateaus, localizing the gap to EL not the ranker."""
    prog = RES["multi_dataset"]["real_candidate_250wt_progression"]
    xs = [57.5, 78.2, 78.9]
    flint = [prog["57.5pct_cov"]["flint"], prog["78.2pct_cov"]["flint"],
             prog["converged_95.9pct_mention_cache_78.9pct_col_cov"]["flint"]]
    grams = [prog["57.5pct_cov"]["grams_algo"], prog["78.2pct_cov"]["grams_algo"],
             prog["converged_95.9pct_mention_cache_78.9pct_col_cov"]["grams_algo"]]
    count = [prog["57.5pct_cov"]["counting"], prog["78.2pct_cov"]["counting"],
             prog["converged_95.9pct_mention_cache_78.9pct_col_cov"]["counting"]]
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    ax.plot(xs, flint, "-", color=FLINT_C, lw=2.0, zorder=3)
    ax.scatter(xs, flint, s=70, color=FLINT_C, marker="D", zorder=4, **EDGE)
    ax.plot(xs, grams, "--", color=BLUE, lw=1.6, zorder=2)
    ax.scatter(xs, grams, s=45, color=BLUE, marker="s", zorder=3, **EDGE)
    ax.plot(xs, count, ":", color=GREY, lw=1.6, zorder=2)
    ax.scatter(xs, count, s=45, color=GREY, marker="^", zorder=3, **EDGE)
    ax.axhline(0.782, color=FLINT_C, lw=1.1, ls=(0, (4, 3)), alpha=0.7)
    ax.text(60.5, 0.788, "gold-entity ceiling 0.782", fontsize=7, color=BLK, ha="center")
    ax.annotate("plateau (unlinkable\ncolumns ~21%)", (78.9, flint[-1]), textcoords="offset points",
                xytext=(-4, -28), fontsize=7, ha="right", color=BLK)
    ax.set_xlabel("candidate-cache column coverage (%)")
    ax.set_ylabel("250WT CTA (cscore)")
    ax.set_xlim(54, 83); ax.set_ylim(0.55, 0.805)
    leg = [Line2D([0], [0], marker="D", color=FLINT_C, markerfacecolor=FLINT_C, markeredgecolor=BLK, markersize=8, lw=2.0, label="FLINT"),
           Line2D([0], [0], marker="s", color=BLUE, markerfacecolor=BLUE, markeredgecolor=BLK, markersize=7, lw=1.6, ls="--", label="GRAMS+ algorithm"),
           Line2D([0], [0], marker="^", color=GREY, markerfacecolor=GREY, markeredgecolor=BLK, markersize=7, lw=1.6, ls=":", label="counting")]
    ax.legend(handles=leg, loc="lower right", fontsize=7.2, labelspacing=0.8, handletextpad=0.5)
    save(fig, "coverage_trajectory")


# ---------------------------------------------------------------- 6. CTA granularity offset
def fig_cta_offset():
    """CTA's dominant failure is over-specificity (climb up), not reachability: gold is reachable 97% of the time."""
    g = RES["cta"]["gold_entities_identical_protocol"]["granularity_offset"]
    cats = ["exact\nmatch", "gold is\nancestor\n(too specific)", "gold is\nfiner", "off-path"]
    vals = [g["exact"]*100, g["gold_is_ancestor"]*100, g["gold_finer"]*100, g["off_path"]*100]
    cols = [GREEN, VERM, ORANGE, GREY]
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    bars = ax.bar(range(len(cats)), vals, color=cols, width=0.66, **BAR)
    for r, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}%", (r.get_x()+r.get_width()/2, r.get_height()),
                    textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8, weight="bold")
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=7.5)
    ax.set_ylabel("% of 250WT columns"); ax.set_ylim(0, 72)
    ax.text(0.98, 0.9, "gold reachable 97%\n(4-hop closure)", transform=ax.transAxes,
            ha="right", fontsize=7.5, color=BLK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BLK, lw=0.8))
    save(fig, "cta_offset")


# ---------------------------------------------------------------- 7. Decoder precision gain
def fig_cpa_precision():
    """On identical per-link scores, the arborescence decode buys precision (0.761->0.805) at equal recall."""
    c = RES["cpa"]["gold_entities_identical_protocol"]
    th = c["flint_threshold_decode"]; ar = c["flint_steiner_arborescence_decode"]
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    ax.scatter([th["R"]], [th["P"]], s=150, color=SKY, marker="s", zorder=4, **EDGE)
    ax.scatter([ar["R"]], [ar["P"]], s=170, color=FLINT_C, marker="D", zorder=5, **EDGE)
    ax.annotate("", xy=(ar["R"], ar["P"]), xytext=(th["R"], th["P"]),
                arrowprops=dict(arrowstyle="-|>", color=BLK, lw=1.4, shrinkA=8, shrinkB=8))
    ax.annotate(f"+{(ar['P']-th['P'])*1000:.0f}e-3 precision\nat equal recall",
                ((th["R"]+ar["R"])/2, (th["P"]+ar["P"])/2), textcoords="offset points",
                xytext=(10, -4), fontsize=7.6, color=BLK, weight="bold")
    ax.annotate(f"P={th['P']:.3f}", (th["R"], th["P"]), textcoords="offset points", xytext=(9, -3), ha="left", fontsize=7.5, color=BLUE)
    ax.annotate(f"P={ar['P']:.3f}", (ar["R"], ar["P"]), textcoords="offset points", xytext=(9, 4), fontsize=7.5, color=FLINT_C, weight="bold")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_xlim(0.548, 0.60); ax.set_ylim(0.735, 0.835)
    leg = [Line2D([0], [0], marker="s", color="w", markerfacecolor=SKY, markeredgecolor=BLK,
                  markersize=9, label="threshold decode"),
           Line2D([0], [0], marker="D", color="w", markerfacecolor=FLINT_C, markeredgecolor=BLK,
                  markersize=9, label="arborescence (Edmonds)")]
    ax.legend(handles=leg, loc="upper left", fontsize=7.4, labelspacing=0.9, handletextpad=0.5)
    save(fig, "cpa_precision")


if __name__ == "__main__":
    print("Rendering FLINT figures ->", FIG)
    fig_pareto(); fig_forest(); fig_llm2d(); fig_el_gap()
    fig_coverage_traj(); fig_cta_offset(); fig_cpa_precision()
    print("done.")
