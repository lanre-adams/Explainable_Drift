"""PDF report for a pilot run (matplotlib figures + reportlab layout)."""
from __future__ import annotations

import io
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .engine import MONITOR_LABELS, MONITORS

NAVY = "#1F3864"
COLORS = {"INPUT": "#7F7F7F", "EXPL": "#C55A11", "EXPLc": "#1F4E79", "CONF": "#548235", "CONFORM": "#7030A0"}
MARK = {"INPUT": "^", "EXPL": "s", "EXPLc": "o", "CONF": "D", "CONFORM": "v"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False})


def _png(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=200, bbox_inches="tight"); plt.close(fig)
    buf.seek(0); return buf


def fig_harmful(res):
    d = pd.DataFrame(res["harm_curve"])
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    for t, c, mk in zip(d.test_set.unique(), ["#1F4E79", "#C55A11"], ["o", "s"]):
        g = d[d.test_set == t]
        ax.plot(g.share * 100, g.error * 100, color=c, marker=mk, ms=3, lw=1.6, label=f"Error, {t}")
        ax.plot(g.share * 100, g.fpr * 100, color=c, marker=mk, ms=3, lw=1.1, ls="--", label=f"False-positive rate, {t}")
    ax.axhline(res["config"]["fail_threshold"] * 100, color="grey", lw=0.8, ls=":")
    ax.text(1, res["config"]["fail_threshold"] * 100 + 0.6, "failure threshold", color="grey", fontsize=7)
    ax.set_xlabel("Share of window from shifted test data (%)"); ax.set_ylabel("%")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    return _png(fig)


def fig_benign(res):
    d = pd.DataFrame(res["benign_curve"])
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    for m in MONITORS:
        ax.plot(d.share * 100, d["est_" + m] * 100, color=COLORS[m], marker=MARK[m], ms=3, lw=1.4, label=MONITOR_LABELS[m])
    ax.plot(d.share * 100, d.error * 100, color="black", ls=":", lw=1.2, label="True error")
    ax.axhline(res["config"]["fail_threshold"] * 100, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Attack share of window (%)"); ax.set_ylabel("Estimated error (%)")
    ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=2)
    return _png(fig)


def fig_tradeoff(res):
    ms = res["monitors"]
    labels = [m["label"].replace("Explanation drift", "Expl. drift") for m in ms]
    cols = [COLORS[m["key"]] for m in ms]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
    y = range(len(ms))
    panels = [([(m["auroc"] or 0) for m in ms], "AUROC, failing vs healthy\n(higher is better)", (0.4, 1.0), "{:.3f}"),
              ([(m["false_alarm"] or 0) * 100 for m in ms], "False alarms, harmless (%)\n(lower is better)", (0, 105), "{:.1f}"),
              ([max(m["ms_per_window"], 0.01) for m in ms], "Cost per window (ms, log)\n(lower is better)", None, "ms")]
    for ax, (vals, title, lim, fmt) in zip(axs, panels):
        ax.barh(list(y), vals, color=cols, height=0.62)
        ax.set_title(title, fontsize=8); ax.invert_yaxis()
        if lim: ax.set_xlim(*lim)
        else: ax.set_xscale("log"); ax.set_xlim(0.005, max(vals) * 8)
        for i, v in enumerate(vals):
            txt = (f"{v:.0f} ms" if v >= 10 else f"{v:.2f} ms") if fmt == "ms" else fmt.format(v)
            ax.text(v, i, " " + txt, va="center", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
    axs[0].set_yticks(list(y)); axs[0].set_yticklabels(labels, fontsize=7.5)
    fig.tight_layout()
    return _png(fig)


def _fmt(v, pct=False, d=3):
    if v is None:
        return "–"
    return f"{v * 100:.1f}%" if pct else f"{v:.{d}f}"


def build_pdf(res: dict, run_id: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Explanation-Drift Pilot Report", author="Olanrewaju (Lanre) Adams Agunloye")
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=17, textColor=colors.HexColor(NAVY), alignment=0, spaceAfter=4)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, textColor=colors.HexColor(NAVY), spaceBefore=8, spaceAfter=4)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9, leading=12.5, alignment=TA_JUSTIFY)
    S = ParagraphStyle("S", parent=B, fontSize=7.5, leading=10, textColor=colors.HexColor("#555555"))
    cfg, det = res["config"], res["detector"]
    mons = {m["key"]: m for m in res["monitors"]}
    best_auc = max(res["monitors"], key=lambda m: (m["auroc"] or 0) - (m["false_alarm"] or 0))
    stamp = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")

    el = [Paragraph("Explanation-Drift Pilot Report", H1),
          Paragraph(f"Can label-free monitors tell when an intrusion detector is failing? · Run {run_id} · {stamp}", S),
          Spacer(1, 6)]

    el += [Paragraph("Summary", H2), Paragraph(
        f"An XGBoost intrusion detector was trained on NSL-KDD and monitored over {res['n_windows']['harmful'] + res['n_windows']['harmless']} "
        f"windows of {cfg['window_size']:,} records across {cfg['seeds']} seed(s). On in-distribution validation data its error was "
        f"{det['val_error'] * 100:.2f}%. As shifted test data were mixed in, error rose to "
        + ", ".join(f"{w['error'] * 100:.1f}% ({w['test_set']})" for w in det["worst_harmful"])
        + f", with no labels available to reveal it. On harmless windows, where only the attack share changed and true error never exceeded "
        f"{det['benign_true_error_max'] * 100:.1f}%, naive explanation drift raised false alarms on {_fmt(mons['EXPL']['false_alarm'], True)} "
        f"of windows and input drift on {_fmt(mons['INPUT']['false_alarm'], True)}. Class-conditional explanation drift reduced this to "
        f"{_fmt(mons['EXPLc']['false_alarm'], True)} (AUROC {_fmt(mons['EXPLc']['auroc'])}), compared with {_fmt(mons['CONF']['false_alarm'], True)} "
        f"for confidence (AUROC {_fmt(mons['CONF']['auroc'])}) and {_fmt(mons['CONFORM']['false_alarm'], True)} for conformal credibility "
        f"(AUROC {_fmt(mons['CONFORM']['auroc'])}). Explanation-based monitors cost about {mons['EXPLc']['ms_per_window']:.0f} ms per window, "
        f"against {mons['CONF']['ms_per_window']:.2f} ms for confidence.", B)]

    el += [Paragraph("Monitor comparison", H2)]
    HS = ParagraphStyle("HS", parent=B, fontSize=7.4, leading=9, textColor=colors.white, fontName="Helvetica-Bold", alignment=1)
    head = ["Monitor", "Rank corr. with error (harmful)", "AUROC failing vs healthy (95% CI)", "False alarms, harmless windows",
            "Mean est. error, harmless", "Cost per window"]
    rows = [[Paragraph(h, HS) for h in head]]
    for m in res["monitors"]:
        ci = f"{_fmt(m['auroc'])}" + (f" ({_fmt(m['auroc_lo'])}–{_fmt(m['auroc_hi'])})" if cfg["seeds"] > 1 else "")
        rows.append([m["label"], _fmt(m["rho"]), ci, _fmt(m["false_alarm"], True), _fmt(m["benign_est_error"], True),
                     f"{m['ms_per_window']:.1f} ms" if m["ms_per_window"] >= 1 else f"{m['ms_per_window']:.2f} ms"])
    t = Table(rows, colWidths=[52 * mm, 25 * mm, 32 * mm, 23 * mm, 23 * mm, 19 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 7.6), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                           ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFBFBF")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")])]))
    el += [t, Paragraph(f"“Failing” = window error above {cfg['fail_threshold'] * 100:.1f}%. Error estimates use an isotonic map fitted on each "
                        "seed’s harmful-drift windows. Cost is mean wall-clock time per window; explanation monitors include TreeSHAP computation.", S)]

    el += [Paragraph("Figure 1. Harmful drift: the detector fails silently", H2), Image(fig_harmful(res), width=170 * mm, height=72 * mm),
           Paragraph("Error and false-positive rate as the share of shifted test records in each window grows (mean across seeds and repetitions).", S),
           PageBreak(),
           Paragraph("Figure 2. Harmless drift: which monitors raise false alarms?", H2), Image(fig_benign(res), width=170 * mm, height=72 * mm),
           Paragraph("Windows drawn only from in-distribution validation data with the attack share changed. True error stays near zero, "
                     "so any estimate above the dashed failure threshold is a false alarm.", S),
           Paragraph("Figure 3. Detection, false alarms and cost", H2), Image(fig_tradeoff(res), width=174 * mm, height=63 * mm),
           Paragraph("A useful monitor detects failing windows (high AUROC) without alarming on harmless ones, at a cost small organisations can afford.", S)]

    el += [KeepTogether([Paragraph("Interpretation", H2), Paragraph(
        f"On this configuration, the best balance of detection and false alarms comes from {best_auc['label'].lower()}. "
        "The key comparison for the proposed PhD is between class-conditional explanation drift and the cheaper confidence and conformal "
        "monitors. Where they perform similarly, explanations have not yet shown added value; the research question is whether they detect "
        "failures that confidence-based monitors miss, notably under evasion attacks designed to produce confident mistakes, which this "
        "pilot does not include.", B)])]

    el += [KeepTogether([Paragraph("Method", H2), Paragraph(
        f"Detector: XGBoost ({cfg['n_estimators']} trees, depth {cfg['max_depth']}) trained on 80% of KDDTrain+; reference set "
        f"{cfg['reference_size']:,} held-out validation records. Harmful drift: windows mixing validation records with 0–100% records from "
        f"{', '.join(cfg['test_sets'])} in {cfg['mix_steps']} steps, {cfg['reps']} repetitions each. Harmless drift: validation records only, "
        f"attack share set to {', '.join(f'{s:.0%}' for s in cfg['benign_shares'])}. Monitors: per-feature KS input drift; TreeSHAP "
        "explanation drift (mean per-feature Wasserstein distance scaled by reference spread), naive and within predicted class; "
        "average thresholded confidence (Garg et al., 2022); and inductive conformal credibility with class-conditional calibration "
        f"(share of predictions with credibility below {cfg['conformal_alpha']}). Monitors never see labels.", B)])]

    el += [KeepTogether([Paragraph("Limitations", H2), Paragraph(
        "NSL-KDD has no timestamps, so mixture shift stands in for temporal drift; it is also an ageing dataset that pre-dates widespread "
        "encryption. Harmless drift here is constructed (attack-share change), which the full study replaces with naturally occurring shifts "
        "defined by outcome. One model family is used, error maps are fitted on the same seed’s harmful windows, and no adversarial "
        "evasion is included. Results should be read as a feasibility check, not as evidence about deployed systems.", B)])]

    el += [KeepTogether([Paragraph("Reproducibility", H2), Paragraph(
        f"Run time {res['elapsed_s']:.0f} s. Configuration: " + ", ".join(f"{k}={v}" for k, v in cfg.items())
        + ". Per-window results are available as CSV from the application. Code: explanation-drift pilot, packaged with Docker.", S)])]

    def footer(canvas, d):
        canvas.saveState(); canvas.setFont("Helvetica", 7); canvas.setFillColor(colors.HexColor("#808080"))
        canvas.drawString(18 * mm, 9 * mm, "O.A. Agunloye · Explanation-drift pilot · NSL-KDD · all figures measured")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {d.page}"); canvas.restoreState()

    doc.build(el, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
