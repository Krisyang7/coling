"""Deterministically render the modular Sequential EMD-OPD method figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "specs" / "emd_opd_pipeline.json"


def rounded(ax, x, y, w, h, fill, stroke, lw=1.5, radius=14, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fill,
        edgecolor=stroke,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=12, color="#10233F", weight="normal", ha="center", va="center", z=5):
    return ax.text(x, y, text, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z)


def arrow(ax, x1, y1, x2, y2, color="#60728C", lw=1.8, style="-", mutation=13, z=3, curve=0.0):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=2,
        shrinkB=2,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def module(ax, x, y, w, h, title, subtitle, color, pale, number):
    rounded(ax, x, y, w, h, "#FFFFFF", color, lw=1.7, radius=18, z=0)
    rounded(ax, x + 1, y + 1, w - 2, 48, pale, pale, lw=0, radius=17, z=1)
    rounded(ax, x + 14, y + 12, 32, 24, color, color, lw=0, radius=8, z=2)
    label(ax, x + 30, y + 24, str(number), 10, "#FFFFFF", "bold")
    label(ax, x + 56, y + 19, title, 11.5, color, "bold", ha="left")
    label(ax, x + 56, y + 36, subtitle, 8.2, "#53657D", ha="left")


def pill(ax, x, y, w, text, fill, stroke, text_color=None, size=8.5):
    rounded(ax, x, y, w, 24, fill, stroke, lw=1.0, radius=12, z=4)
    label(ax, x + w / 2, y + 12, text, size, text_color or stroke, "bold", z=5)


def matrix(ax, x, y, rows, cols, cell, colors, border, title, row_text, col_text):
    for i in range(rows):
        for j in range(cols):
            value = (i * 7 + j * 3 + i * j) % len(colors)
            ax.add_patch(
                Rectangle(
                    (x + j * cell, y + i * cell),
                    cell - 1,
                    cell - 1,
                    facecolor=colors[value],
                    edgecolor="none",
                    zorder=3,
                )
            )
    ax.add_patch(Rectangle((x, y), cols * cell, rows * cell, fill=False, edgecolor=border, linewidth=1.2, zorder=4))
    label(ax, x + cols * cell / 2, y - 15, title, 9.2, border, "bold")
    label(ax, x - 12, y + rows * cell / 2, row_text, 8, border, ha="right")
    label(ax, x + cols * cell / 2, y + rows * cell + 14, col_text, 8, border)


def render(spec):
    W, H = spec["canvas"]["width"], spec["canvas"]["height"]
    p = spec["palette"]
    fig = plt.figure(figsize=(16, 7.8), dpi=100, facecolor=spec["canvas"]["background"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    # Header
    rounded(ax, 28, 22, 8, 52, "#2F6BFF", "#2F6BFF", lw=0, radius=4)
    label(ax, 52, 36, spec["title"], 21, p["ink"], "bold", ha="left")
    label(ax, 52, 62, spec["subtitle"], 10.5, "#566982", ha="left")
    pill(ax, 1438, 28, 130, "EMD-OPD", "#EEF4FF", "#2F6BFF", "#144CC7", 9.5)

    # Module 0: trajectory ribbon
    rounded(ax, 32, 92, 1536, 92, "#F8FAFD", "#C9D3E2", lw=1.2, radius=16)
    pill(ax, 48, 103, 235, "MODULE 0 · ON-POLICY TRAJECTORY", "#EDF1F6", p["neutral"], p["ink"], 9)
    label(ax, 62, 158, "At each stage", 9, p["neutral"], "bold", ha="left")
    rounded(ax, 300, 126, 120, 42, "#FFFFFF", p["neutral"], radius=10)
    label(ax, 360, 147, r"Prompt $x$", 11, p["ink"], "bold")
    arrow(ax, 422, 147, 475, 147, p["neutral"])
    rounded(ax, 480, 117, 205, 60, p["student_light"], p["student"], radius=12)
    label(ax, 582, 139, "Current Student actor", 10.5, p["student"], "bold")
    label(ax, 582, 160, r"$\pi_{\theta}$", 14, p["student"])
    arrow(ax, 687, 147, 745, 147, p["student"])
    rounded(ax, 750, 117, 250, 60, "#FFFFFF", p["student"], radius=12)
    label(ax, 875, 139, "Student-generated response", 10.5, p["ink"], "bold")
    label(ax, 875, 160, r"$y=(y_1,\ldots,y_L)\sim\pi_{\theta}$", 12, p["student"])
    arrow(ax, 1002, 147, 1050, 147, p["neutral"])
    rounded(ax, 1055, 117, 473, 60, "#FFFFFF", "#AEBACC", radius=12)
    label(ax, 1292, 139, "Teacher and Student process the same prompt-response sequence", 9.8, p["ink"], "bold")
    label(ax, 1292, 160, "Losses are evaluated only at valid response-token positions", 9, p["neutral"])

    # Main modules
    module(ax, 25, 212, 230, 430, "FROZEN BRIDGE", "shared low-rank coordinates", p["bridge"], p["bridge_light"], 1)
    module(ax, 275, 212, 690, 430, "LAYER-LEVEL EMD", "representation alignment · Stage 1", p["transport"], p["transport_light"], 2)
    module(ax, 985, 212, 165, 430, "HANDOFF", "final EMD checkpoint", p["success"], p["success_light"], 3)
    module(ax, 1170, 212, 405, 430, "SAMPLED-TOKEN OPD", "output calibration · Stage 2", p["student"], p["student_light"], 4)

    # Bridge interface
    pill(ax, 62, 276, 155, "FROZEN AFTER FITTING", "#FFF8EC", p["bridge"], p["bridge"], 8.2)
    rounded(ax, 50, 320, 180, 74, p["student_light"], p["student"], radius=12)
    label(ax, 140, 342, "Projected Student layers", 9.5, p["student"], "bold")
    label(ax, 140, 370, r"$Z^S=\{z_i^S\}_{i=1}^{28}$", 13, p["student"])
    rounded(ax, 50, 435, 180, 74, p["teacher_light"], p["teacher"], radius=12)
    label(ax, 140, 457, "Detached Teacher layers", 9.5, p["teacher"], "bold")
    label(ax, 140, 485, r"$\mathrm{sg}(Z^T)=\{\mathrm{sg}(z_j^T)\}_{j=1}^{36}$", 11.2, p["teacher"])
    pill(ax, 54, 540, 172, r"$r\in\{8,32,64\}$", "#FFFFFF", p["bridge"], p["bridge"], 10)
    label(ax, 140, 588, r"$P_S,\,P_T$ frozen · all valid response tokens", 8.2, p["neutral"])

    # EMD module: representations -> cost -> flow -> loss
    arrow(ax, 232, 357, 325, 372, p["student"], lw=2.0)
    arrow(ax, 232, 472, 325, 430, p["teacher"], lw=2.0)
    matrix(
        ax,
        330,
        345,
        6,
        8,
        18,
        ["#F2F8F8", "#D2ECEA", "#A9D9D5", "#80C5C0", "#4CA9A5", "#138A8C"],
        p["student"],
        r"Layer cost $C\in\mathbb{R}^{28\times36}$",
        r"Student $i$",
        r"Teacher $j$",
    )
    rounded(ax, 308, 492, 190, 72, "#FFFFFF", "#A7B6C9", radius=10)
    label(ax, 403, 513, "Pairwise response-state cost", 8.8, p["ink"], "bold")
    label(ax, 403, 540, r"$C_{ij}=\mathbb{E}_b\,\operatorname{MSE}_{t\in\mathcal{R}_b}(z^S_{i,t},\mathrm{sg}(z^T_{j,t}))$", 8.9, p["ink"])
    label(ax, 403, 585, "No token×token Gram matrix", 8.5, p["neutral"], "bold")

    arrow(ax, 500, 399, 555, 399, p["transport"], lw=2.0)
    matrix(
        ax,
        565,
        345,
        6,
        8,
        18,
        ["#FBF9FF", "#EFE8FF", "#DDCEFF", "#C4AAFF", "#A478F5", "#7C3AED"],
        p["transport"],
        r"Transport plan $F^*$",
        r"mass $a_i$",
        r"mass $b_j$",
    )
    rounded(ax, 540, 492, 200, 82, p["transport_light"], p["transport"], radius=10)
    label(ax, 640, 512, "Exact balanced OT", 9.4, p["transport"], "bold")
    label(ax, 640, 536, r"$F^*=\arg\min_{F\geq0}\langle F,C\rangle$", 11, p["transport"])
    label(ax, 640, 558, r"$F\mathbf{1}=a,\quad F^{\mathsf{T}}\mathbf{1}=b$", 9.8, p["transport"])
    pill(ax, 573, 585, 134, r"$\tau=1$ · exact LP", "#FFFFFF", p["transport"], p["transport"], 8.5)

    arrow(ax, 712, 399, 770, 399, p["transport"], lw=2.0)
    rounded(ax, 780, 330, 160, 150, p["loss_light"], p["loss"], lw=1.8, radius=14)
    pill(ax, 806, 345, 108, "DETACH PLAN", "#FFFFFF", p["loss"], p["loss"], 8.2)
    label(ax, 860, 392, "EMD representation loss", 9.4, p["loss"], "bold")
    label(ax, 860, 427, r"$\mathcal{L}_{\mathrm{EMD}}=$", 13, p["loss"], "bold")
    label(ax, 860, 454, r"$\lambda_{\mathrm{EMD}}\sum_{i,j}\mathrm{sg}(F^*_{ij})\,C_{ij}$", 10.3, p["loss"])
    label(ax, 860, 503, r"$\lambda_{\mathrm{EMD}}=1$", 9.5, p["loss"], "bold")
    label(ax, 860, 532, "one plan per global mini-batch", 8.1, p["neutral"])

    # Student-only gradient feedback.
    ax.plot([860, 860, 510, 265], [482, 590, 618, 618], color=p["bridge"], linewidth=1.8, linestyle="--", zorder=2)
    arrow(ax, 265, 618, 215, 402, p["bridge"], lw=1.8, style="--", mutation=12, curve=-0.08)
    pill(ax, 533, 611, 250, "GRADIENT → STUDENT BACKBONE ONLY", "#FFF8EC", p["bridge"], p["bridge"], 8.2)
    pill(ax, 790, 611, 150, "Teacher: no_grad", "#F3F5F8", p["neutral"], p["neutral"], 8.2)

    # Handoff
    arrow(ax, 942, 405, 1006, 405, p["success"], lw=2.4)
    rounded(ax, 1010, 320, 115, 168, p["success_light"], p["success"], lw=1.7, radius=16)
    label(ax, 1068, 347, "EMD-aligned", 9.2, p["success"], "bold")
    label(ax, 1068, 370, "Student", 10.5, p["success"], "bold")
    ax.add_patch(Rectangle((1038, 392), 60, 14, facecolor="#FFFFFF", edgecolor=p["success"], linewidth=1.2, zorder=3))
    ax.add_patch(Rectangle((1038, 411), 60, 14, facecolor="#FFFFFF", edgecolor=p["success"], linewidth=1.2, zorder=3))
    ax.add_patch(Rectangle((1038, 430), 60, 14, facecolor="#FFFFFF", edgecolor=p["success"], linewidth=1.2, zorder=3))
    label(ax, 1068, 468, "step 174", 8.7, p["success"], "bold")
    pill(ax, 1005, 520, 125, "MERGE MODEL", "#FFFFFF", p["success"], p["success"], 8)
    label(ax, 1068, 570, "fresh optimizer", 8.5, p["neutral"], "bold")
    label(ax, 1068, 590, "+ fresh cosine schedule", 7.8, p["neutral"])
    arrow(ax, 1127, 405, 1190, 405, p["success"], lw=2.4)

    # OPD module
    rounded(ax, 1195, 278, 355, 60, "#FFFFFF", p["student"], radius=11)
    label(ax, 1372, 298, "New on-policy Student trajectory", 9.5, p["student"], "bold")
    label(ax, 1372, 321, r"$y_t\sim\pi_{\theta}(\cdot\mid s_t)$", 12, p["student"])
    arrow(ax, 1372, 340, 1372, 368, p["neutral"])
    rounded(ax, 1195, 372, 168, 84, p["teacher_light"], p["teacher"], radius=11)
    label(ax, 1279, 392, "Teacher score", 9.2, p["teacher"], "bold")
    label(ax, 1279, 416, r"$\log p_T(y_t\mid s_t)$", 10.8, p["teacher"])
    label(ax, 1279, 439, "frozen · no_grad", 8, p["neutral"])
    rounded(ax, 1382, 372, 168, 84, "#F4F6F9", p["neutral"], radius=11)
    label(ax, 1466, 392, "Old Student score", 9.2, p["neutral"], "bold")
    label(ax, 1466, 416, r"$\log\pi_{\mathrm{old}}(y_t\mid s_t)$", 10.2, p["neutral"])
    label(ax, 1466, 439, "frozen within update", 8, p["neutral"])
    arrow(ax, 1279, 458, 1279, 482, p["teacher"])
    arrow(ax, 1466, 458, 1466, 482, p["neutral"])
    rounded(ax, 1195, 486, 355, 72, "#FFFFFF", "#9FB0C5", radius=11)
    label(ax, 1372, 505, r"$A_t=\mathrm{sg}[\log p_T(y_t\mid s_t)-\log\pi_{\mathrm{old}}(y_t\mid s_t)]$", 9.1, p["ink"])
    label(ax, 1372, 536, r"$\rho_t=\pi_{\theta}(y_t\mid s_t)/\pi_{\mathrm{old}}(y_t\mid s_t)$", 9.4, p["student"])
    arrow(ax, 1372, 560, 1372, 578, p["loss"])
    rounded(ax, 1195, 576, 355, 51, p["loss_light"], p["loss"], radius=10)
    label(ax, 1372, 589, "Sampled-token clipped objective → update Student", 8.4, p["loss"], "bold")
    label(ax, 1372, 610, r"$\mathcal{L}_{\mathrm{OPD}}=-\mathbb{E}[\min(\rho_tA_t,\,\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)]$", 7.9, p["loss"])
    label(ax, 1517, 621, r"$\epsilon=0.2$", 7.8, p["loss"], ha="right")

    # Footer schedule strip
    rounded(ax, 32, 674, 1536, 76, "#F7F9FC", "#CCD6E4", lw=1.2, radius=14)
    pill(ax, 52, 690, 118, "KEY DESIGN", "#FFFFFF", p["neutral"], p["ink"], 8.5)
    rounded(ax, 190, 688, 380, 42, p["transport_light"], p["transport"], radius=11)
    label(ax, 380, 709, r"Stage 1: $\mathcal{L}=\lambda_{\mathrm{EMD}}\mathcal{L}_{\mathrm{EMD}}$ only", 10.3, p["transport"], "bold")
    arrow(ax, 578, 709, 650, 709, p["success"], lw=2.0)
    pill(ax, 660, 697, 230, "CHECKPOINT INITIALIZATION", p["success_light"], p["success"], p["success"], 8.4)
    arrow(ax, 900, 709, 972, 709, p["success"], lw=2.0)
    rounded(ax, 982, 688, 390, 42, p["student_light"], p["student"], radius=11)
    label(ax, 1177, 709, r"Stage 2: $\mathcal{L}=\mathcal{L}_{\mathrm{OPD}}$ only", 10.3, p["student"], "bold")
    pill(ax, 1390, 697, 150, "NO JOINT LOSS", "#FFFFFF", p["loss"], p["loss"], 8.5)
    label(ax, 800, 739, "Representation alignment supplies initialization; output calibration remains the original OPD code path.", 8.7, p["neutral"])

    return fig


def main():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    fig = render(spec)
    (ROOT / "vector").mkdir(parents=True, exist_ok=True)
    (ROOT / "previews").mkdir(parents=True, exist_ok=True)
    svg_path = ROOT / "vector" / "emd_opd_pipeline.svg"
    fig.savefig(svg_path, format="svg", facecolor="white")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(ROOT / "vector" / "emd_opd_pipeline.pdf", format="pdf", facecolor="white")
    fig.savefig(ROOT / "previews" / "emd_opd_pipeline.png", format="png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
