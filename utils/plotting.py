import ast
import json
import textwrap
import numpy as np
from pathlib import Path
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

from animals_utils import SUBLIMINAL_PROMPT_TEMPLATES, RELATION_MAP, SYNONYM_GROUPS
from data_loading import (
    get_common_animals, order_items,
    calculate_correlation_matrices_by_animal,
    calculate_correlation_matrices_by_relation,
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "Qwen2.5-7B-Instruct"

SYNONYM_PAIR_KEYS = {
    frozenset((a1[0], a2[0]))
    for group in SYNONYM_GROUPS.values()
    for i, a1 in enumerate(group)
    for a2 in group[i + 1:]
}


def wrap_text(text, max_length=50):
    segments = text.split('<br>')
    lines = []
    for segment in segments:
        words = segment.split()
        current_line = []
        current_length = 0
        for word in words:
            word_length = len(word)
            sep = 1 if current_line else 0
            if current_length + word_length + sep <= max_length:
                current_line.append(word)
                current_length += word_length + sep
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
        if current_line or segment == "":
            lines.append(' '.join(current_line))
    return '<br>'.join(lines)


def format_prompt_example(combination, example_number="X"):
    template_type = combination["template_type"]
    number_relation = combination["number_relation"]
    animal_relation = combination["animal_relation"]

    if template_type in SUBLIMINAL_PROMPT_TEMPLATES:
        prompt_template = SUBLIMINAL_PROMPT_TEMPLATES[template_type]
        system_msg = prompt_template.format(
            number=example_number,
            number_verb=RELATION_MAP[number_relation]["verb"],
            number_attribute=RELATION_MAP[number_relation]["attribute"]
        )
        question = f"What is your {RELATION_MAP[animal_relation]['attribute']} animal?"
        answer = f"My {RELATION_MAP[animal_relation]['attribute']} animal is the"

        system_wrapped = wrap_text(system_msg, max_length=55)
        question_wrapped = wrap_text(question, max_length=55)
        answer_wrapped = wrap_text(answer, max_length=55)

        return f"<b>System:</b> {system_wrapped}<br><b>User:</b> {question_wrapped}<br><b>Assistant:</b> {answer_wrapped}"
    return f"{template_type}_{number_relation}_{animal_relation}"


def get_animal_color_map(animals):
    colors = px.colors.qualitative.Plotly
    return {animal: colors[idx % len(colors)] for idx, animal in enumerate(animals)}


def _infer_scatter_figsize(
    plot_df,
    x_col,
    y_col,
    figsize=None,
    include_legend=False,
    base_height=2.9,
    min_width=3.4,
    max_width=5.7,
):
    if figsize is not None:
        return figsize

    x_values = pd.to_numeric(plot_df[x_col], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(plot_df[y_col], errors="coerce").to_numpy(dtype=float)

    x_values = x_values[np.isfinite(x_values)]
    y_values = y_values[np.isfinite(y_values)]

    if len(x_values) == 0 or len(y_values) == 0:
        return (4.8, base_height)

    x_min = min(np.min(x_values), 0.0)
    x_max = max(np.max(x_values), 0.0)
    y_min = min(np.min(y_values), 0.0)
    y_max = max(np.max(y_values), 0.0)

    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    aspect_ratio = np.clip(x_span / y_span, 1.15, 2.35)
    width = float(np.clip(base_height * aspect_ratio, min_width, max_width))

    if include_legend:
        width = min(width + 0.3, max_width + 0.3)

    return (width, base_height)


# --- Scatter plots ---

def _load_prompt_text(combination):
    rs = combination["response_start"]
    tt = combination["template_type"]
    nr = combination["number_relation"]
    ar = combination["animal_relation"]
    path = RESULTS_DIR / "subliminal_prompting" / f"{rs}_{tt}_{nr}_{ar}.txt"
    if path.exists():
        return path.read_text().strip()
    return None


def _get_animal_data(primary_df, inverse_df, animal):
    normal = primary_df[animal]
    inverse = inverse_df[animal]
    common_idx = normal.index.intersection(inverse.index)
    x_vals = normal.loc[common_idx].values.tolist()
    y_vals = inverse.loc[common_idx].values.tolist()
    num_labels = common_idx.tolist()
    return x_vals, y_vals, num_labels


def _normalized_label_pair(a, b):
    return tuple(sorted((str(a).strip().lower(), str(b).strip().lower())))


def _label_pair_columns(df):
    if {"animal_1", "animal_2"}.issubset(df.columns):
        return ("animal_1", "animal_2")
    if {"relation_1", "relation_2"}.issubset(df.columns):
        return ("relation_1", "relation_2")
    return None


def _collect_labeled_points(plot_df, label_pairs):
    if not label_pairs:
        return []

    pair_cols = _label_pair_columns(plot_df)
    if pair_cols is None:
        return []

    label_keys = {_normalized_label_pair(a, b) for a, b in label_pairs}
    seen = set()
    points = []
    for _, row in plot_df.iterrows():
        a = str(row[pair_cols[0]]).strip()
        b = str(row[pair_cols[1]]).strip()
        key = _normalized_label_pair(a, b)
        if key not in label_keys or key in seen:
            continue
        points.append((a, b, row))
        seen.add(key)

    return points


def _add_annotation_box(fig, text):
    fig.add_annotation(
        text=text,
        xref="paper", yref="paper",
        x=1.1, y=0.5,
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        align="left",
        font=dict(size=18),
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=10,
    )


def _standard_layout(fig):
    fig.update_layout(
        xaxis=dict(domain=[0, 0.74], scaleanchor="y", scaleratio=1),
        yaxis=dict(constrain="domain"),
        legend=dict(
            x=0.75, y=0.5,
            xanchor="left", yanchor="middle",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="gray", borderwidth=1,
        ),
        margin=dict(r=650),
    )


def _legend_layout(fig):
    fig.update_layout(
        legend=dict(
            x=0.75, y=0.5,
            xanchor="left", yanchor="middle",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="gray", borderwidth=1,
        ),
        margin=dict(r=650),
    )


def _scatter_groups(
    groups,
    x_label, y_label,
    title="", note_text="",
    jitter_x=0.05, jitter_y=0.0,
    color_map=None,
    lock_aspect=False,
    x_range=None, y_range=None,
    height=800, width=1500,
    hover_texts=None,
    groups_for_regression=None,
    vlines=None,
):
    """Core grouped scatter.

    groups: {label: (x_vals, y_vals)} — data shown in the plot.
    groups_for_regression: optional {label: (x_vals, y_vals)} — used for correlation and
        regression line fitting instead of groups (e.g. when threshold-filtered).
    hover_texts: {label: [str_per_point]} — shown on hover.
    vlines: [{"x": val, "color": c, "label": text}] — vertical reference lines.
    """
    reg_groups = groups_for_regression if groups_for_regression is not None else groups

    if color_map is None:
        colors = px.colors.qualitative.Plotly
        color_map = {label: colors[idx % len(colors)] for idx, label in enumerate(groups)}

    correlations = {}
    for label, (x_vals, y_vals) in reg_groups.items():
        if len(x_vals) > 2:
            corr, p_val = pearsonr(x_vals, y_vals)
            correlations[label] = {"corr": corr, "p_val": p_val, "n": len(x_vals)}

    x_all, y_all, group_all, hover_all = [], [], [], []
    for label, (x_vals, y_vals) in groups.items():
        x_jit = [x + np.random.normal(0, jitter_x) for x in x_vals]
        y_jit = [y + np.random.normal(0, jitter_y) for y in y_vals] if jitter_y > 0 else list(y_vals)
        x_all.extend(x_jit)
        y_all.extend(y_jit)
        group_all.extend([label] * len(x_vals))
        texts = hover_texts.get(label, []) if hover_texts else []
        hover_all.extend(texts if texts else [""] * len(x_vals))

    df_plot = pd.DataFrame({"x": x_all, "y": y_all, "group": group_all, "hover": hover_all})
    fig = px.scatter(
        df_plot, x="x", y="y", color="group",
        color_discrete_map=color_map,
        custom_data=["hover"],
        labels={"x": x_label, "y": y_label, "group": ""},
        title=title, height=height, width=width,
    )
    fig.update_traces(
        marker=dict(opacity=0.5, size=3),
        hovertemplate='%{customdata[0]}<br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>',
    )

    if lock_aspect:
        _standard_layout(fig)
    else:
        _legend_layout(fig)

    x_arr = np.array(x_range if x_range else [
        min(x_all) - 1 if x_all else -12,
        max(x_all) + 1 if x_all else 12,
    ])
    for label in groups:
        if label not in correlations:
            continue
        x_vals, y_vals = reg_groups[label]
        if len(x_vals) > 1:
            slope, intercept, *_ = linregress(x_vals, y_vals)
            corr_text = f"{label}: r={correlations[label]['corr']:.3f}, p={correlations[label]['p_val']:.4f}"
            fig.add_scatter(
                x=x_arr, y=slope * x_arr + intercept,
                mode="lines", name=corr_text,
                line=dict(color=color_map.get(label, "gray"), dash="dash", width=2),
                hovertemplate="Trend: " + corr_text + "<extra></extra>",
            )

    if vlines:
        for vl in vlines:
            fig.add_vline(
                x=vl["x"],
                line_color=vl.get("color", "gray"),
                line_dash="dot", line_width=2,
                annotation_text=vl.get("label", ""),
                annotation_position="top",
                annotation_font_size=10,
                annotation_font_color=vl.get("color", "gray"),
            )

    if note_text:
        _add_annotation_box(fig, note_text)
    if x_range:
        fig.update_xaxes(range=list(x_range))
    if y_range:
        fig.update_yaxes(range=list(y_range))

    fig.show()

    return correlations


def scatter_relation_pair_avg_r_vs_unembedding_cosine(
    pair_df,
    x_col="avg_r",
    y_col="cosine_similarity",
    figsize=None,
    alpha=0.7,
    size=10,
    title="Single-token relation pairs: average r vs unembedding cosine",
    x_axis_text="Average Pearson r across animals (relation pair)",
    y_axis_text="Cosine similarity (relation unembedding vectors)",
    label_pairs=None,
):
    if pair_df is None or pair_df.empty:
        print("No relation-pair data available for plotting.")
        return

    plot_df = pair_df.copy()
    if "pair" not in plot_df.columns and {"relation_1", "relation_2"}.issubset(plot_df.columns):
        plot_df["pair"] = plot_df["relation_1"] + " ↔ " + plot_df["relation_2"]

    hover_data = {
        x_col: ":.4f",
        y_col: ":.4f",
    }
    for optional_col in ["n_animals", "n_relations", "relation_1", "relation_2", "animal_1", "animal_2"]:
        if optional_col in plot_df.columns:
            hover_data[optional_col] = True

    color_args = {}
    if {"animal_1", "animal_2"}.issubset(plot_df.columns):
        plot_df["pair_type"] = plot_df.apply(
            lambda row: "synonym pair"
            if frozenset((row["animal_1"], row["animal_2"])) in SYNONYM_PAIR_KEYS
            else "other pair",
            axis=1,
        )
        color_args = {
            "color": "pair_type",
            "color_discrete_map": {
                "synonym pair": "green",
                "other pair": "#636EFA",
            },
        }

    width, height = _infer_scatter_figsize(
        plot_df,
        x_col=x_col,
        y_col=y_col,
        figsize=figsize,
        include_legend={"animal_1", "animal_2"}.issubset(plot_df.columns),
    )

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        hover_name="pair" if "pair" in plot_df.columns else None,
        hover_data=hover_data,
        title=title,
        **color_args,
    )
    fig.update_traces(marker=dict(size=size, opacity=alpha))
    fig.add_vline(x=0, line_color="gray", line_dash="dot", line_width=1)
    fig.add_hline(y=0, line_color="gray", line_dash="dot", line_width=1)
    fig.update_xaxes(title_text=x_axis_text)
    fig.update_yaxes(title_text=y_axis_text)
    fig.update_layout(
        width=int(width * 120),
        height=int(height * 120),
    )

    labeled_points = _collect_labeled_points(plot_df, label_pairs)
    if labeled_points:
        labeled_x = []
        labeled_y = []
        for a, b, row in labeled_points:
            fig.add_annotation(
                x=row[x_col],
                y=row[y_col],
                text=f"{a} ↔ {b}",
                showarrow=True,
                arrowhead=2,
                ax=12,
                ay=10,
                arrowcolor="black",
                bgcolor="rgba(255,255,255,0.75)",
                borderpad=2,
            )
            labeled_x.append(row[x_col])
            labeled_y.append(row[y_col])

        fig.add_scatter(
            x=labeled_x,
            y=labeled_y,
            mode="markers",
            showlegend=False,
            hoverinfo="skip",
            marker=dict(
                size=size + 4,
                color="rgba(0,0,0,0)",
                line=dict(color="black", width=1.5),
            ),
        )

    fig.show()


def scatter_relation_pair_avg_r_vs_unembedding_cosine_export(
    pair_df,
    x_col="avg_r",
    y_col="cosine_similarity",
    x_axis_text="Average Pearson r across animals (relation pair)",
    y_axis_text="Cosine similarity (relation unembedding vectors)",
    figsize=None,
    alpha=0.7,
    size=14,
    export_path=None,
    dpi=300,
    label_pairs=None,
):
    if pair_df is None or pair_df.empty:
        print("No relation-pair data available for plotting.")
        return

    plot_df = pair_df.copy()
    if "pair" not in plot_df.columns and {"relation_1", "relation_2"}.issubset(plot_df.columns):
        plot_df["pair"] = plot_df["relation_1"] + " ↔ " + plot_df["relation_2"]

    pair_type_present = {"animal_1", "animal_2"}.issubset(plot_df.columns)
    if pair_type_present:
        plot_df["pair_type"] = plot_df.apply(
            lambda row: "synonym pair"
            if frozenset((row["animal_1"], row["animal_2"])) in SYNONYM_PAIR_KEYS
            else "other pair",
            axis=1,
        )

    inferred_figsize = _infer_scatter_figsize(
        plot_df,
        x_col=x_col,
        y_col=y_col,
        figsize=figsize,
        include_legend=pair_type_present,
    )

    fig, ax = plt.subplots(figsize=inferred_figsize)

    if pair_type_present:
        color_map = {
            "synonym pair": "green",
            "other pair": "#636EFA",
        }
        for pair_type, sub_df in plot_df.groupby("pair_type"):
            ax.scatter(
                sub_df[x_col],
                sub_df[y_col],
                alpha=alpha,
                s=size,
                color=color_map.get(pair_type, "#636EFA"),
                label=pair_type,
                rasterized=True,
            )
        ax.legend(fontsize=8, loc="best", framealpha=0.9, borderpad=0.5)
    else:
        ax.scatter(
            plot_df[x_col],
            plot_df[y_col],
            alpha=alpha,
            s=size,
            color="#636EFA",
            rasterized=True,
        )

    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel(x_axis_text, fontsize=10)
    ax.set_ylabel(y_axis_text, fontsize=10)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    labeled_points = _collect_labeled_points(plot_df, label_pairs)
    if labeled_points:
        labeled_x = []
        labeled_y = []
        for a, b, row in labeled_points:
            ax.annotate(
                f"{a} ↔ {b}",
                xy=(row[x_col], row[y_col]),
                xytext=(6, -6),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
            )
            labeled_x.append(row[x_col])
            labeled_y.append(row[y_col])

        ax.scatter(
            labeled_x,
            labeled_y,
            s=size * 2.0,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

    plt.tight_layout()

    if export_path:
        output_path = Path(export_path)
        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, format="png", bbox_inches="tight", pad_inches=0.05)
        print(f"Saved to {output_path}")

    plt.show()

    return fig, ax


def scatter_logprob_vs_logprob(x_diff_df, y_diff_df, x_combination, y_combination,
                                animals=None, note=None):
    shared_animals = set(x_diff_df.columns).intersection(y_diff_df.columns)
    if animals is None:
        common_animals = sorted(shared_animals)
    else:
        common_animals = [a for a in animals if a in shared_animals]

    if not common_animals:
        print("No animals available in both combinations.")
        return

    groups = {}
    for animal in common_animals:
        x_vals, y_vals, _ = _get_animal_data(x_diff_df, y_diff_df, animal)
        groups[animal] = (x_vals, y_vals)

    comb_str = lambda c: f"{c['template_type']} {c['number_relation']} {c['animal_relation']}"
    x_prompt_raw = _load_prompt_text(x_combination)
    y_prompt_raw = _load_prompt_text(y_combination)
    x_prompt = wrap_text(x_prompt_raw.replace("\n", "<br>"), max_length=55) if x_prompt_raw else ""
    y_prompt = wrap_text(y_prompt_raw.replace("\n", "<br>"), max_length=55) if y_prompt_raw else ""
    note_text = f"<b>X-axis prompt: (baseline: {x_combination['baseline']})</b><br>{x_prompt}<br><br><b>Y-axis prompt: (baseline: {y_combination['baseline']})</b><br>{y_prompt}"
    if note:
        note_text = f"<b>Note:</b><br>{wrap_text(note, max_length=55)}<br><br>{note_text}"

    _scatter_groups(
        groups,
        x_label=f"Logprob Difference {comb_str(x_combination)}",
        y_label=f"Logprob Difference {comb_str(y_combination)}",
        title="Qwen2.5-7B-Instruct — Common Animals: Normal vs Inverse Differences (Scatter with Jitter)",
        note_text=note_text,
        jitter_x=0.05, jitter_y=0.05,
        lock_aspect=True, x_range=(-12, 12), y_range=(-12, 12),
    )


def scatter_logprob_vs_logprob_mpl(x_diff_df, y_diff_df, x_combination, y_combination,
                                    animals=None, note=None, save_path=None, figsize=(6, 6)):
    shared_animals = set(x_diff_df.columns).intersection(y_diff_df.columns)
    if animals is None:
        common_animals = sorted(shared_animals)
    else:
        common_animals = [a for a in animals if a in shared_animals]

    if not common_animals:
        print("No animals available in both combinations.")
        return

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(common_animals))]

    if note:
        fig, (ax, ax_note) = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
        ax_note.axis("off")

        def _plain(html):
            return html.replace("<br>", "\n").replace("<b>", "").replace("</b>", "")

        x_prompt_raw = _load_prompt_text(x_combination)
        y_prompt_raw = _load_prompt_text(y_combination)
        x_prompt = x_prompt_raw or ""
        y_prompt = y_prompt_raw or ""
        panel_text = (
            _plain(note)
            + f"\n\n--- X-axis prompt (baseline: {x_combination['baseline']}) ---\n"
            + x_prompt
            + f"\n\n--- Y-axis prompt (baseline: {y_combination['baseline']}) ---\n"
            + y_prompt
        )
        wrapped_text = "\n".join(
            textwrap.fill(line, width=52) if line.strip() else line
            for line in panel_text.split("\n")
        )
        ax_note.text(
            0.05, 0.95, wrapped_text,
            transform=ax_note.transAxes,
            fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="lightgray"),
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)

    correlations = {}
    for idx, animal in enumerate(common_animals):
        x_vals, y_vals, _ = _get_animal_data(x_diff_df, y_diff_df, animal)
        if not x_vals:
            continue
        color = colors[idx]
        ax.scatter(x_vals, y_vals, color=color, alpha=0.4, s=6, label=animal, rasterized=True)
        if len(x_vals) > 2:
            corr, p_val = pearsonr(x_vals, y_vals)
            slope, intercept, *_ = linregress(x_vals, y_vals)
            correlations[animal] = {"corr": corr, "p_val": p_val, "n": len(x_vals)}
            x_line = np.array([-12.0, 12.0])
            ax.plot(x_line, slope * x_line + intercept, color=color, linestyle="--", linewidth=1.0, alpha=0.8)

    comb_str = lambda c: f"{c['template_type']} / {c['number_relation']} / {c['animal_relation']}"
    ax.set_xlabel(f"$\\Delta\\log p$ [{comb_str(x_combination)}]", fontsize=10)
    ax.set_ylabel(f"$\\Delta\\log p$ [{comb_str(y_combination)}]", fontsize=10)
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    handles, leg_labels = ax.get_legend_handles_labels()
    leg_labels = [
        f"{lbl}  $r={correlations[lbl]['corr']:+.3f}$" if lbl in correlations else lbl
        for lbl in leg_labels
    ]
    ax.legend(handles, leg_labels, fontsize=7, loc="best", framealpha=0.9, borderpad=0.5)

    if note:
        fig.subplots_adjust(wspace=0.3)
    else:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()

    return fig, ax


def scatter_logprob_vs_logprob_export(
    x_diff_df,
    y_diff_df,
    x_combination,
    y_combination,
    x_axis_text,
    y_axis_text,
    animals=None,
    export_path=None,
    figsize=(5, 5),
    dpi=300,
):
    shared_animals = set(x_diff_df.columns).intersection(y_diff_df.columns)
    if animals is None:
        common_animals = sorted(shared_animals)
    else:
        common_animals = [a for a in animals if a in shared_animals]

    if not common_animals:
        print("No animals available in both combinations.")
        return

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(common_animals))]

    fig, ax = plt.subplots(figsize=figsize)

    correlations = {}
    for idx, animal in enumerate(common_animals):
        x_vals, y_vals, _ = _get_animal_data(x_diff_df, y_diff_df, animal)
        if not x_vals:
            continue
        ax.scatter(x_vals, y_vals, color=colors[idx], alpha=0.4, s=6, label=animal, rasterized=True)
        if len(x_vals) > 2:
            corr, _ = pearsonr(x_vals, y_vals)
            slope, intercept, *_ = linregress(x_vals, y_vals)
            correlations[animal] = corr
            x_line = np.array([-12.0, 12.0])
            ax.plot(x_line, slope * x_line + intercept, color=colors[idx], linestyle="--", linewidth=1.0, alpha=0.8)

    ax.set_xlabel(x_axis_text, fontsize=10)
    ax.set_ylabel(y_axis_text, fontsize=10)
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal")

    handles, leg_labels = ax.get_legend_handles_labels()
    if handles:
        leg_labels = [
            f"{lbl}  $r={correlations[lbl]:+.3f}$" if lbl in correlations else lbl
            for lbl in leg_labels
        ]
        ax.legend(handles, leg_labels, fontsize=7, loc="best", framealpha=0.9, borderpad=0.5)

    plt.tight_layout()

    if export_path:
        output_path = Path(export_path)
        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", format="png")
        print(f"Saved to {output_path}")

    plt.show()

    return fig, ax


def scatter_single_animal(x_diff_df, y_diff_df, animal, x_combination, y_combination,
                           note=None):
    x_vals, y_vals, num_labels = _get_animal_data(x_diff_df, y_diff_df, animal)

    if len(x_vals) > 2:
        corr, p_val = pearsonr(x_vals, y_vals)
        slope, intercept, *_ = linregress(x_vals, y_vals)
    else:
        corr, p_val = None, None
        slope, intercept = None, None

    jitter_strength = 0.05
    x_jittered = [x + np.random.normal(0, jitter_strength) for x in x_vals]
    y_jittered = [y + np.random.normal(0, jitter_strength) for y in y_vals]

    fig = px.scatter(
        x=x_jittered, y=y_jittered,
        labels={
            "x": f"Logprob Difference {x_combination['template_type']} {x_combination['number_relation']} {x_combination['animal_relation']}",
            "y": f"Logprob Difference {y_combination['template_type']} {y_combination['number_relation']} {y_combination['animal_relation']}",
        },
        title=f"Qwen2.5-7B-Instruct — {animal}: Normal vs Inverse Differences" + (f"\nr={corr:.4f}, p={p_val:.4f}" if corr is not None else ""),
        height=800, width=1500,
    )
    fig.update_traces(
        marker=dict(opacity=0.5, size=6),
        text=num_labels,
        hovertemplate='Number: %{text}<br>Normal: %{x:.4f}<br>Inverse: %{y:.4f}<extra></extra>',
    )
    _standard_layout(fig)

    if corr is not None and slope is not None:
        x_line = np.array([-12, 12])
        y_line = slope * x_line + intercept
        line_style = 'solid' if p_val < 0.05 else 'dash'
        line_width = 3 if p_val < 0.05 else 2
        fig.add_scatter(
            x=x_line, y=y_line,
            mode='lines', name=f'Trend (p={p_val:.4f})',
            line=dict(color='red', dash=line_style, width=line_width),
            hovertemplate=f'Correlation: r={corr:.4f}<br>p-value: {p_val:.4f}<extra></extra>',
        )

    x_prompt = format_prompt_example(x_combination, "X")
    y_prompt = format_prompt_example(y_combination, "Y")
    prompt_note = f"<b>X-axis prompt:</b><br>{x_prompt}<br><br><b>Y-axis prompt:</b><br>{y_prompt}"
    if note:
        prompt_note = f"<b>Note:</b><br>{wrap_text(note, max_length=45)}<br><br>{prompt_note}"
    _add_annotation_box(fig, prompt_note)

    fig.update_xaxes(range=[-12, 12])
    fig.update_yaxes(range=[-12, 12])
    fig.show()


def scatter_animal_vs_animal(diff_df, animal_pairs, combination, note=None):
    """Scatter logprob diffs of one animal against another within the same combination.

    animal_pairs: list of (animal_x, animal_y) tuples. Each pair is a colored group.
        Points are the number tokens. Hover shows the number string.
        Useful for seeing whether the same numbers drive the logprob change for two animals.
    """
    groups = {}
    hover_texts = {}
    for animal_x, animal_y in animal_pairs:
        if animal_x not in diff_df.columns or animal_y not in diff_df.columns:
            print(f"Skipping ({animal_x}, {animal_y}): not in diff_df")
            continue
        label = f"{animal_x} vs {animal_y}"
        groups[label] = (diff_df[animal_x].values.tolist(), diff_df[animal_y].values.tolist())
        hover_texts[label] = [str(n) for n in diff_df.index]

    if not groups:
        print("No valid animal pairs found in diff_df.")
        return

    comb_str = f"{combination['template_type']} / {combination['number_relation']} / {combination['animal_relation']}"

    if len(animal_pairs) == 1:
        a_x, a_y = animal_pairs[0]
        x_label = f"Logprob diff — {a_x} ({comb_str})"
        y_label = f"Logprob diff — {a_y} ({comb_str})"
    else:
        x_label = f"Logprob diff — animal X ({comb_str})"
        y_label = f"Logprob diff — animal Y ({comb_str})"

    x_prompt = format_prompt_example(combination, "X")
    note_text = f"<b>Combination: (baseline: {combination['baseline']})</b><br>{x_prompt}<br><br>Each point is a number token. Hover to see which number."
    if note:
        note_text = f"<b>Note:</b><br>{wrap_text(note, max_length=55)}<br><br>{note_text}"

    _scatter_groups(
        groups,
        x_label=x_label, y_label=y_label,
        title=f"Animal vs Animal Logprob Difference — {comb_str}",
        note_text=note_text,
        jitter_x=0.05, jitter_y=0.05,
        lock_aspect=True, x_range=(-12, 12), y_range=(-12, 12),
        hover_texts=hover_texts,
    )


def scatter_logprob_vs_metric(diff_df, metric_per_animal, x_combination, x_label, y_label,
                               animals, note=None, color_map=None, show_mean_vlines=False,
                               min_threshold=None):
    """Generalized scatter: x = logprob difference, y = any per-animal metric.

    metric_per_animal: {animal: list_of_y_values} aligned row-for-row with diff_df.
    min_threshold: if set, only include points where y >= threshold for correlation/regression
                   (but still plot all points).
    """
    dataset_animals = [a for a in animals if a in diff_df.columns and a in metric_per_animal]
    if color_map is None:
        color_map = get_animal_color_map(animals)

    groups_all = {}
    groups_filtered = {} if min_threshold is not None else None

    for animal in dataset_animals:
        x_vals = diff_df[animal].values.tolist()
        y_vals = list(metric_per_animal[animal])
        groups_all[animal] = (x_vals, y_vals)

        if min_threshold is not None:
            x_f = [x for x, y in zip(x_vals, y_vals) if y >= min_threshold]
            y_f = [y for y in y_vals if y >= min_threshold]
            groups_filtered[animal] = (x_f, y_f)

    reg_groups = groups_filtered if groups_filtered is not None else groups_all
    vlines = None
    if show_mean_vlines:
        vlines = []
        for animal in dataset_animals:
            x_vals, _ = reg_groups[animal]
            if x_vals:
                mean_x = float(np.mean(x_vals))
                vlines.append({"x": mean_x, "color": color_map.get(animal, "gray"),
                                "label": f"{animal} μ={mean_x:.2f}"})

    x_prompt = format_prompt_example(x_combination, "X")
    threshold_note = f"<br>Min threshold: {min_threshold}" if min_threshold is not None else ""
    note_text = f"<b>X-axis prompt: (baseline: {x_combination['baseline']})</b><br>{x_prompt}<br><br><b>Y-axis:</b> {y_label}{threshold_note}"
    if note:
        note_text = f"<b>Note:</b><br>{wrap_text(note, max_length=55)}<br><br>{note_text}"

    _scatter_groups(
        groups_all,
        x_label=x_label, y_label=y_label,
        title=f"Qwen2.5-7B-Instruct — {y_label} vs Logprob Difference",
        note_text=note_text,
        jitter_x=0.05, jitter_y=0.0,
        color_map=color_map,
        lock_aspect=False,
        groups_for_regression=groups_filtered,
        vlines=vlines,
    )


# --- Heatmaps ---

def plot_heatmap_grid(matrices, labels, panel_names, title=None, max_cols=3,
                      show_average=True, figsize=None, avg_figsize=None, cmap="RdBu_r",
                      vmin=-1, vmax=1, center=0, annot_size=8, annot_fmt=".2f",
                      print_summary=False, item_order=None, only_average=True):
    """Unified small-multiples heatmap.

    matrices: dict {panel_name: 2D numpy array}
    labels: tick labels for both axes of each heatmap
    panel_names: list of panel names in display order
    item_order: optional ordering for panel_names (overrides the order in panel_names)
    """
    if item_order is not None:
        panel_names = [p for p in item_order if p in matrices] + [p for p in panel_names if p not in item_order and p in matrices]

    n = len(panel_names)
    if n == 0:
        print("No panel names available to plot.")
        return None, None

    fig, axes = None, None
    avg_fig, avg_ax = None, None
    all_matrices = [matrices[name] for name in panel_names]
    avg_mat = np.mean(all_matrices, axis=0) if all_matrices else None

    if not only_average:
        ncols = min(max_cols, n)
        nrows = int(np.ceil(n / ncols))

        if figsize is None:
            label_count = max(1, len(labels))
            panel_w = max(4.6, min(7.0, 3.2 + 0.30 * label_count))
            panel_h = max(3.8, min(6.2, 2.6 + 0.30 * label_count))
            fig_w = min(18.0, panel_w * ncols)
            fig_h = min(14.0, panel_h * nrows + 0.8)
            figsize = (fig_w, fig_h)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = np.atleast_2d(axes)

        for idx, name in enumerate(panel_names):
            r = idx // ncols
            c = idx % ncols
            ax = axes[r, c]
            mat = matrices[name]

            sns.heatmap(
                mat, xticklabels=labels, yticklabels=labels,
                cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                annot=True, fmt=annot_fmt, cbar=False, ax=ax,
                annot_kws={"size": annot_size},
            )
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)

        for idx in range(len(panel_names), nrows * ncols):
            r = idx // ncols
            c = idx % ncols
            axes[r, c].axis('off')

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    if show_average and avg_mat is not None:
        if avg_figsize is None:
            label_count = max(1, len(labels))
            side = max(6.5, min(10.0, 3.8 + 0.48 * label_count))
            avg_figsize = (side, side)

        avg_fig, avg_ax = plt.subplots(figsize=avg_figsize)
        sns.heatmap(
            avg_mat, xticklabels=labels, yticklabels=labels,
            cmap=cmap, center=center, vmin=vmin, vmax=vmax,
            annot=True, fmt=annot_fmt, cbar=True, ax=avg_ax,
            annot_kws={"size": max(annot_size, 9)},
            cbar_kws={"label": "Pearson r"},
        )
        avg_ax.tick_params(axis='x', rotation=45)
        avg_ax.tick_params(axis='y', rotation=0)
        plt.tight_layout()
        plt.show()

    if print_summary and show_average and avg_mat is not None:
        print(f"\n=== Average Matrix (across {len(panel_names)} panels) ===")
        print(f"{'Label 1':<15} {'Label 2':<15} {'avg':<10}")
        print("-" * 45)
        for i, l1 in enumerate(labels):
            for j, l2 in enumerate(labels):
                if j >= i:
                    print(f"{l1:<15} {l2:<15} {avg_mat[i, j]:<10.4f}")

    if only_average and avg_fig is not None:
        return avg_fig, np.array([[avg_ax]])

    return fig, axes


def heatmap_correlations_by_animal(all_relation_data, animals=None, animal_order=None,
                                    relation_order=None, title=None, only_average=True, **kwargs):
    correlation_by_animal = calculate_correlation_matrices_by_animal(all_relation_data)

    if animals is not None:
        animals_list = [a for a in animals if a in correlation_by_animal]
    else:
        all_animals = sorted(correlation_by_animal.keys())
        animals_list = order_items(all_animals, animal_order)

    if relation_order is None:
        relations_list = sorted(next(iter(correlation_by_animal.values())).keys())
    else:
        relations_list = [r for r in relation_order if r in next(iter(correlation_by_animal.values())).keys()]

    matrices = {}
    for animal in animals_list:
        mat = np.zeros((len(relations_list), len(relations_list)))
        corr_matrix = correlation_by_animal[animal]
        for i, rel1 in enumerate(relations_list):
            for j, rel2 in enumerate(relations_list):
                if rel1 in corr_matrix and rel2 in corr_matrix[rel1]:
                    mat[i, j] = corr_matrix[rel1][rel2]["r"]
        matrices[animal] = mat

    if title is None:
        title = "Correlation Matrix per Animal: Logprob Differences for Different Relations"

    return plot_heatmap_grid(
        matrices,
        relations_list,
        animals_list,
        title=title,
        only_average=only_average,
        **kwargs,
    )


def heatmap_correlations_by_relation(all_relation_data, animals=None, animal_order=None,
                                      relation_order=None, title=None, only_average=True, **kwargs):
    correlation_by_relation = calculate_correlation_matrices_by_relation(all_relation_data, animals=animals)

    if relation_order is None:
        relations_list = sorted(correlation_by_relation.keys())
    else:
        relations_list = [r for r in relation_order if r in correlation_by_relation]

    first_corr = correlation_by_relation[relations_list[0]]
    if animals is not None:
        animals_list = [a for a in animals if a in first_corr]
    else:
        animals_list = order_items(list(first_corr.keys()), animal_order)

    matrices = {}
    for relation in relations_list:
        mat = np.zeros((len(animals_list), len(animals_list)))
        corr_matrix = correlation_by_relation[relation]
        for i, a1 in enumerate(animals_list):
            for j, a2 in enumerate(animals_list):
                if a1 in corr_matrix and a2 in corr_matrix[a1]:
                    mat[i, j] = corr_matrix[a1][a2]["r"]
        matrices[relation] = mat

    if title is None:
        title = "Correlation Matrix per Relation: Logprob Differences for Different Animals"

    return plot_heatmap_grid(
        matrices,
        animals_list,
        relations_list,
        title=title,
        only_average=only_average,
        **kwargs,
    )


def _save_figure(fig, export_path, dpi=300):
    if fig is None:
        print("No figure available to save")
        return None

    output_path = Path(export_path)
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", format="png")
    print(f"Saved to {output_path}")
    return output_path


def heatmap_correlations_by_animal_export(
    all_relation_data,
    export_path,
    animals=None,
    animal_order=None,
    relation_order=None,
    title=None,
    dpi=300,
    only_average=True,
    **kwargs,
):
    fig, _ = heatmap_correlations_by_animal(
        all_relation_data,
        animals=animals,
        animal_order=animal_order,
        relation_order=relation_order,
        title=title,
        only_average=only_average,
        **kwargs,
    )
    return _save_figure(fig, export_path, dpi=dpi)


def heatmap_correlations_by_relation_export(
    all_relation_data,
    export_path,
    animals=None,
    animal_order=None,
    relation_order=None,
    title=None,
    dpi=300,
    only_average=True,
    **kwargs,
):
    fig, _ = heatmap_correlations_by_relation(
        all_relation_data,
        animals=animals,
        animal_order=animal_order,
        relation_order=relation_order,
        title=title,
        only_average=only_average,
        **kwargs,
    )
    return _save_figure(fig, export_path, dpi=dpi)


def plot_similarity_heatmap(tokens, unembedding_df, title=None, cmap="RdBu_r"):
    if unembedding_df is None:
        print("Unembedding vectors not loaded")
        return

    token_vectors = {}
    labels = {}

    for token in tokens:
        if token not in unembedding_df.index:
            print(f"  Missing '{token}' in unembedding vectors")
            continue
        row = unembedding_df.loc[token]
        token_list = ast.literal_eval(row["tokens"])
        vectors = np.array(json.loads(row["vector"]))
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]
        token_vectors[token] = vectors
        labels[token] = str(token_list)

    if len(token_vectors) < 2:
        print(f"Not enough tokens found. Found: {list(token_vectors.keys())}")
        return

    tokens_found = list(token_vectors.keys())
    axis_labels = [labels[t] for t in tokens_found]

    n = len(tokens_found)
    sim_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(token_vectors[tokens_found[i]],
                                    token_vectors[tokens_found[j]]).mean()
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    fig_size = max(6, n * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        sim_matrix, xticklabels=axis_labels, yticklabels=axis_labels,
        cmap=cmap, center=0, vmin=-1, vmax=1,
        annot=True, fmt=".3f",
        cbar_kws={"label": "Cosine Similarity"},
        ax=ax, annot_kws={"size": 9},
    )

    if title is None:
        dim = token_vectors[tokens_found[0]].shape[-1]
        title = f"Cosine Similarity Between Tokens\n(Based on Unembedding Vectors: {dim} dimensions)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Tokens", fontsize=12)
    ax.set_ylabel("Tokens", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return fig, ax
