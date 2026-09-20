"""Step 4 - every figure used in the report.

Print media, so the figures commit to the light surface. Community identity is
carried by facets and direct labels rather than by eight categorical hues,
because eight slots cannot clear the all-pairs colour-blind separation gate.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import config as C
import vizstyle as V

V.apply_style()


# ------------------------------------------------------------------- inputs
def load():
    G = nx.read_graphml(C.NETWORK / "opinion_network_annotated.graphml")
    df = pd.read_csv(C.PROCESSED / "responses_numeric.csv", index_col="respondent")
    df.index = df.index.astype(str)
    cen = pd.read_csv(C.TABLES / "centralities.csv", dtype={"respondent": str}).set_index(
        "respondent"
    )
    items = pd.read_csv(C.TABLES / "item_statistics.csv")
    build = json.loads((C.NETWORK / "build_report.json").read_text())
    res = json.loads((C.OUT / "analysis_results.json").read_text())
    clean = json.loads((C.PROCESSED / "cleaning_report.json").read_text())
    return G, df, cen, items, build, res, clean


def layout(G, boost: float = 8.0):
    """One layout reused across every view so panels stay comparable.

    Plain spring layout on a graph this dense leaves the communities visually
    interleaved. Intra-community ties are therefore given extra weight *for
    layout purposes only* - it changes where nodes are drawn, never any
    measured quantity - so that groups Louvain found read as groups on the page.
    """
    comm = nx.get_node_attributes(G, "community")
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        same = comm.get(u) == comm.get(v) if comm else False
        H.add_edge(u, v, weight=d["weight"] * (boost if same else 1.0))
    return nx.spring_layout(H, weight="weight", seed=C.RANDOM_SEED, k=0.55, iterations=600)


def save(fig, name):
    path = C.FIGS / name
    fig.savefig(path)
    plt.close(fig)
    print(f"[viz] {name}")
    return path


# ----------------------------------------------------------------- figure 1
def fig_dataset(df, clean, raw_counts):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))

    order = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree",
             "No Comments", "(blank)"]
    vals = [raw_counts.get(k, 0) for k in order]
    total = sum(vals)
    # Ordinal magnitude -> one blue ramp; the two non-responses sit in grey.
    cols = [V.SEQ(x) for x in np.linspace(0.30, 0.95, 5)] + [V.NEUTRAL, "#eceae4"]
    ax = axes[0]
    bars = ax.barh(range(len(order)), vals, color=cols, height=0.68)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("responses (of 5,760 cells)")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + total * 0.012, b.get_y() + b.get_height() / 2,
                f"{v:,} ({v/total:.0%})", va="center", fontsize=7.3, color=V.INK2)
    ax.set_xlim(0, max(vals) * 1.28)
    ax.grid(axis="y", visible=False)
    V.title(ax, "Responses skew heavily toward agreement",
            "72% of all answers are Agree or Stronger; only 6% disagree")

    # Right panel: the raw answered/not-answered mask itself, respondents
    # ordered by completeness. Plotting the mask rather than a summary is what
    # makes the block-wise shape of the drop-out visible.
    ax = axes[1]
    mask = pd.read_csv(C.PROCESSED / "answered_mask.csv", index_col=0)
    comp = pd.read_csv(C.PROCESSED / "respondent_completeness.csv", index_col=0)
    order = comp.sort_values("completeness", ascending=False).index
    M = mask.loc[order].to_numpy()
    ax.imshow(M, aspect="auto", interpolation="nearest",
              cmap=plt.matplotlib.colors.ListedColormap(["#eceae4", V.SEQ(0.72)]),
              vmin=0, vmax=1)
    n_keep = int(comp["retained"].sum())
    ax.axhline(n_keep - 0.5, color=V.CAT[1], lw=1.6, ls="--")
    ax.text(0.6, n_keep + 2.4, f"inclusion cut · {C.MIN_COMPLETENESS:.0%} of items",
            color=V.CAT[1], fontsize=7.3, va="top")
    for i in range(1, len(C.TOPICS)):          # 2px surface gap between blocks
        ax.axvline(i * 15 - 0.5, color=V.SURFACE, lw=1.6)
    ax.set_xticks([7, 22, 37, 52])
    ax.set_xticklabels(["Technology", "Education", "Society", "Environment"], fontsize=7)
    ax.set_ylabel("respondents, ranked by completeness")
    ax.grid(False)
    ax.legend(handles=[Patch(facecolor=V.SEQ(0.72), label="answered"),
                       Patch(facecolor="#eceae4", label="blank / No Comments")],
              loc="lower left", bbox_to_anchor=(0.0, -0.40), ncol=2)
    V.title(ax, "Drop-out is block-wise, not scattered",
            f"{n_keep} respondents kept · "
            f"{int((~comp['retained']).sum())} abandoned whole topic blocks")

    fig.tight_layout()
    return save(fig, "fig1_dataset.png")


# ----------------------------------------------------------------- figure 2
def fig_similarity(build):
    naive = np.load(C.NETWORK / "naive_similarity.npy")
    naive = naive[np.triu_indices_from(naive, k=1)]
    R = np.load(C.NETWORK / "similarity_matrix.npy")
    obs = R[np.triu_indices_from(R, k=1)]
    null = np.load(C.NETWORK / "null_pool_sample.npy")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))

    ax = axes[0]
    ax.hist(naive, bins=44, color=V.CAT[1], alpha=0.85)
    ax.axvline(naive.mean(), color=V.INK, lw=1.4, ls="--")
    ax.text(naive.mean() - 0.004, ax.get_ylim()[1] * 0.93, f"mean {naive.mean():.2f} ",
            ha="right", fontsize=7.3, color=V.INK)
    ax.set_xlabel("raw agreement similarity,  1 − mean|Δ|/4")
    ax.set_ylabel("respondent pairs")
    V.title(ax, "Raw similarity cannot separate anybody",
            f"{build['naive_frac_above_0_7']:.0%} of all pairs score above 0.70")

    ax = axes[1]
    bins = np.linspace(-0.6, 0.6, 70)
    ax.hist(null, bins=bins, density=True, color=V.NEUTRAL,
            label=f"permutation null ({C.NULL_PERMUTATIONS} shuffles)")
    ax.hist(obs, bins=bins, density=True, histtype="step", lw=2.0,
            color=V.CAT[0], label="observed pairs")
    ax.axvline(build["threshold_tau"], color=V.CAT[1], lw=1.8, ls="--")
    ax.text(build["threshold_tau"] + 0.015, ax.get_ylim()[1] * 0.82,
            f"τ = {build['threshold_tau']:.3f}\n(null p95)", color=V.CAT[1], fontsize=7.3)
    ax.set_xlabel("Pearson correlation of double-centred profiles")
    ax.set_ylabel("density")
    ax.legend(loc="upper left")
    V.title(ax, "After double centring, real structure appears",
            "observed distribution is wider and right-shifted against the null")

    fig.tight_layout()
    return save(fig, "fig2_similarity.png")


# ----------------------------------------------------------------- figure 3
def fig_threshold(build):
    sw = pd.read_csv(C.TABLES / "threshold_sweep.csv")
    tau = build["threshold_tau"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9))

    for ax, col, lab, ttl, sub in [
        (axes[0], "edges", "edges retained", "Edge count falls away steeply",
         "the network is dense below 0.20"),
        (axes[1], "giant_component_frac", "largest component (fraction of nodes)",
         "Connectivity survives only below ≈0.25", "beyond it the graph shatters"),
        (axes[2], "avg_clustering", "average clustering",
         "Clustering degrades past the same point", "local structure disappears with it"),
    ]:
        ax.plot(sw["threshold"], sw[col], color=V.CAT[0], marker="o", ms=4.5, zorder=3)
        ax.axvline(tau, color=V.CAT[1], lw=1.6, ls="--", zorder=2)
        ax.set_xlabel("correlation threshold")
        ax.set_ylabel(lab)
        V.title(ax, ttl, sub)
    axes[1].set_ylim(0, 1.08)
    axes[0].text(tau + 0.012, axes[0].get_ylim()[1] * 0.80, f"τ = {tau:.3f}",
                 color=V.CAT[1], fontsize=7.3)

    fig.tight_layout()
    return save(fig, "fig3_threshold.png")



# ----------------------------------------------------------------- figure 4
def draw_edges(ax, G, pos, color="#c9c8c2", lw_scale=1.0, alpha=0.55):
    wmax = max(d["weight"] for *_, d in G.edges(data=True))
    for u, v, d in G.edges(data=True):
        ax.plot(*zip(pos[u], pos[v]), color=color, alpha=alpha,
                lw=0.35 + 1.5 * lw_scale * d["weight"] / wmax, zorder=1,
                solid_capstyle="round")


def fig_network(G, df, cen, res):
    pos = layout(G)
    comm = nx.get_node_attributes(G, "community")
    nodes = list(G.nodes())
    mean_op = df.reindex(nodes).mean(axis=1).to_numpy()
    deg = np.array([G.degree(n) for n in nodes])

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    draw_edges(ax, G, pos)

    # Community identity is carried by position + a direct label, not by hue:
    # eight categorical hues cannot clear the all-pairs colour-blind gate.
    for cid in sorted(set(comm.values())):
        pts = np.array([pos[n] for n in nodes if comm[n] == cid])
        c = np.median(pts, axis=0)
        ax.text(c[0], c[1], f"C{cid}", fontsize=9.5, fontweight="bold", color=V.INK,
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.24", fc=V.SURFACE, ec=V.AXIS,
                          lw=0.8, alpha=0.92))

    sc = ax.scatter([pos[n][0] for n in nodes], [pos[n][1] for n in nodes],
                    s=26 + 13 * deg, c=mean_op, cmap=V.DIV, vmin=0.2, vmax=2.0,
                    edgecolors=V.SURFACE, linewidths=2.0, zorder=4)

    V.bare(ax)
    cb = fig.colorbar(sc, ax=ax, fraction=0.030, pad=0.015)
    cb.set_label("mean response  (0 = neutral, +2 = strongly agree)", fontsize=7.5,
                 color=V.INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, color=V.MUTED)

    for lab, d in [("degree 4", 4), ("12", 12), ("18", 18)]:
        ax.scatter([], [], s=26 + 13 * d, c=V.NEUTRAL, edgecolors=V.SURFACE,
                   linewidths=1.6, label=lab)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.035), ncol=3,
              title="node size = number of ties", title_fontsize=7.5,
              labelspacing=0.7, handletextpad=1.0, columnspacing=1.6)

    g = res["global"]
    ax.set_title(
        f"The opinion network: {g['nodes']} students, {g['edges']} similarity ties",
        color=V.INK, loc="left", pad=20, fontsize=11)
    ax.text(0.0, 1.012,
            f"one connected component · density {g['density']:.3f} · "
            f"{res['communities']['n_communities']} communities (Q = "
            f"{res['communities']['modularity']:.3f}) labelled at their centres",
            transform=ax.transAxes, fontsize=8, color=V.INK2, va="bottom")
    fig.tight_layout()
    return save(fig, "fig4_network.png")


# ----------------------------------------------------------------- figure 5
def fig_community_facets(G, df, res):
    pos = layout(G)
    comm = nx.get_node_attributes(G, "community")
    nodes = list(G.nodes())
    cids = sorted(set(comm.values()))
    prof = pd.DataFrame(res["community_profiles"]).set_index("community")

    fig, axes = plt.subplots(2, 4, figsize=(9.2, 5.0))
    for ax, cid in zip(axes.ravel(), cids):
        draw_edges(ax, G, pos, color="#e3e2dc", alpha=0.6, lw_scale=0.7)
        inside = [n for n in nodes if comm[n] == cid]
        sub = G.subgraph(inside)
        for u, v, d in sub.edges(data=True):
            ax.plot(*zip(pos[u], pos[v]), color=V.HIGHLIGHT, alpha=0.55, lw=1.2, zorder=2)
        out = [n for n in nodes if comm[n] != cid]
        ax.scatter([pos[n][0] for n in out], [pos[n][1] for n in out], s=11,
                   c=V.NEUTRAL, edgecolors=V.SURFACE, linewidths=0.8, zorder=3)
        ax.scatter([pos[n][0] for n in inside], [pos[n][1] for n in inside], s=38,
                   c=V.HIGHLIGHT, edgecolors=V.SURFACE, linewidths=1.4, zorder=5)
        V.bare(ax)
        r = prof.loc[cid]
        ax.set_title(f"C{cid} · n={int(r['size'])}", color=V.INK, loc="left",
                     fontsize=9, pad=12)
        ax.text(0.0, 1.005,
                f"internal density {r['internal_density']:.2f} · mean {r['mean_response']:+.2f}",
                transform=ax.transAxes, fontsize=6.8, color=V.INK2, va="bottom")
    fig.suptitle("Each community occupies its own region of opinion space",
                 x=0.008, y=0.995, ha="left", va="top", fontsize=11,
                 fontweight="bold", color=V.INK)
    fig.text(0.008, 0.945,
             "the same layout in every panel; the highlighted community and its "
             "internal ties are shown against the rest of the network in grey",
             fontsize=8, color=V.INK2, va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.915])
    return save(fig, "fig5_communities.png")

# ----------------------------------------------------------------- figure 6
def fig_structure(G, cen, res):
    g = res["global"]
    deg = cen["degree"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9))

    ax = axes[0]
    ax.hist(deg, bins=np.arange(deg.min() - 0.5, deg.max() + 1.5), color=V.CAT[0])
    ax.axvline(deg.mean(), color=V.INK, ls="--", lw=1.4)
    ax.text(deg.mean() + 0.4, ax.get_ylim()[1] * 0.9, f"mean {deg.mean():.1f}",
            fontsize=7.3, color=V.INK)
    ax.set_xlabel("degree (number of ties)")
    ax.set_ylabel("students")
    V.title(ax, "Degrees are unimodal, not scale-free",
            f"range {int(deg.min())}–{int(deg.max())} · CV {g['degree_cv']:.2f} · "
            f"skew {g['degree_skew']:+.2f}")

    ax = axes[1]
    # Clustering (~0.2) and path length (~2.6) share no scale, so both are
    # expressed as a ratio to the equivalent random graph - which is exactly
    # the quantity the small-world test is about.
    ratios = {
        "clustering\nC / C_rand": g["clustering_ratio"],
        "path length\nL / L_rand": g["avg_shortest_path"] / g["er_avg_shortest_path"],
    }
    x = np.arange(2)
    ax.axhline(1.0, color=V.AXIS, lw=1.0, zorder=1)
    ax.bar(x, list(ratios.values()), width=0.46,
           color=[V.CAT[0], V.NEUTRAL], zorder=3)
    for i, v in enumerate(ratios.values()):
        ax.text(i, v + 0.08, f"×{v:.2f}", ha="center", fontsize=8.5,
                color=V.INK, fontweight="bold")
    ax.text(-0.42, 1.07, "random graph = 1.0", fontsize=6.8, color=V.INK2, ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels(list(ratios), fontsize=7.5)
    ax.set_ylabel("ratio to random graph (same n, m)")
    ax.set_ylim(0, 3.5)
    ax.grid(axis="x", visible=False)
    V.title(ax, "Small-world: clustered but still short",
            f"σ = {g['small_world_sigma']:.2f} — far more clustered, barely further apart")

    ax = axes[2]
    ax.scatter(cen["degree"], cen["betweenness"], s=26, c=V.CAT[0],
               edgecolors=V.SURFACE, linewidths=1.2, zorder=3)
    for r in res["bridges"][:3]:
        ax.annotate(f"#{r['respondent']}", (r["degree"], r["betweenness"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7, color=V.INK2)
    ax.set_xlabel("degree")
    ax.set_ylabel("betweenness centrality")
    V.title(ax, "Hubs are also the bridges",
            f"degree vs betweenness r = "
            f"{res['centrality_correlations']['degree_vs_betweenness']:.2f}")

    fig.tight_layout()
    return save(fig, "fig6_structure.png")


# ----------------------------------------------------------------- figure 7
def fig_topics(G, res):
    pos = layout(G)
    agree = pd.DataFrame(res["topic_agreement"])
    fig = plt.figure(figsize=(9.2, 5.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1.0], hspace=0.42, wspace=0.16)

    for i, (letter, name) in enumerate(C.TOPICS.items()):
        ax = fig.add_subplot(gs[0, i])
        Gt = nx.read_graphml(C.NETWORK / f"opinion_network_{letter}.graphml")
        col = V.TOPIC_COLOR[letter]
        for u, v in Gt.edges():
            ax.plot(*zip(pos[u], pos[v]), color=col, alpha=0.22, lw=0.6, zorder=1)
        d = dict(Gt.degree())
        ax.scatter([pos[n][0] for n in Gt], [pos[n][1] for n in Gt],
                   s=[6 + 4.5 * d[n] for n in Gt], c=col,
                   edgecolors=V.SURFACE, linewidths=0.8, zorder=3)
        V.bare(ax)
        m = res["topic_metrics"][letter]
        ax.set_title(name, color=col, loc="left", fontsize=9, pad=12)
        ax.text(0.0, 1.005,
                f"{m['edges']} ties · Q={m['modularity']:.2f} · shared layout",
                transform=ax.transAxes, fontsize=6.8, color=V.INK2, va="bottom")

    ax = fig.add_subplot(gs[1, :2])
    lab = [f"{r.topic_a}–{r.topic_b}" for r in agree.itertuples()]
    ax.bar(lab, agree["jaccard"], color=V.SEQ(0.62), width=0.62)
    for i, v in enumerate(agree["jaccard"]):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=7.2, color=V.INK)
    ax.set_ylabel("Jaccard overlap of edge sets")
    ax.set_ylim(0, agree["jaccard"].max() * 1.30)
    ax.grid(axis="x", visible=False)
    V.title(ax, "The four topic networks barely share ties",
            "similarity on one topic hardly predicts similarity on another")

    ax = fig.add_subplot(gs[1, 2:])
    # Density and modularity live on different scales, so both are indexed to
    # the pooled network (= 1.0) rather than drawn against two axes.
    tm = pd.DataFrame(res["topic_metrics"]).T
    base_d, base_q = res["global"]["density"], res["communities"]["modularity"]
    x = np.arange(len(tm))
    ax.axhline(1.0, color=V.AXIS, lw=1.0, zorder=1)
    ax.bar(x - 0.19, tm["density"] / base_d, width=0.34, color=V.CAT[0],
           label="density", zorder=3)
    ax.bar(x + 0.19, tm["modularity"] / base_q, width=0.34, color=V.CAT[2],
           label="modularity Q", zorder=3)
    for i, (d, q) in enumerate(zip(tm["density"] / base_d, tm["modularity"] / base_q)):
        ax.text(i - 0.19, d + 0.03, f"{d:.2f}", ha="center", fontsize=7, color=V.INK)
        ax.text(i + 0.19, q + 0.03, f"{q:.2f}", ha="center", fontsize=7, color=V.INK)
    ax.text(len(tm) - 0.45, 1.04, "pooled network = 1.0", fontsize=6.8,
            color=V.INK2, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([C.TOPICS[i] for i in tm.index], fontsize=7)
    ax.set_ylabel("ratio to pooled network")
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(0, 1.75)
    ax.grid(axis="x", visible=False)
    V.title(ax, "Every topic is sparser yet more modular",
            "single-topic graphs split into tighter, more separated groups")

    fig.suptitle("Opinion structure is topic-specific, not global",
                 x=0.008, y=0.995, ha="left", va="top", fontsize=11,
                 fontweight="bold", color=V.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save(fig, "fig7_topics.png")

# ----------------------------------------------------------------- figure 8
def fig_discriminating(res):
    disc = pd.read_csv(C.TABLES / "discriminating_items.csv")
    top = disc.head(14).copy()
    ccols = sorted([c for c in disc.columns if c.startswith("c") and c[1:].isdigit()],
                   key=lambda x: int(x[1:]))
    M = top[ccols].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8),
                             gridspec_kw={"width_ratios": [1.75, 1.0]})

    ax = axes[0]
    im = ax.imshow(M, cmap=V.DIV, vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(ccols)))
    ax.set_xticklabels([c.upper() for c in ccols], fontsize=7.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(
        [f"{r.code}  {r.statement[:36]}{'…' if len(r.statement) > 36 else ''}"
         for r in top.itertuples()], fontsize=6.8)
    # Values are written in, so the heatmap never relies on colour alone.
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=6.0,
                    color=V.INK if abs(M[i, j]) < 1.2 else "#ffffff")
    ax.grid(False)
    ax.set_xlabel("community")
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
    cb.set_label("community mean (−2 … +2)", fontsize=7.3, color=V.INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, color=V.MUTED)
    V.title(ax, "What the communities actually disagree about",
            f"{res['n_items_significant']} of 60 items separate the groups "
            f"(Kruskal–Wallis, BH-FDR 5%)")

    ax = axes[1]
    y = np.arange(len(top))[::-1]
    ax.barh(y, top["spread"], color=[V.TOPIC_COLOR[t] for t in top["topic"]], height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(top["code"], fontsize=7)
    for yy, v in zip(y, top["spread"]):
        ax.text(v + 0.03, yy, f"{v:.2f}", va="center", fontsize=6.8, color=V.INK2)
    ax.set_xlabel("spread between community means (Likert points)")
    ax.set_xlim(0, top["spread"].max() * 1.22)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=V.TOPIC_COLOR[k], label=v)
                       for k, v in C.TOPICS.items()],
              loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2)
    V.title(ax, "Attendance and AI rules split hardest",
            "gap between the most- and least-agreeing community")

    fig.tight_layout()
    return save(fig, "fig8_discriminating.png")


# ----------------------------------------------------------------- figure 9
def fig_items(items):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    for letter, name in C.TOPICS.items():
        sub = items[items["topic"] == letter]
        ax.scatter(sub["mean"], sub["std"], s=34, c=V.TOPIC_COLOR[letter],
                   edgecolors=V.SURFACE, linewidths=1.2, label=name, zorder=3)
    extremes = pd.concat([items.nlargest(4, "std"), items.nsmallest(3, "std")])
    for r in extremes.itertuples():
        ax.annotate(r.code, (r.mean, r.std), textcoords="offset points",
                    xytext=(6, -2), fontsize=6.8, color=V.INK2)
    r = float(np.corrcoef(items["mean"], items["std"])[0, 1])
    ax.set_xlabel("mean response  (−2 … +2)")
    ax.set_ylabel("standard deviation  (disagreement)")
    ax.set_xlim(-0.12, 1.92)
    ax.legend(loc="lower left", ncol=2)
    V.title(ax, "Consensus rises as agreement rises",
            f"the most endorsed statements are the least contested (r = {r:.2f}, "
            f"n = {len(items)} items)")

    ax = axes[1]
    tp = items.groupby("topic", observed=True).agg(
        mean=("mean", "mean"), std=("std", "mean")).reindex(list(C.TOPICS))
    y = np.arange(len(tp))[::-1]
    ax.barh(y, tp["mean"], color=[V.TOPIC_COLOR[i] for i in tp.index], height=0.6)
    for yy, (m, sd) in zip(y, tp[["mean", "std"]].to_numpy()):
        ax.text(m + 0.02, yy, f"mean {m:+.2f}   sd {sd:.2f}", va="center",
                fontsize=7.2, color=V.INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([C.TOPICS[i] for i in tp.index], fontsize=8)
    ax.set_xlim(0, tp["mean"].max() * 1.75)
    ax.set_xlabel("mean agreement across the topic's 15 statements")
    ax.grid(axis="y", visible=False)
    V.title(ax, "Environment draws the widest agreement",
            "Education is the least endorsed and the most contested block")

    fig.tight_layout()
    return save(fig, "fig9_items.png")
def main() -> None:
    G, df, cen, items, build, res, clean = load()
    fig_dataset(df, clean, clean["raw_label_counts"])
    fig_similarity(build)
    fig_threshold(build)
    fig_network(G, df, cen, res)
    fig_community_facets(G, df, res)
    fig_structure(G, cen, res)
    fig_topics(G, res)
    fig_discriminating(res)
    fig_items(items)


if __name__ == "__main__":
    main()
