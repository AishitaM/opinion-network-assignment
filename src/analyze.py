"""Step 3 - measure the network and interpret its communities.

Produces every number the Analysis and Results sections quote:
global structure, centrality rankings, Louvain communities benchmarked against
degree-preserving randomisations, and an opinion profile for each community so
the structure can be read back as substance.
"""
from __future__ import annotations

import json

import community as community_louvain          # python-louvain
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

import config as C

RNG = np.random.default_rng(C.RANDOM_SEED)


def load():
    G = nx.read_graphml(C.NETWORK / "opinion_network_main.graphml")
    df = pd.read_csv(C.PROCESSED / "responses_numeric.csv", index_col="respondent")
    df.index = df.index.astype(str)
    return G, df


# ------------------------------------------------------------ global metrics
def global_metrics(G: nx.Graph) -> dict:
    n, m = G.number_of_nodes(), G.number_of_edges()
    deg = np.array([d for _, d in G.degree()])
    giant = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    # Erdos-Renyi reference with the same n and m, averaged over 200 draws.
    er_cc, er_path = [], []
    for _ in range(200):
        R = nx.gnm_random_graph(n, m, seed=int(RNG.integers(1 << 31)))
        er_cc.append(nx.average_clustering(R))
        if nx.is_connected(R):
            er_path.append(nx.average_shortest_path_length(R))

    cc = nx.average_clustering(G)
    apl = nx.average_shortest_path_length(giant)
    return {
        "nodes": n,
        "edges": m,
        "density": nx.density(G),
        "mean_degree": float(deg.mean()),
        "median_degree": float(np.median(deg)),
        "min_degree": int(deg.min()),
        "max_degree": int(deg.max()),
        "degree_std": float(deg.std(ddof=1)),
        "degree_skew": float(stats.skew(deg)),
        "degree_cv": float(deg.std(ddof=1) / deg.mean()),
        "isolates": int((deg == 0).sum()),
        "components": nx.number_connected_components(G),
        "giant_fraction": giant.number_of_nodes() / n,
        "avg_clustering": cc,
        "transitivity": nx.transitivity(G),
        "avg_shortest_path": apl,
        "diameter": nx.diameter(giant),
        "radius": nx.radius(giant),
        "degree_assortativity": nx.degree_assortativity_coefficient(G),
        "er_avg_clustering": float(np.mean(er_cc)),
        "er_avg_shortest_path": float(np.mean(er_path)) if er_path else float("nan"),
        "clustering_ratio": cc / float(np.mean(er_cc)),
        "small_world_sigma": (cc / float(np.mean(er_cc))) / (apl / float(np.mean(er_path)))
        if er_path
        else float("nan"),
        "mean_edge_weight": float(np.mean([d["weight"] for *_, d in G.edges(data=True)])),
    }


# --------------------------------------------------------------- centrality
def centralities(G: nx.Graph, df: pd.DataFrame) -> pd.DataFrame:
    # Betweenness/closeness need a *distance*; correlation is a similarity, so
    # 1 - w is used as the edge length.
    nodes = list(G.nodes())
    measures = {
        "degree": dict(G.degree()),
        "strength": dict(G.degree(weight="weight")),
        "betweenness": nx.betweenness_centrality(G, weight="distance"),
        "closeness": nx.closeness_centrality(G, distance="distance"),
        "eigenvector": nx.eigenvector_centrality_numpy(G, weight="weight"),
        "pagerank": nx.pagerank(G, weight="weight"),
        "clustering": nx.clustering(G),
        "core_number": nx.core_number(G),
    }
    # Indexed explicitly by node name so no step relies on positional alignment.
    tbl = pd.DataFrame({k: pd.Series(v) for k, v in measures.items()}).reindex(nodes)
    tbl.index.name = "respondent"
    tbl["mean_response"] = df.reindex(tbl.index).mean(axis=1)
    tbl["response_std"] = df.reindex(tbl.index).std(axis=1, ddof=1)
    return tbl.sort_values("degree", ascending=False)


# -------------------------------------------------------------- communities
def detect_communities(G: nx.Graph) -> tuple[dict, dict]:
    part = community_louvain.best_partition(G, weight="weight", random_state=C.RANDOM_SEED)
    Q = community_louvain.modularity(part, G, weight="weight")

    # Is Q larger than a degree-preserving rewiring would give?
    # The null must match the observed graph in every respect that is not the
    # community structure itself, so:
    #   * double_edge_swap rewires edges while preserving each node's degree
    #     EXACTLY (unlike expected_degree_graph, which matches degree only in
    #     expectation and so is not a like-for-like comparison), and
    #   * the observed edge weights are randomly reassigned to the rewired
    #     edges, so the null is scored with the same weighted Louvain and the
    #     same weighted modularity as the observed graph.
    weights = [d["weight"] for *_, d in G.edges(data=True)]
    null_Q = []
    for _ in range(200):
        R = G.copy()
        nx.double_edge_swap(
            R, nswap=10 * R.number_of_edges(), max_tries=500 * R.number_of_edges(),
            seed=int(RNG.integers(1 << 31)),
        )
        shuffled = RNG.permutation(weights)
        for (u, v), w in zip(R.edges(), shuffled):
            R[u][v]["weight"] = float(w)
        p = community_louvain.best_partition(R, weight="weight", random_state=C.RANDOM_SEED)
        null_Q.append(community_louvain.modularity(p, R, weight="weight"))
    null_Q = np.array(null_Q)

    # Stability: how often do two nodes land together across 100 restarts?
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    co = np.zeros((len(nodes), len(nodes)))
    for s in range(100):
        p = community_louvain.best_partition(G, weight="weight", random_state=s)
        lab = np.array([p[u] for u in nodes])
        co += (lab[:, None] == lab[None, :]).astype(float)
    co /= 100.0
    base = np.array([part[u] for u in nodes])
    same = base[:, None] == base[None, :]
    iu = np.triu_indices(len(nodes), k=1)

    stats_ = {
        "modularity": Q,
        "n_communities": len(set(part.values())),
        "sizes": pd.Series(part).value_counts().sort_index().to_dict(),
        "null_modularity_mean": float(null_Q.mean()),
        "null_modularity_std": float(null_Q.std(ddof=1)),
        "modularity_z": float((Q - null_Q.mean()) / null_Q.std(ddof=1)),
        "modularity_p": float((null_Q >= Q).mean()),
        "co_assignment_within": float(co[iu][same[iu]].mean()),
        "co_assignment_between": float(co[iu][~same[iu]].mean()),
        "n_restarts": 100,
    }
    # Convert numpy keys for JSON.
    stats_["sizes"] = {str(int(k)): int(v) for k, v in stats_["sizes"].items()}
    return part, stats_


def community_profiles(part: dict, df: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
    """Read each community back as an opinion stance, topic by topic."""
    rows = []
    for cid in sorted(set(part.values())):
        members = [n for n, c in part.items() if c == cid]
        sub = df.reindex(members)
        row = {
            "community": cid,
            "size": len(members),
            "internal_edges": G.subgraph(members).number_of_edges(),
            "internal_density": nx.density(G.subgraph(members)),
            "mean_response": sub.to_numpy().mean(),
        }
        for letter, name in C.TOPICS.items():
            cols = [c for c in df.columns if c.startswith(letter)]
            row[f"mean_{letter}"] = sub[cols].to_numpy().mean()
            row[f"std_{letter}"] = sub[cols].to_numpy().std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def discriminating_items(part: dict, df: pd.DataFrame, top_k: int = 12) -> pd.DataFrame:
    """Items whose answers differ most between communities (Kruskal-Wallis)."""
    groups = {}
    for n, c in part.items():
        groups.setdefault(c, []).append(n)
    items = pd.read_csv(C.PROCESSED / "items.csv")
    text = dict(zip(items["code"], items["statement"]))

    rows = []
    for col in df.columns:
        samples = [df.loc[m, col].to_numpy() for m in groups.values() if len(m) > 1]
        try:
            H, p = stats.kruskal(*samples)
        except ValueError:            # all values identical
            H, p = 0.0, 1.0
        means = {f"c{c}": float(df.loc[m, col].mean()) for c, m in sorted(groups.items())}
        rows.append(
            {
                "code": col,
                "topic": col[0],
                "statement": text.get(col, ""),
                "H": H,
                "p": p,
                "spread": max(means.values()) - min(means.values()),
                **means,
            }
        )
    out = pd.DataFrame(rows).sort_values("H", ascending=False)
    # Benjamini-Hochberg across the 60 items.
    m = len(out)
    out["p_adj"] = np.minimum.accumulate(
        (out.sort_values("p")["p"].to_numpy() * m / np.arange(1, m + 1))[::-1]
    )[::-1][np.argsort(np.argsort(out["p"].to_numpy()))]
    out["p_adj"] = out["p_adj"].clip(upper=1.0)
    return out


def topic_agreement() -> pd.DataFrame:
    """Do the four topic-specific networks agree with each other?"""
    graphs = {
        k: nx.read_graphml(C.NETWORK / f"opinion_network_{k}.graphml") for k in C.TOPICS
    }
    keys = list(C.TOPICS)
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            Ea = {frozenset(e) for e in graphs[a].edges()}
            Eb = {frozenset(e) for e in graphs[b].edges()}
            inter, union = len(Ea & Eb), len(Ea | Eb)
            rows.append(
                {
                    "topic_a": a,
                    "topic_b": b,
                    "edges_a": len(Ea),
                    "edges_b": len(Eb),
                    "shared": inter,
                    "jaccard": inter / union if union else 0.0,
                }
            )
    return pd.DataFrame(rows)


def main() -> dict:
    G, df = load()

    gm = global_metrics(G)
    cen = centralities(G, df)
    part, cstats = detect_communities(G)
    prof = community_profiles(part, df, G)
    disc = discriminating_items(part, df)
    tagree = topic_agreement()

    nx.set_node_attributes(G, part, "community")
    nx.set_node_attributes(G, {n: float(v) for n, v in cen["betweenness"].items()}, "betweenness")
    nx.set_node_attributes(G, {n: int(v) for n, v in cen["degree"].items()}, "degree_")
    nx.write_graphml(G, C.NETWORK / "opinion_network_annotated.graphml")

    cen.to_csv(C.TABLES / "centralities.csv")
    prof.to_csv(C.TABLES / "community_profiles.csv", index=False)
    disc.to_csv(C.TABLES / "discriminating_items.csv", index=False)
    tagree.to_csv(C.TABLES / "topic_network_agreement.csv", index=False)
    pd.Series(part, name="community").to_csv(C.TABLES / "communities.csv")

    # per-topic global metrics for the comparison table
    topic_metrics = {}
    for k in C.TOPICS:
        Gt = nx.read_graphml(C.NETWORK / f"opinion_network_{k}.graphml")
        pt = community_louvain.best_partition(Gt, weight="weight", random_state=C.RANDOM_SEED)
        big = Gt.subgraph(max(nx.connected_components(Gt), key=len))
        topic_metrics[k] = {
            "edges": Gt.number_of_edges(),
            "density": nx.density(Gt),
            "avg_clustering": nx.average_clustering(Gt),
            "components": nx.number_connected_components(Gt),
            "giant_fraction": big.number_of_nodes() / Gt.number_of_nodes(),
            "modularity": community_louvain.modularity(pt, Gt, weight="weight"),
            "n_communities": len(set(pt.values())),
            "isolates": sum(1 for n in Gt if Gt.degree(n) == 0),
        }

    results = {
        "global": gm,
        "communities": cstats,
        "community_profiles": prof.to_dict("records"),
        "top_discriminating": disc.head(10).to_dict("records"),
        "n_items_significant": int((disc["p_adj"] < 0.05).sum()),
        "topic_metrics": topic_metrics,
        "topic_agreement": tagree.to_dict("records"),
        "hubs": cen.head(5).reset_index()[["respondent", "degree", "betweenness", "eigenvector"]]
        .to_dict("records"),
        "bridges": cen.nlargest(5, "betweenness").reset_index()[
            ["respondent", "degree", "betweenness"]
        ].to_dict("records"),
        "peripheral": cen.nsmallest(5, "degree").reset_index()[
            ["respondent", "degree", "mean_response"]
        ].to_dict("records"),
        "centrality_correlations": {
            "degree_vs_mean_response": float(cen["degree"].corr(cen["mean_response"])),
            "degree_vs_response_std": float(cen["degree"].corr(cen["response_std"])),
            "degree_vs_betweenness": float(cen["degree"].corr(cen["betweenness"])),
        },
    }
    (C.OUT / "analysis_results.json").write_text(json.dumps(results, indent=2, default=float))

    print(f"[analyze] density {gm['density']:.3f} | mean degree {gm['mean_degree']:.1f} "
          f"| clustering {gm['avg_clustering']:.3f} vs ER {gm['er_avg_clustering']:.3f} "
          f"(x{gm['clustering_ratio']:.2f})")
    print(f"[analyze] path {gm['avg_shortest_path']:.2f} (ER {gm['er_avg_shortest_path']:.2f}) "
          f"| diameter {gm['diameter']} | sigma {gm['small_world_sigma']:.2f}")
    print(f"[analyze] degree assortativity {gm['degree_assortativity']:+.3f}")
    print(f"[analyze] Louvain: {cstats['n_communities']} communities {cstats['sizes']}, "
          f"Q={cstats['modularity']:.3f} vs null {cstats['null_modularity_mean']:.3f} "
          f"(z={cstats['modularity_z']:.1f}, p={cstats['modularity_p']:.3f})")
    print(f"[analyze] partition stability: within {cstats['co_assignment_within']:.2f} / "
          f"between {cstats['co_assignment_between']:.2f}")
    print(f"[analyze] items separating communities (BH<0.05): {results['n_items_significant']}/60")
    print("[analyze] top discriminating items:")
    for r in results["top_discriminating"][:5]:
        print(f"[analyze]   {r['code']} H={r['H']:.1f} p_adj={r['p_adj']:.4f} "
              f"spread={r['spread']:.2f}  {r['statement'][:62]}")
    return results


if __name__ == "__main__":
    main()
