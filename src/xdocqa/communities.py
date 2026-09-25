"""Step 6 - Communities: find groups of chunks, from different books, that talk about the same thing.

Runs the Leiden algorithm (Neo4j Graph Data Science) on the SIMILAR_TO links of step 5 and writes:
  - `community` on every Chunk;
  - (:Community {id, size, authors, n_authors, cross_document})<-[:IN_COMMUNITY]-(:Chunk);
  - data/communities/communities.json   members of every community, to read and to generate questions.

Why Leiden and not Louvain: both group nodes that are densely linked to each other, but Louvain can
return a community whose members are not connected to each other (a node that held the group
together moves away, the rest keeps the same label). Leiden adds a refinement step that splits
such groups, so every community is guaranteed to be connected. For us this matters: the chunks of
one community are given to the LLM together, and they must really be about the same topic.

A community is useful for cross-document questions only if it contains at least 2 authors
(`cross_document = true`).

Usage:
    python -m xdocqa.communities
    python -m xdocqa.communities --resolution 4       # more, smaller communities
    python -m xdocqa.communities --sweep              # try several resolutions, write nothing
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GRAPH_NAME = "chunks"
RESOLUTION = 3.0     # "gamma": higher = more and smaller communities (chosen with --sweep, see README)
SWEEP = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
SEED = 42            # same seed, same communities: the run is reproducible


def drop(session) -> None:
    session.run("CALL gds.graph.drop($name, false) YIELD graphName RETURN graphName", name=GRAPH_NAME)


def project(session) -> None:
    """Copy the chunks and their SIMILAR_TO links into GDS's in-memory graph."""
    drop(session)                                       # remove a leftover copy, if any
    session.run("""
        CALL gds.graph.project($name, 'Chunk',
             {SIMILAR_TO: {orientation: 'UNDIRECTED', properties: 'score'}})
        YIELD graphName RETURN graphName
    """, name=GRAPH_NAME)


def sweep(session, resolutions: list[float], seed: int) -> list[dict]:
    """Run Leiden at several resolutions WITHOUT writing anything, and describe the result of each."""
    project(session)
    table = []
    for gamma in resolutions:
        rows = session.run("""
            CALL gds.leiden.stream($name, {relationshipWeightProperty: 'score', gamma: $gamma,
                                           randomSeed: $seed, concurrency: 1})
            YIELD nodeId, communityId
            RETURN communityId AS community, gds.util.asNode(nodeId).author AS author
        """, name=GRAPH_NAME, gamma=gamma, seed=seed).data()
        members: dict[int, list[str]] = {}
        for r in rows:
            members.setdefault(r["community"], []).append(r["author"])
        cross = [m for m in members.values() if len(set(m)) >= 2]
        sizes = sorted(len(m) for m in cross) or [0]
        table.append({"resolution": gamma, "cross_document": len(cross),
                      "chunks_in_cross": sum(sizes), "median_size": statistics.median(sizes),
                      "max_size": sizes[-1],
                      "four_authors": sum(1 for m in cross if len(set(m)) == 4)})
    drop(session)
    return table


def run_leiden(session, resolution: float, seed: int) -> dict:
    # 1. copy the chunks and their SIMILAR_TO links into GDS's in-memory graph
    project(session)
    # 2. Leiden, using the similarity score as the strength of each link
    stats = session.run("""
        CALL gds.leiden.write($name, {
            writeProperty: 'community',
            relationshipWeightProperty: 'score',
            gamma: $gamma,
            randomSeed: $seed,
            concurrency: 1
        })
        YIELD communityCount, modularity, ranLevels
        RETURN communityCount, modularity, ranLevels
    """, name=GRAPH_NAME, gamma=resolution, seed=seed).single().data()
    drop(session)
    return stats


def read_communities(session) -> list[dict]:
    rows = session.run("""
        MATCH (c:Chunk)
        RETURN c.community AS community, c.id AS id, c.author AS author,
               c.section AS section, c.start_page AS start_page
        ORDER BY community, id
    """).data()
    groups: dict[int, dict] = {}
    for r in rows:
        g = groups.setdefault(r["community"], {"id": r["community"], "members": []})
        g["members"].append({k: r[k] for k in ("id", "author", "section", "start_page")})
    out = []
    for g in groups.values():
        authors = sorted({m["author"] for m in g["members"]})
        out.append({**g, "size": len(g["members"]), "authors": authors, "n_authors": len(authors),
                    "cross_document": len(authors) >= 2})
    return sorted(out, key=lambda g: (-g["n_authors"], -g["size"]))


def write_community_nodes(session, groups: list[dict]) -> None:
    session.run("MATCH (k:Community) DETACH DELETE k")
    session.run("""
        UNWIND $rows AS r
        CREATE (k:Community {id: r.id, size: r.size, authors: r.authors,
                             n_authors: r.n_authors, cross_document: r.cross_document})
        WITH k, r
        UNWIND r.ids AS cid
        MATCH (c:Chunk {id: cid})
        MERGE (c)-[:IN_COMMUNITY]->(k)
    """, rows=[{**{k: g[k] for k in ("id", "size", "authors", "n_authors", "cross_document")},
               "ids": [m["id"] for m in g["members"]]} for g in groups])


def summarize(groups: list[dict]) -> dict:
    cross = [g for g in groups if g["cross_document"]]
    sizes = [g["size"] for g in cross] or [0]
    return {
        "communities": len(groups),
        "cross_document": len(cross),
        "single_author": sum(1 for g in groups if g["n_authors"] == 1 and g["size"] > 1),
        "isolated_chunks": sum(1 for g in groups if g["size"] == 1),
        "chunks_in_cross_document": sum(g["size"] for g in cross),
        "cross_size": {"min": min(sizes), "median": statistics.median(sizes), "max": max(sizes)},
    }


AUTHOR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # one fixed color per author, in corpus order


def plot(groups: list[dict], resolution: float, out: Path) -> None:
    """One bar per cross-document community; each bar split by author."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cross = sorted([g for g in groups if g["cross_document"]], key=lambda g: g["size"])
    authors = sorted({m["author"] for g in cross for m in g["members"]})
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(cross) + 1.4))
    left = [0] * len(cross)
    for author, color in zip(authors, AUTHOR_COLORS):
        n = [sum(1 for m in g["members"] if m["author"] == author) for g in cross]
        ax.barh(range(len(cross)), n, left=left, color=color, label=author,
                height=0.7, edgecolor="white", linewidth=1.5)
        left = [a + b for a, b in zip(left, n)]
    for y, g in enumerate(cross):
        ax.text(g["size"] + 0.3, y, str(g["size"]), va="center", fontsize=8, color="#555555")
    ax.set_yticks(range(len(cross)), [f"#{g['id']}" for g in cross], fontsize=8)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("chunks in the community")
    ax.set_title(f"Cross-document communities (Leiden, resolution {resolution:g}): who talks in each topic",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, loc="lower right", ncol=len(authors))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolution", type=float, default=RESOLUTION)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="data/communities")
    ap.add_argument("--sweep", action="store_true", help="try several resolutions and print a table, write nothing")
    args = ap.parse_args()

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"],
                                  auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
                                  notifications_min_severity="OFF")   # no deprecation warnings in the output
    with driver.session() as session:
        links = session.run("MATCH ()-[l:SIMILAR_TO]->() RETURN count(l) AS n").single()["n"]
        if not links:
            raise SystemExit("No SIMILAR_TO links: run xdocqa.similarity first")
        if args.sweep:
            table = sweep(session, SWEEP, args.seed)
        else:
            stats = run_leiden(session, args.resolution, args.seed)
            groups = read_communities(session)
            write_community_nodes(session, groups)
    driver.close()                                       # after the session is closed: no warning

    if args.sweep:
        print("resolution | cross-document communities | chunks in them | median size | max size | with 4 authors")
        for t in table:
            print(f"{t['resolution']:>10g} | {t['cross_document']:>26} | {t['chunks_in_cross']:>14} | "
                  f"{t['median_size']:>11} | {t['max_size']:>8} | {t['four_authors']:>14}")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "sweep.json").write_text(json.dumps(table, indent=2))
        return

    s = summarize(groups)
    print(f"Leiden: {stats['communityCount']} communities, modularity {stats['modularity']:.3f}, "
          f"levels {stats['ranLevels']} (resolution {args.resolution:g})")
    print(f"cross-document communities (2+ authors): {s['cross_document']} "
          f"| chunks in them: {s['chunks_in_cross_document']} "
          f"| size min/median/max {s['cross_size']['min']}/{s['cross_size']['median']}/{s['cross_size']['max']}")
    print(f"single-author communities: {s['single_author']} | isolated chunks: {s['isolated_chunks']}")
    print("\nlargest cross-document communities:")
    for g in [g for g in groups if g["cross_document"]][:10]:
        per_author = {}
        for m in g["members"]:                           # one example section per author
            per_author.setdefault(m["author"], f"{m['section'][:45]} (p. {m['start_page']})")
        counts = {a: sum(1 for m in g["members"] if m["author"] == a) for a in g["authors"]}
        print(f"  #{g['id']:<4} {g['size']:3d} chunks | " + ", ".join(f"{a} {n}" for a, n in counts.items()))
        for a, sec in per_author.items():
            print(f"         - {a}: {sec}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "communities.json").write_text(json.dumps(
        {"resolution": args.resolution, "seed": args.seed, "leiden": stats, "summary": s, "communities": groups},
        indent=2, ensure_ascii=False))
    print(f"\ncommunities -> {out / 'communities.json'}")
    plot(groups, args.resolution, Path("docs/images/communities.png"))
    print("plot -> docs/images/communities.png")


if __name__ == "__main__":
    main()