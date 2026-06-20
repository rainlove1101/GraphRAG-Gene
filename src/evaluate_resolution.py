import argparse
import re
from pathlib import Path

import igraph as ig
import leidenalg as la
import pandas as pd


def extract_score(text: str) -> float | None:
    if not isinstance(text, str):
        return None
    m = re.search(r"(?:置信度得分|confidence score|score)[:：]\s*([0-9.]+)", text, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def strip_prefix(x: str) -> str:
    return str(x).replace("gene_", "").replace("pathway_", "").strip()


def add_entity_keys(entities: pd.DataFrame) -> pd.DataFrame:
    e = entities.copy()
    e["node_key"] = e["id"].astype(str).map(strip_prefix)
    e["type_lower"] = e["type"].astype(str).str.lower()
    return e


def build_graph(data_dir: str, min_confidence: float):
    data_path = Path(data_dir)
    entities = pd.read_csv(data_path / "graphrag_entities.csv")
    relationships = pd.read_csv(data_path / "graphrag_relationships.csv")
    entities = add_entity_keys(entities).drop_duplicates("node_key").reset_index(drop=True)

    relationships["confidence_score"] = relationships["description"].apply(extract_score)
    relationships.loc[relationships["confidence_score"].isna(), "confidence_score"] = 1.0
    relationships = relationships[relationships["confidence_score"] >= min_confidence].copy()
    relationships["source_key"] = relationships["source"].map(strip_prefix)
    relationships["target_key"] = relationships["target"].map(strip_prefix)

    valid = set(entities["node_key"])
    relationships = relationships[relationships["source_key"].isin(valid) & relationships["target_key"].isin(valid)].copy()
    connected = set(relationships["source_key"]).union(set(relationships["target_key"]))
    entities = entities[entities["node_key"].isin(connected)].reset_index(drop=True)

    entity_id_map = {key: i for i, key in enumerate(entities["node_key"])}
    edges = [(entity_id_map[s], entity_id_map[t]) for s, t in zip(relationships["source_key"], relationships["target_key"])]

    g = ig.Graph()
    g.add_vertices(len(entity_id_map))
    g.add_edges(edges)
    g.simplify(multiple=True, loops=True)
    return g, entities, relationships


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--output_dir", default="results/tables")
    parser.add_argument("--min_confidence", type=float, default=0.9)
    parser.add_argument("--resolutions", default="0.2,0.5,0.8,1.0,1.2,1.5,2.0")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    g, entities, relationships = build_graph(args.data_dir, args.min_confidence)
    resolutions = [float(x.strip()) for x in args.resolutions.split(",") if x.strip()]

    rows = []
    for res in resolutions:
        partition = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            resolution_parameter=res,
            n_iterations=-1,
            seed=42,
        )
        sizes = pd.Series(partition.membership).value_counts()
        rows.append({
            "resolution": res,
            "nodes": g.vcount(),
            "edges": g.ecount(),
            "gene_nodes": int((entities["type_lower"] == "gene").sum()),
            "pathway_nodes": int((entities["type_lower"] == "pathway").sum()),
            "communities": int(sizes.shape[0]),
            "mean_size": round(float(sizes.mean()), 3),
            "median_size": round(float(sizes.median()), 3),
            "min_size": int(sizes.min()),
            "max_size": int(sizes.max()),
            "singletons": int((sizes == 1).sum()),
            "communities_ge_3": int((sizes >= 3).sum()),
            "communities_ge_5": int((sizes >= 5).sum()),
            "communities_ge_10": int((sizes >= 10).sum()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / "resolution_sweep.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved to {out / 'resolution_sweep.csv'}")


if __name__ == "__main__":
    main()
