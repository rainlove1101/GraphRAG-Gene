import argparse
import asyncio
import re
from pathlib import Path

import igraph as ig
import leidenalg as la
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from config import Settings, COMMUNITY_SUMMARY_PROMPT


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
    """Use stable graph keys. Gene key = gene symbol; pathway key = Reactome pathway ID."""
    e = entities.copy()
    e["node_key"] = e["id"].astype(str).map(strip_prefix)
    # For malformed rows without useful IDs, fall back to name.
    missing = e["node_key"].eq("") | e["node_key"].isna()
    e.loc[missing, "node_key"] = e.loc[missing, "name"].astype(str)
    e["type_lower"] = e["type"].astype(str).str.lower()
    return e


async def api_request(session, url, headers, payload, max_retries=3):
    for retry in range(max_retries):
        try:
            async with session.post(url, headers=headers, json=payload, timeout=120) as resp:
                if resp.status == 200:
                    return await resp.json()
                if retry == max_retries - 1:
                    text = await resp.text()
                    print(f"API failed: HTTP {resp.status}: {text[:300]}")
                    return None
        except Exception as e:
            if retry == max_retries - 1:
                print(f"API exception: {e}")
                return None
        await asyncio.sleep(2 ** retry)
    return None


async def summarize_module(session, settings: Settings, node_texts: list[str], skip_llm: bool) -> str:
    joined = "\n".join(node_texts[:80])
    if skip_llm or not settings.oneapi_api_key:
        return "Extractive module summary: " + " ".join(node_texts[:8])[:1800]
    if session is None:
        return "Summary generation skipped because no API session was available. " + joined[:1200]
    headers = {"Authorization": f"Bearer {settings.oneapi_api_key}", "Content-Type": "application/json"}
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": COMMUNITY_SUMMARY_PROMPT.format(node_texts=joined)}],
        "temperature": 0.1,
    }
    result = await api_request(session, f"{settings.oneapi_base_url}/chat/completions", headers, payload)
    if result and "choices" in result:
        return result["choices"][0]["message"]["content"]
    return "Summary generation failed. " + joined[:1200]


async def build_index(data_dir: str, output_dir: str, skip_llm: bool, resolution: float | None = None, min_community_size: int | None = None):
    settings = Settings()
    if resolution is None:
        resolution = settings.leiden_resolution
    if min_community_size is None:
        min_community_size = settings.min_community_size

    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    entities = pd.read_csv(data_path / "graphrag_entities.csv")
    relationships = pd.read_csv(data_path / "graphrag_relationships.csv")

    entities = add_entity_keys(entities)
    entities = entities.drop_duplicates("node_key").reset_index(drop=True)

    relationships["confidence_score"] = relationships["description"].apply(extract_score)
    relationships.loc[relationships["confidence_score"].isna(), "confidence_score"] = 1.0
    relationships = relationships[relationships["confidence_score"] >= settings.min_confidence_score].copy()
    relationships["source_key"] = relationships["source"].map(strip_prefix)
    relationships["target_key"] = relationships["target"].map(strip_prefix)

    valid_keys = set(entities["node_key"])
    relationships = relationships[
        relationships["source_key"].isin(valid_keys) & relationships["target_key"].isin(valid_keys)
    ].copy()

    connected = set(relationships["source_key"]).union(set(relationships["target_key"]))
    entities = entities[entities["node_key"].isin(connected)].reset_index(drop=True)

    entity_id_map = {key: i for i, key in enumerate(entities["node_key"])}
    edges = [(entity_id_map[s], entity_id_map[t]) for s, t in zip(relationships["source_key"], relationships["target_key"])]

    g = ig.Graph()
    g.add_vertices(len(entity_id_map))
    g.add_edges(edges)
    g.simplify(multiple=True, loops=True)

    partition = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        n_iterations=-1,
        seed=42,
    )
    entities["community"] = partition.membership

    community_inputs = []
    for comm_id, group in entities.groupby("community"):
        if len(group) < min_community_size:
            continue
        # Prefer pathway descriptions for module-level biological meaning, then genes.
        group_sorted = pd.concat([
            group[group["type_lower"].eq("pathway")],
            group[group["type_lower"].eq("gene")],
            group[~group["type_lower"].isin(["pathway", "gene"])],
        ])
        community_inputs.append((comm_id, group_sorted["description"].dropna().astype(str).tolist()))

    if skip_llm or not settings.oneapi_api_key:
        summaries = [
            await summarize_module(None, settings, node_texts, skip_llm=True)
            for _, node_texts in community_inputs
        ]
    else:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            tasks = [
                summarize_module(session, settings, node_texts, skip_llm=False)
                for _, node_texts in community_inputs
            ]
            summaries = await tqdm_asyncio.gather(*tasks, desc="Summarizing communities") if tasks else []

    community_rows = []
    for (comm_id, _), summary in zip(community_inputs, summaries):
        group = entities[entities["community"] == comm_id]
        example = "; ".join(group.sort_values(["type_lower", "name"])["name"].head(15).astype(str).tolist())
        community_rows.append({
            "community": comm_id,
            "summary": summary,
            "size": len(group),
            "gene_count": int((group["type_lower"] == "gene").sum()),
            "pathway_count": int((group["type_lower"] == "pathway").sum()),
            "example_nodes": example,
        })

    comm_df = pd.DataFrame(community_rows).sort_values("size", ascending=False)
    sizes = entities.groupby("community").size()
    kg_stats = pd.DataFrame([
        {"metric": "nodes_after_filtering", "value": len(entities)},
        {"metric": "gene_nodes_after_filtering", "value": int((entities["type_lower"] == "gene").sum())},
        {"metric": "pathway_nodes_after_filtering", "value": int((entities["type_lower"] == "pathway").sum())},
        {"metric": "relationships_after_confidence_filtering", "value": len(relationships)},
        {"metric": "graph_edges_after_simplification", "value": len(g.es)},
        {"metric": "leiden_resolution", "value": resolution},
        {"metric": "min_community_size_for_summary", "value": min_community_size},
        {"metric": "communities_detected", "value": entities["community"].nunique()},
        {"metric": "communities_summarized", "value": len(comm_df)},
        {"metric": "mean_community_size", "value": round(float(sizes.mean()), 3)},
        {"metric": "median_community_size", "value": round(float(sizes.median()), 3)},
        {"metric": "max_community_size", "value": int(sizes.max())},
        {"metric": "singleton_communities", "value": int((sizes == 1).sum())},
    ])

    entities.to_parquet(out_path / "create_final_nodes.parquet", index=False)
    relationships.to_parquet(out_path / "create_final_relationships.parquet", index=False)
    comm_df.to_parquet(out_path / "create_final_communities.parquet", index=False)
    kg_stats.to_csv(out_path / "kg_statistics.csv", index=False)
    comm_df[["community", "size", "gene_count", "pathway_count", "example_nodes"]].to_csv(out_path / "community_statistics.csv", index=False)

    print("Index built successfully.")
    print(kg_stats.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--output_dir", default="results/index")
    parser.add_argument("--skip_llm", action="store_true", help="Use extractive summaries instead of LLM calls.")
    parser.add_argument("--resolution", type=float, default=None, help="Leiden RBConfiguration resolution parameter.")
    parser.add_argument("--min_community_size", type=int, default=None, help="Minimum community size retained for summaries.")
    args = parser.parse_args()
    asyncio.run(build_index(args.data_dir, args.output_dir, args.skip_llm, args.resolution, args.min_community_size))


if __name__ == "__main__":
    main()
