import argparse
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd

from config import Settings, GENE_INTERPRET_PROMPT

GENE_PATTERN = re.compile(r"^[A-Z0-9][A-Za-z0-9-]*$")
STOP_WORDS = {"AND", "OR", "NOT", "THE", "WITH", "FOR", "IN", "ON", "TO", "OF", "IS", "IT", "GENE", "GENES"}
BIO_PRIORITY = re.compile(
    r"DNA|repair|homologous|recombination|HDR|BRCA|PALB|CHEK|damage|checkpoint|mitochond|respiratory|electron transport|oxidative|phosphorylation|TCA|complex I|complex II|complex IV|disease|cancer|tumou?r|drug|metabolism",
    flags=re.I,
)
LOW_PRIORITY = re.compile(r"generic transcription|gene expression|cellular response to stimuli|cellular response to stress|disease$|metabolism$", flags=re.I)


def extract_genes(question: str) -> list[str]:
    words = re.split(r"[^\w-]+", question)
    genes = []
    for w in words:
        w_clean = w.strip().upper()
        if 2 <= len(w_clean) <= 15 and GENE_PATTERN.match(w_clean) and w_clean not in STOP_WORDS:
            genes.append(w_clean)
    return sorted(set(genes))


def strip_prefix(x: str) -> str:
    return str(x).replace("gene_", "").replace("pathway_", "").strip()


def short_text(x, n=360):
    if not isinstance(x, str):
        return ""
    x = re.sub(r"\s+", " ", x).strip()
    return x[:n] + ("..." if len(x) > n else "")


def ensure_keys(entities: pd.DataFrame, relationships: pd.DataFrame):
    e = entities.copy()
    if "node_key" not in e.columns:
        e["node_key"] = e["id"].astype(str).map(strip_prefix)
    e["type_lower"] = e["type"].astype(str).str.lower()
    e["name_upper"] = e["name"].astype(str).str.upper()

    r = relationships.copy()
    if "source_key" not in r.columns:
        r["source_key"] = r["source"].map(strip_prefix)
    if "target_key" not in r.columns:
        r["target_key"] = r["target"].map(strip_prefix)
    return e, r


def pathway_score(name: str, desc: str, hit_genes: set[str], total_genes: int):
    text = f"{name} {desc}"
    shared = len(hit_genes)
    priority = 1 if BIO_PRIORITY.search(text) else 0
    low = 1 if LOW_PRIORITY.search(name) else 0
    # Prefer pathways involving more query genes; then biologically specific keywords; penalize generic broad pathways.
    return (shared, priority, -low, -len(name))


def build_gene_context(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    communities: pd.DataFrame,
    genes: list[str],
    max_pathways: int = 12,
    max_direct: int = 10,
    include_community: bool = False,
) -> str:
    e, rel = ensure_keys(entities, relationships)

    key_to_name = dict(zip(e["node_key"].astype(str), e["name"].astype(str)))
    key_to_type = dict(zip(e["node_key"].astype(str), e["type_lower"].astype(str)))
    key_to_desc = dict(zip(e["node_key"].astype(str), e["description"].astype(str)))

    matched = e[(e["type_lower"] == "gene") & (e["name_upper"].isin(genes))].drop_duplicates("name_upper")
    found_genes = matched["name_upper"].tolist()
    missing = sorted(set(genes) - set(found_genes))
    gene_keys = set(matched["node_key"].astype(str))

    sections = []

    gene_lines = []
    for row in matched.itertuples(index=False):
        gene_lines.append(f"- {row.name}: community={getattr(row, 'community', 'NA')}; {short_text(getattr(row, 'description', ''), 520)}")
    if gene_lines:
        sections.append("[1. Query genes]" + "\n" + "\n".join(gene_lines))
    if missing:
        sections.append("[Missing genes]\n" + ", ".join(missing))

    # Direct relationships between query genes.
    direct = rel[rel["source_key"].isin(gene_keys) & rel["target_key"].isin(gene_keys)].copy()
    reverse_direct = rel[rel["target_key"].isin(gene_keys) & rel["source_key"].isin(gene_keys)].copy()
    direct = pd.concat([direct, reverse_direct]).drop_duplicates()
    if not direct.empty:
        lines = []
        for row in direct.head(max_direct).itertuples(index=False):
            s = key_to_name.get(str(row.source_key), str(row.source_key))
            t = key_to_name.get(str(row.target_key), str(row.target_key))
            lines.append(f"- {s} -> {t}: {short_text(getattr(row, 'description', ''), 220)}")
        sections.append("[2. Direct relationships among query genes]\n" + "\n".join(lines))

    # Pathways directly linked to query genes.
    pathway_hits = defaultdict(set)
    pathway_evidence = defaultdict(list)
    around = rel[rel["source_key"].isin(gene_keys) | rel["target_key"].isin(gene_keys)]
    for row in around.itertuples(index=False):
        src, tgt = str(row.source_key), str(row.target_key)
        if src in gene_keys:
            gene = key_to_name.get(src, src).upper()
            other = tgt
        elif tgt in gene_keys:
            gene = key_to_name.get(tgt, tgt).upper()
            other = src
        else:
            continue
        if key_to_type.get(other, "") == "pathway":
            pathway_hits[other].add(gene)
            pathway_evidence[other].append(short_text(getattr(row, 'description', ''), 120))

    if pathway_hits:
        ranked = sorted(
            pathway_hits.items(),
            key=lambda item: pathway_score(key_to_name.get(item[0], item[0]), key_to_desc.get(item[0], ""), item[1], len(genes)),
            reverse=True,
        )[:max_pathways]
        lines = []
        for key, hit_genes in ranked:
            name = key_to_name.get(key, key)
            desc = short_text(key_to_desc.get(key, ""), 280)
            score = len(hit_genes)
            lines.append(f"- {name} ({key}): matched {score}/{len(genes)} genes [{', '.join(sorted(hit_genes))}]. {desc}")
        sections.append("[3. Top shared or nearby pathways]\n" + "\n".join(lines))

    if include_community:
        comm_ids = matched["community"].dropna().unique().tolist()
        comm_lines = []
        for cid in comm_ids:
            comm = communities[communities["community"] == cid]
            if comm.empty:
                continue
            row = comm.iloc[0]
            comm_lines.append(
                f"- Community {cid}: size={row.get('size', 'NA')}; genes={row.get('gene_count', 'NA')}; pathways={row.get('pathway_count', 'NA')}; "
                f"examples={row.get('example_nodes', '')}; summary={short_text(row.get('summary', ''), 420)}"
            )
        if comm_lines:
            sections.append("[4. Community-level context]\n" + "\n".join(comm_lines))

    if not sections:
        return "No matched evidence was found in the current knowledge graph."
    return "\n\n".join(sections)


def offline_report(genes: list[str], context: str) -> str:
    return (
        "Offline evidence report based on retrieved graph context.\n\n"
        f"Detected genes: {', '.join(genes)}\n\n"
        "Note: offline mode does not make clinical conclusions. It only organizes evidence retrieved from the knowledge graph.\n\n"
        + context[:7000]
        + "\n\nLimitation: LLM generation was disabled or ONEAPI_API_KEY was not configured."
    )


def generate_answer(context: str, question: str, genes: list[str], offline: bool = False) -> str:
    settings = Settings()
    if offline or not settings.oneapi_api_key:
        return offline_report(genes, context)
    import requests

    prompt = GENE_INTERPRET_PROMPT.format(context=context, question=question)
    headers = {"Authorization": f"Bearer {settings.oneapi_api_key}", "Content-Type": "application/json"}
    response = requests.post(
        f"{settings.oneapi_base_url}/chat/completions",
        headers=headers,
        json={"model": settings.llm_model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def query(index_dir: str, question: str, offline: bool = False, max_pathways: int = 12, include_community: bool = False):
    index_path = Path(index_dir)
    entities = pd.read_parquet(index_path / "create_final_nodes.parquet")
    relationships = pd.read_parquet(index_path / "create_final_relationships.parquet")
    communities = pd.read_parquet(index_path / "create_final_communities.parquet")

    genes = extract_genes(question)
    if not genes:
        raise ValueError("No gene symbols were detected from the question.")

    context = build_gene_context(
        entities, relationships, communities, genes,
        max_pathways=max_pathways,
        include_community=include_community,
    )
    answer = generate_answer(context, question, genes, offline=offline)
    return genes, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="results/index")
    parser.add_argument("--question", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max_pathways", type=int, default=12)
    parser.add_argument("--include_community", action="store_true", help="Print community-level summaries. Disabled by default to keep output concise.")
    args = parser.parse_args()
    genes, answer = query(args.index_dir, args.question, args.offline, args.max_pathways, args.include_community)
    print("Detected genes:", ", ".join(genes))
    print("\n" + answer)


if __name__ == "__main__":
    main()
