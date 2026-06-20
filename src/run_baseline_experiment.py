
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from evaluation_utils import keyword_hit, strict_keyword_hit


BROAD_PATHWAY_KEYWORDS = [
    "metabolism",
    "gene expression",
    "transcription",
    "cellular responses",
    "cellular response",
    "disease",
    "diseases",
    "developmental biology",
    "signal transduction",
]


def normalize_gene(gene: str) -> str:
    return str(gene).strip().upper()


def split_genes(value: str):
    return [normalize_gene(x) for x in re.split(r"[;,|\s]+", str(value)) if x.strip()]


def split_keywords(value: str):
    return [x.strip().lower() for x in re.split(r"[;|]+", str(value)) if x.strip()]


def find_col(df, candidates, default=None):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return default


def infer_node_columns(nodes: pd.DataFrame):
    name_col = find_col(nodes, ["name", "title", "label", "node_name"], nodes.columns[0])
    desc_col = find_col(nodes, ["description", "text", "summary", "details"], None)
    type_col = find_col(nodes, ["type", "entity_type", "node_type", "kind"], None)
    id_col = find_col(nodes, ["id", "entity_id", "node_id", "reactome_id"], None)
    community_col = find_col(nodes, ["community", "community_id"], None)
    return name_col, desc_col, type_col, id_col, community_col


def infer_relationship_columns(rels: pd.DataFrame):
    source_col = find_col(rels, ["source", "src", "from"], rels.columns[0])
    target_col = find_col(rels, ["target", "dst", "to"], rels.columns[1] if len(rels.columns) > 1 else rels.columns[0])
    desc_col = find_col(rels, ["description", "text", "relation", "relationship", "type"], None)
    confidence_col = find_col(rels, ["confidence", "score", "weight", "confidence_score"], None)
    return source_col, target_col, desc_col, confidence_col


def clean_gene_from_endpoint(value: str):
    s = str(value)
    s = re.sub(r"^gene_", "", s, flags=re.I)
    if s.startswith("pathway_"):
        return None
    if re.match(r"^[A-Za-z0-9\-_.]+$", s):
        return s.upper()
    return s.upper()


def clean_pathway_from_endpoint(value: str):
    s = str(value)
    if s.lower().startswith("pathway_"):
        return s.split("_", 1)[1]
    if "R-HSA-" in s:
        m = re.search(r"R-HSA-\d+", s)
        return m.group(0) if m else s
    return None


def is_gene_row(row, name_col, type_col, id_col):
    name = str(row.get(name_col, ""))
    typ = str(row.get(type_col, "")).lower() if type_col else ""
    node_id = str(row.get(id_col, "")) if id_col else ""
    if "gene" in typ:
        return True
    if "pathway" in typ:
        return False
    if "R-HSA-" in name or "R-HSA-" in node_id:
        return False
    return bool(re.match(r"^[A-Z0-9][A-Z0-9\-]{1,15}$", name.upper()))


def is_pathway_row(row, name_col, type_col, id_col):
    name = str(row.get(name_col, ""))
    typ = str(row.get(type_col, "")).lower() if type_col else ""
    node_id = str(row.get(id_col, "")) if id_col else ""
    if "pathway" in typ:
        return True
    if "R-HSA-" in name or "R-HSA-" in node_id:
        return True
    return False


def extract_reactome_id(text):
    if text is None:
        return None
    m = re.search(r"R-HSA-\d+", str(text))
    return m.group(0) if m else None


def pathway_specificity_score(pathway_name, description=""):
    text = f"{pathway_name} {description}".lower()
    penalty = 0.0
    for kw in BROAD_PATHWAY_KEYWORDS:
        if kw in text:
            penalty += 0.15
    gene_count_match = re.search(r"contains.*?(\d+).*?genes|包含.*?(\d+).*?基因", text)
    size_penalty = 0.0
    if gene_count_match:
        n = next((int(x) for x in gene_count_match.groups() if x), 0)
        if n > 1000:
            size_penalty = 0.35
        elif n > 300:
            size_penalty = 0.20
        elif n > 100:
            size_penalty = 0.10
    return max(0.0, 1.0 - penalty - size_penalty)


def load_index(index_dir: Path):
    nodes_path = index_dir / "create_final_nodes.parquet"
    rels_path = index_dir / "create_final_relationships.parquet"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing {nodes_path}")
    if not rels_path.exists():
        raise FileNotFoundError(f"Missing {rels_path}")
    nodes = pd.read_parquet(nodes_path)
    rels = pd.read_parquet(rels_path)
    return nodes, rels


def build_maps(nodes, rels):
    name_col, desc_col, type_col, id_col, community_col = infer_node_columns(nodes)
    source_col, target_col, rel_desc_col, confidence_col = infer_relationship_columns(rels)

    gene_nodes = {}
    pathway_nodes = {}
    pathway_id_to_name = {}
    pathway_name_to_desc = {}

    for _, row in nodes.iterrows():
        name = str(row.get(name_col, "")).strip()
        desc = str(row.get(desc_col, "")) if desc_col else ""
        rid = extract_reactome_id(row.get(id_col, "")) if id_col else extract_reactome_id(name)
        if is_gene_row(row, name_col, type_col, id_col):
            gene_nodes[name.upper()] = {
                "name": name.upper(),
                "description": desc,
                "community": row.get(community_col, None) if community_col else None,
            }
        elif is_pathway_row(row, name_col, type_col, id_col):
            pathway_nodes[name] = {
                "name": name,
                "description": desc,
                "reactome_id": rid,
            }
            if rid:
                pathway_id_to_name[rid] = name
            pathway_name_to_desc[name] = desc

    gene_to_pathways = defaultdict(set)
    pathway_to_genes = defaultdict(set)
    direct_gene_edges = defaultdict(list)

    for _, row in rels.iterrows():
        src = str(row.get(source_col, ""))
        tgt = str(row.get(target_col, ""))
        rel_text = str(row.get(rel_desc_col, "")) if rel_desc_col else ""
        confidence = row.get(confidence_col, None) if confidence_col else None

        src_gene = clean_gene_from_endpoint(src)
        tgt_gene = clean_gene_from_endpoint(tgt)
        src_path_id = clean_pathway_from_endpoint(src)
        tgt_path_id = clean_pathway_from_endpoint(tgt)

        if src_gene and tgt_path_id:
            pname = pathway_id_to_name.get(tgt_path_id, tgt_path_id)
            gene_to_pathways[src_gene].add(pname)
            pathway_to_genes[pname].add(src_gene)
        elif tgt_gene and src_path_id:
            pname = pathway_id_to_name.get(src_path_id, src_path_id)
            gene_to_pathways[tgt_gene].add(pname)
            pathway_to_genes[pname].add(tgt_gene)
        elif src_gene and tgt_gene and src_gene != tgt_gene:
            direct_gene_edges[tuple(sorted([src_gene, tgt_gene]))].append({
                "source": src_gene,
                "target": tgt_gene,
                "description": rel_text,
                "confidence": confidence,
            })

    return {
        "gene_nodes": gene_nodes,
        "pathway_nodes": pathway_nodes,
        "pathway_name_to_desc": pathway_name_to_desc,
        "gene_to_pathways": gene_to_pathways,
        "pathway_to_genes": pathway_to_genes,
        "direct_gene_edges": direct_gene_edges,
    }


def rank_pathways_pathway_only(genes, maps):
    counts = defaultdict(set)
    for g in genes:
        for p in maps["gene_to_pathways"].get(g, set()):
            counts[p].add(g)
    rows = []
    for p, gs in counts.items():
        desc = maps["pathway_name_to_desc"].get(p, "")
        rows.append({
            "pathway": p,
            "matched_genes": sorted(gs),
            "score": len(gs),
            "method_score": len(gs),
            "specificity": pathway_specificity_score(p, desc),
        })
    return sorted(rows, key=lambda x: (-x["score"], x["pathway"].lower()))


def rank_pathways_gene_only(genes, maps):
    # Gene-only baseline extracts pathway mentions from each gene node description.
    # It does not use graph edges, community information, or direct relations.
    counts = defaultdict(set)
    known_pathways = list(maps["pathway_nodes"].keys())
    for g in genes:
        desc = maps["gene_nodes"].get(g, {}).get("description", "")
        desc_low = desc.lower()
        for p in known_pathways:
            p_low = p.lower()
            if p_low and p_low in desc_low:
                counts[p].add(g)
    rows = []
    for p, gs in counts.items():
        desc = maps["pathway_name_to_desc"].get(p, "")
        rows.append({
            "pathway": p,
            "matched_genes": sorted(gs),
            "score": len(gs),
            "method_score": len(gs),
            "specificity": pathway_specificity_score(p, desc),
        })
    return sorted(rows, key=lambda x: (-x["score"], x["pathway"].lower()))


def rank_pathways_graphrag_gene(genes, maps):
    base = rank_pathways_pathway_only(genes, maps)
    gene_set = set(genes)

    # Add a direct relationship bonus when pathway contains genes connected to each other.
    direct_pairs = {
        pair for pair in maps["direct_gene_edges"]
        if pair[0] in gene_set and pair[1] in gene_set
    }

    rows = []
    for row in base:
        matched_count = len(row["matched_genes"])
        specificity = row["specificity"]
        relation_bonus = 0.0
        for a, b in direct_pairs:
            if a in row["matched_genes"] and b in row["matched_genes"]:
                relation_bonus += 0.25
        score = matched_count * 10 + specificity + relation_bonus
        rows.append({
            **row,
            "score": score,
            "method_score": score,
            "specificity": specificity,
            "direct_relation_bonus": relation_bonus,
        })
    return sorted(rows, key=lambda x: (-x["score"], -len(x["matched_genes"]), x["pathway"].lower()))


def is_hit(pathway_name, expected_keywords, match_mode="synonym"):
    if match_mode == "strict":
        return strict_keyword_hit(pathway_name, expected_keywords)
    return keyword_hit(pathway_name, expected_keywords)


def evaluate_method(ranked, expected_keywords, match_mode="synonym"):
    top_paths = [r["pathway"] for r in ranked]
    return {
        "top1_hit": int(any(is_hit(p, expected_keywords, match_mode) for p in top_paths[:1])),
        "top3_hit": int(any(is_hit(p, expected_keywords, match_mode) for p in top_paths[:3])),
        "top5_hit": int(any(is_hit(p, expected_keywords, match_mode) for p in top_paths[:5])),
        "top1_pathway": top_paths[0] if top_paths else "",
        "top3_pathways": " | ".join(top_paths[:3]),
        "top5_pathways": " | ".join(top_paths[:5]),
        "mean_top5_specificity": round(sum(r.get("specificity", 0) for r in ranked[:5]) / max(1, len(ranked[:5])), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", required=True, help="Directory containing create_final_nodes.parquet and create_final_relationships.parquet")
    parser.add_argument("--cases_file", default="data/evaluation_cases.csv", help="CSV file with columns: case_id, genes, expected_keywords")
    parser.add_argument("--output_dir", default="results/tables", help="Output directory for result CSV files")
    parser.add_argument("--match_mode", choices=["strict", "synonym"], default="synonym", help="Keyword matching mode for expected pathway labels.")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    cases_file = Path(args.cases_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes, rels = load_index(index_dir)
    maps = build_maps(nodes, rels)

    cases = pd.read_csv(cases_file)

    method_funcs = {
        "Gene-only retrieval": rank_pathways_gene_only,
        "Pathway-only retrieval": rank_pathways_pathway_only,
        "GraphRAG-Gene": rank_pathways_graphrag_gene,
    }

    case_rows = []
    all_top_rows = []

    for _, case in cases.iterrows():
        case_id = str(case["case_id"])
        genes = split_genes(case["genes"])
        expected_keywords = split_keywords(case["expected_keywords"])

        for method_name, func in method_funcs.items():
            ranked = func(genes, maps)
            metrics = evaluate_method(ranked, expected_keywords, args.match_mode)
            case_rows.append({
                "case_id": case_id,
                "genes": ";".join(genes),
                "expected_keywords": ";".join(expected_keywords),
                "method": method_name,
                "match_mode": args.match_mode,
                **metrics,
            })
            for rank, item in enumerate(ranked[:10], start=1):
                all_top_rows.append({
                    "case_id": case_id,
                    "method": method_name,
                    "rank": rank,
                    "pathway": item["pathway"],
                    "matched_genes": ";".join(item["matched_genes"]),
                    "score": item.get("score", ""),
                    "specificity": item.get("specificity", ""),
                    "match_mode": args.match_mode,
                    "is_expected_hit": int(is_hit(item["pathway"], expected_keywords, args.match_mode)),
                })

    case_df = pd.DataFrame(case_rows)
    top_df = pd.DataFrame(all_top_rows)

    summary_rows = []
    for method, group in case_df.groupby("method"):
        summary_rows.append({
            "method": method,
            "n_cases": len(group),
            "top1_hit_rate": round(group["top1_hit"].mean(), 4),
            "top3_hit_rate": round(group["top3_hit"].mean(), 4),
            "top5_hit_rate": round(group["top5_hit"].mean(), 4),
            "mean_top5_specificity": round(group["mean_top5_specificity"].mean(), 4),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("method")

    summary_path = output_dir / "baseline_comparison.csv"
    case_path = output_dir / "case_level_retrieval_results.csv"
    top_path = output_dir / "top10_retrieved_pathways_by_case.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    case_df.to_csv(case_path, index=False, encoding="utf-8-sig")
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")

    print("\nBaseline comparison summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to:\n- {summary_path}\n- {case_path}\n- {top_path}")


if __name__ == "__main__":
    main()
