import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from run_baseline_experiment import (
    build_maps,
    evaluate_method,
    is_hit,
    load_index,
    pathway_specificity_score,
    rank_pathways_pathway_only,
    split_genes,
    split_keywords,
)


def direct_pairs_for_query(genes, maps):
    gene_set = set(genes)
    return {
        pair for pair in maps["direct_gene_edges"]
        if pair[0] in gene_set and pair[1] in gene_set
    }


def rank_graph_variant(genes, maps, use_specificity=True, use_relation_bonus=True):
    base = rank_pathways_pathway_only(genes, maps)
    direct_pairs = direct_pairs_for_query(genes, maps)

    rows = []
    for row in base:
        matched_genes = set(row["matched_genes"])
        matched_count = len(matched_genes)
        specificity = pathway_specificity_score(
            row["pathway"],
            maps["pathway_name_to_desc"].get(row["pathway"], ""),
        )
        relation_bonus = 0.0
        if use_relation_bonus:
            for a, b in direct_pairs:
                if a in matched_genes and b in matched_genes:
                    relation_bonus += 0.25

        score = matched_count * 10
        if use_specificity:
            score += specificity
        score += relation_bonus

        rows.append({
            **row,
            "score": score,
            "method_score": score,
            "specificity": specificity,
            "direct_relation_bonus": relation_bonus,
        })

    return sorted(rows, key=lambda x: (-x["score"], -len(x["matched_genes"]), x["pathway"].lower()))


def rank_overlap_only(genes, maps):
    counts = defaultdict(set)
    for g in genes:
        for p in maps["gene_to_pathways"].get(g, set()):
            counts[p].add(g)

    rows = []
    for pathway, matched_genes in counts.items():
        rows.append({
            "pathway": pathway,
            "matched_genes": sorted(matched_genes),
            "score": len(matched_genes),
            "method_score": len(matched_genes),
            "specificity": pathway_specificity_score(
                pathway,
                maps["pathway_name_to_desc"].get(pathway, ""),
            ),
            "direct_relation_bonus": 0.0,
        })

    return sorted(rows, key=lambda x: (-x["score"], x["pathway"].lower()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--cases_file", default="data/evaluation_cases.csv")
    parser.add_argument("--output_dir", default="results/tables")
    parser.add_argument("--match_mode", choices=["strict", "synonym"], default="synonym")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes, rels = load_index(Path(args.index_dir))
    maps = build_maps(nodes, rels)
    cases = pd.read_csv(args.cases_file)

    variants = {
        "GraphRAG-Gene full": lambda genes: rank_graph_variant(genes, maps, True, True),
        "No direct-relation bonus": lambda genes: rank_graph_variant(genes, maps, True, False),
        "No specificity weighting": lambda genes: rank_graph_variant(genes, maps, False, True),
        "Overlap count only": lambda genes: rank_overlap_only(genes, maps),
    }

    case_rows = []
    top_rows = []

    for _, case in cases.iterrows():
        case_id = str(case["case_id"])
        genes = split_genes(case["genes"])
        expected_keywords = split_keywords(case["expected_keywords"])

        for method, ranker in variants.items():
            ranked = ranker(genes)
            metrics = evaluate_method(ranked, expected_keywords, args.match_mode)
            case_rows.append({
                "case_id": case_id,
                "genes": ";".join(genes),
                "expected_keywords": ";".join(expected_keywords),
                "method": method,
                "match_mode": args.match_mode,
                **metrics,
            })
            for rank, item in enumerate(ranked[:10], start=1):
                top_rows.append({
                    "case_id": case_id,
                    "method": method,
                    "rank": rank,
                    "pathway": item["pathway"],
                    "matched_genes": ";".join(item["matched_genes"]),
                    "score": item.get("score", ""),
                    "specificity": item.get("specificity", ""),
                    "direct_relation_bonus": item.get("direct_relation_bonus", ""),
                    "match_mode": args.match_mode,
                    "is_expected_hit": int(is_hit(item["pathway"], expected_keywords, args.match_mode)),
                })

    case_df = pd.DataFrame(case_rows)
    top_df = pd.DataFrame(top_rows)

    summary_df = (
        case_df.groupby("method", as_index=False)
        .agg(
            n_cases=("case_id", "count"),
            top1_hit_rate=("top1_hit", "mean"),
            top3_hit_rate=("top3_hit", "mean"),
            top5_hit_rate=("top5_hit", "mean"),
            mean_top5_specificity=("mean_top5_specificity", "mean"),
        )
        .sort_values("method")
    )
    for col in ["top1_hit_rate", "top3_hit_rate", "top5_hit_rate", "mean_top5_specificity"]:
        summary_df[col] = summary_df[col].round(4)

    summary_path = output_dir / "ablation_comparison.csv"
    case_path = output_dir / "ablation_case_level_results.csv"
    top_path = output_dir / "ablation_top10_retrieved_pathways_by_case.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    case_df.to_csv(case_path, index=False, encoding="utf-8-sig")
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")

    print("\nAblation comparison summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to:\n- {summary_path}\n- {case_path}\n- {top_path}")


if __name__ == "__main__":
    main()
