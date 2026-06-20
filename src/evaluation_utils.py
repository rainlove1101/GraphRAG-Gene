import re


SYNONYM_GROUPS = [
    ["nfkb", "nf-kb", "nf-kappa b", "nf-kappab", "nf-kappab", "nf-kappaB", "nf-kappa b"],
    ["rna polymerase ii", "rna polymerase 2", "pol ii", "polr2"],
    ["proteasome", "proteasomal", "apc/c", "ubiquitin", "protein degradation"],
    ["pi3k", "pi3k/akt", "phosphatidylinositol 3-kinase", "pip3"],
    ["tgf", "tgf-beta", "transforming growth factor"],
    ["egfr", "erbb", "receptor tyrosine kinase", "receptor tyrosine kinases"],
    ["toll like receptor", "toll-like receptor", "tlr"],
    ["ecm", "extracellular matrix"],
    ["base excision repair", "base-excision repair", "ber"],
    ["nucleotide excision repair", "ner"],
    ["homologous recombination", "homology directed repair", "hrr", "hdr"],
]


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("κ", "kappa")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def expanded_terms(term: str) -> set[str]:
    norm = normalize_text(term)
    terms = {norm}
    compact = norm.replace(" ", "")
    for group in SYNONYM_GROUPS:
        normalized_group = {normalize_text(x) for x in group}
        compact_group = {x.replace(" ", "") for x in normalized_group}
        if norm in normalized_group or compact in compact_group:
            terms.update(normalized_group)
    return {x for x in terms if x}


def keyword_hit(pathway_name: str, expected_keywords) -> bool:
    pathway_norm = normalize_text(pathway_name)
    pathway_compact = pathway_norm.replace(" ", "")
    for keyword in expected_keywords:
        for term in expanded_terms(keyword):
            if term in pathway_norm or term.replace(" ", "") in pathway_compact:
                return True
    return False


def strict_keyword_hit(pathway_name: str, expected_keywords) -> bool:
    text = str(pathway_name).lower()
    return any(str(keyword).lower() in text for keyword in expected_keywords)
