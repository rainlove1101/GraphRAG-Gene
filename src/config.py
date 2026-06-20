import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

@dataclass
class Settings:
    oneapi_base_url: str = os.getenv("ONEAPI_BASE_URL", "http://localhost:3000/v1")
    oneapi_api_key: str | None = os.getenv("ONEAPI_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "qwen-plus")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
    batch_size: int = int(os.getenv("BATCH_SIZE", "10"))
    min_community_size: int = int(os.getenv("MIN_COMMUNITY_SIZE", "3"))
    min_confidence_score: float = float(os.getenv("MIN_CONFIDENCE_SCORE", "0.9"))
    leiden_resolution: float = float(os.getenv("LEIDEN_RESOLUTION", "1.0"))

COMMUNITY_SUMMARY_PROMPT = """
You are summarizing a biomedical knowledge graph community for pathway-level gene interpretation.

Use only the node descriptions below. Write a concise, evidence-grounded module summary that covers:
1. Dominant biological processes or pathway themes
2. Representative genes and pathways when present
3. Any disease-related pathway evidence only when explicitly stated
4. Important limitations of the module context

Avoid clinical diagnosis, personal risk estimation, treatment recommendations, or unsupported causal claims.

Node descriptions:
{node_texts}
""".strip()

GENE_INTERPRET_PROMPT = """
You are a biomedical knowledge graph interpretation assistant.

Your task is to interpret the user's query genes based ONLY on the retrieved knowledge graph context provided below.

Important rules:
1. Do NOT make clinical diagnoses.
2. Do NOT estimate personal disease risk.
3. Do NOT recommend treatment or medication.
4. Do NOT infer cancer type, treatment sensitivity, drug response, or clinical risk unless explicitly supported by the retrieved context.
5. Do NOT use external medical knowledge that is not present in the retrieved context.
6. If variant-level information is not provided, clearly state that variant pathogenicity cannot be assessed.
7. If disease association is only implied by pathway names, describe it cautiously as "disease-related pathway evidence" rather than a confirmed disease conclusion.
8. Always include a limitations section.
9. Avoid strong causal language such as "strongly suggests", "proves", or "confirms" unless the retrieved context explicitly supports it. Prefer cautious terms such as "supports", "is consistent with", or "indicates".

Please generate a structured interpretation with the following sections:

1. Gene functions and pathway associations
2. Direct relationships among query genes
3. Shared or nearby biological pathways
4. Disease-related pathway evidence, if explicitly present in the context
5. Limitations
6. Evidence-grounded conclusion

Retrieved knowledge graph context:
{context}

User query:
{question}
""".strip()
