from __future__ import annotations


def build_query_expansion_prompt(question: str, variant_count: int) -> str:
    return (
        f"Generate exactly {variant_count} alternative phrasings of the question below. "
        "Each phrasing must preserve the exact same intent and meaning -- do not introduce "
        "new questions, assumptions, or narrower/broader scope. Vary vocabulary and sentence "
        "structure so a search system that only matches on wording (not meaning) has more "
        "chances to find relevant documents, phrased the way those documents might phrase it.\n\n"
        f"Question: {question}\n\n"
        f"Respond with exactly {variant_count} phrasings."
    )


def build_generation_prompt(question: str, numbered_context: str) -> str:
    return (
        "You are a precise assistant answering strictly from the provided context.\n"
        "Rules:\n"
        "1. Answer ONLY using facts present in the numbered context blocks below.\n"
        "2. Cite every factual claim inline with its source marker, e.g. [1], [2].\n"
        "3. If the context does not contain the answer, say so plainly instead of guessing.\n"
        "4. Never use outside knowledge, even if you believe it to be true.\n\n"
        f"Context:\n{numbered_context}\n\n"
        f"Question: {question}\n\n"
        "Respond with the answer text and the list of source markers you actually used."
    )


def build_judge_prompt(question: str, answer: str, numbered_context: str) -> str:
    return (
        "You are a strict fact-checking judge. Decide whether the ANSWER below is fully "
        "supported by the CONTEXT below. Use ONLY the context — never outside or world "
        "knowledge, even if the answer happens to be true in general.\n\n"
        "Steps:\n"
        "1. Break the answer into its atomic factual claims.\n"
        "2. For each claim, decide if it is supported by the context. If supported, quote the "
        "exact literal sentence or phrase from the context that supports it — the quote must be "
        "a verbatim substring of the context.\n"
        "3. Give an overall verdict: 'grounded' (all claims supported), 'partially_grounded' "
        "(some supported), or 'hallucinated' (none supported, or contradicted by the context).\n"
        "4. Give an overall confidence score from 0 to 100 reflecting how well-grounded the "
        "answer is in the context.\n\n"
        f"Context:\n{numbered_context}\n\n"
        f"Question: {question}\n\n"
        f"Answer to judge: {answer}\n"
    )
