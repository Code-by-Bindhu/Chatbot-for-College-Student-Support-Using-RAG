"""
Optimized RAGAS evaluation for College RAG Chatbot

Generation  → 70B model (quality)
Evaluation  → 8B model (cost-efficient)

Usage:
    python eval_ragas.py
    python eval_ragas.py --limit 5
"""

import argparse
import os
import csv
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# -------------------------------
# MODELS (Split Setup)
# -------------------------------

MODELS = {
    "generation": "llama-3.3-70b-versatile",  # High-quality answers
    "evaluation": "llama-3.1-8b-instant",     # Cheaper evaluation
}


def get_llm(model_name: str):
    return ChatGroq(
        model=model_name,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        max_retries=5,
        max_tokens=1024
    )





def get_embeddings():  # Embeddings + Vector Store
    return HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
    )


def get_vectorstore():
    return PineconeVectorStore.from_existing_index(
        index_name=os.getenv("PINECONE_INDEX"),
        embedding=get_embeddings(),
    )


def generate_answer(llm, context_docs, question):  # RAG Generation
    context = "\n\n".join([d.page_content for d in context_docs])

    prompt = f"""
Answer ONLY from the context below.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# -------------------------------
# Load Evaluation Questions
# -------------------------------

def load_eval_questions(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("question") or "").strip()
            if not q:
                continue
            gt = (row.get("ground_truth") or "").strip() or None
            rows.append({"question": q, "ground_truth": gt})
    return rows


# -------------------------------
# RAG Pipeline
# -------------------------------

def run_rag_pipeline(vectorstore, llm, questions, k=5):
    results = []

    for item in questions:
        question = item["question"]

        docs = vectorstore.similarity_search(question, k=k)
        answer = generate_answer(llm, docs, question)
        contexts = [d.page_content for d in docs]

        results.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": item.get("ground_truth"),
        })

    return results


# -------------------------------
# CLI Arguments
# -------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Optimized RAGAS Evaluation")
    parser.add_argument(
        "--limit",
        type=int,
        default=3,  # SAFE default for free tier
        help="Number of questions to evaluate (default=3 for safety)",
    )
    return parser.parse_args()


# -------------------------------
# MAIN
# -------------------------------

def main():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from ragas import evaluate, EvaluationDataset
    from ragas.run_config import RunConfig
    from ragas.metrics import (
        AnswerRelevancy,
        faithfulness,
        ContextUtilization,
        context_precision,
        context_recall,
    )

    args = parse_args()

    csv_path = Path(__file__).parent / "data" / "eval_questions.csv"

    if not csv_path.exists():
        print("Evaluation CSV not found.")
        return

    questions = load_eval_questions(csv_path)

    if not questions:
        print("No evaluation questions found.")
        return

    questions = questions[: args.limit]
    print(f"Evaluating {len(questions)} questions...")

    # -------------------------------
    # Load Components
    # -------------------------------

    print("Loading RAG components...")

    generation_llm = get_llm(MODELS["generation"])
    evaluation_llm = get_llm(MODELS["evaluation"])

    embeddings = get_embeddings()
    vectorstore = get_vectorstore()

    # -------------------------------
    # Run RAG Generation
    # -------------------------------

    print("Running RAG pipeline...")
    eval_data = run_rag_pipeline(vectorstore, generation_llm, questions)

    dataset = EvaluationDataset.from_list(eval_data)

    # -------------------------------
    # Metrics (Reduced but strong)
    # -------------------------------

    metrics = [
        ContextUtilization(),
        faithfulness,
        AnswerRelevancy(strictness=1),
        context_precision,
        context_recall,
    ]

    run_config = RunConfig(
        timeout=300,
        max_workers=1,   # Prevent parallel calls (avoid 429)
        max_retries=5,
    )

    # -------------------------------
    # Run Evaluation (8B Model)
    # -------------------------------

    print("Running RAGAS evaluation...")
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=evaluation_llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    print("\n--- RAGAS Results ---")
    print(result)

    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        out_path = Path(__file__).parent / "ragas_results.csv"
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()