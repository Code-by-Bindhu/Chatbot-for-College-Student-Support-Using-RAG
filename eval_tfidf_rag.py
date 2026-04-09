import os
import csv
import argparse
from pathlib import Path
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()


# -------------------------------
# CLI Arguments
# -------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="TF-IDF RAG Evaluation")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of questions to evaluate"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting index"
    )
    return parser.parse_args()


# -------------------------------
# Load Knowledge Base
# -------------------------------

def load_chunks():
    data_path = Path("data/data.txt")
    text = data_path.read_text(encoding="utf-8")
    chunks = [line.strip() for line in text.split("\n") if line.strip()]
    return chunks




class TfidfRetriever:  # TF-IDF Retriever
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(documents)
    def retrieve(self, query, k=3):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors)[0]
        top_indices = similarities.argsort()[-k:][::-1]
        return [self.documents[i] for i in top_indices]
generation_llm = ChatGroq(    # LLM (Generation)
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=512
)
def generate_answer(context_docs, question):
    context = "\n\n".join(context_docs)
    prompt = f"""
You are a helpful college information assistant. Use ONLY the information in the Context to answer.

Context:
{context}

Question:
{question}
"""

    response = generation_llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# -------------------------------
# Load Evaluation Questions
# -------------------------------

def load_eval_questions():
    csv_path = Path("data/eval_questions.csv")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "question": row["question"],
                "ground_truth": row["ground_truth"]
            })
    return rows


# -------------------------------
# MAIN
# -------------------------------

def main():
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import (
        faithfulness,
        AnswerRelevancy,
        ContextUtilization,
        context_precision,
        context_recall,
    )
    from ragas.run_config import RunConfig
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    args = parse_args()

    documents = load_chunks()
    retriever = TfidfRetriever(documents)

    questions = load_eval_questions()

    start = args.offset
    end = start + args.limit if args.limit else None
    questions = questions[start:end]

    print(f"Evaluating {len(questions)} questions (TF-IDF RAG)...")

    eval_data = []

    for item in questions:
        docs = retriever.retrieve(item["question"], k=3)
        answer = generate_answer(docs, item["question"])

        eval_data.append({
            "user_input": item["question"],
            "retrieved_contexts": docs,
            "response": answer,
            "reference": item["ground_truth"],
        })

    dataset = EvaluationDataset.from_list(eval_data)

    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
    )

    run_config = RunConfig(
        timeout=300,
        max_workers=1
    )

    result = evaluate(
        dataset,
        metrics=[
            ContextUtilization(),
            faithfulness,
            AnswerRelevancy(strictness=1),
            context_precision,
            context_recall,
        ],
        llm=generation_llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    print("\n--- TF-IDF RAG Results ---")
    print(result)


if __name__ == "__main__":
    main()