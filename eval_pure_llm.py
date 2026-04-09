import os
import csv
import argparse
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

# CLI Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Pure LLM Evaluation (No Retrieval)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of questions to evaluate (default = all)"
    )
    return parser.parse_args()


# LLM (Pure Generative)
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # Smaller model for stable evaluation
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=1024
)
def generate_answer(question):
    prompt = f"""
You are a helpful assistant.Answer the question clearly and concisely.

Question:
{question}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# -------------------------------
# Load Evaluation Questions
# -------------------------------

def load_eval_questions(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "").strip()
            gt = row.get("ground_truth", "").strip()
            if q:
                rows.append({
                    "question": q,
                    "ground_truth": gt
                })
    return rows


# -------------------------------
# MAIN
# -------------------------------

def main():
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import faithfulness, AnswerRelevancy
    from ragas.run_config import RunConfig

    args = parse_args()

    csv_path = Path("data/eval_questions.csv")
    questions = load_eval_questions(csv_path)

    if args.limit:
        questions = questions[:args.limit]

    print(f"Evaluating {len(questions)} questions (Pure LLM)...")

    eval_data = []

    for item in questions:
        question = item["question"]
        answer = generate_answer(question)

        eval_data.append({
            "user_input": question,
            "retrieved_contexts": [],   # No retrieval
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
        max_workers=1,   # Prevent parallel overload
        max_retries=3
    )

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            AnswerRelevancy(strictness=1),
        ],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    print("\n--- Pure LLM Results ---")
    print(result)


if __name__ == "__main__":
    main()