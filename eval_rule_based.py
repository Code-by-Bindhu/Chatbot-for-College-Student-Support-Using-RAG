import os
import csv
from pathlib import Path
from dotenv import load_dotenv

from ragas import evaluate, EvaluationDataset
from ragas.run_config import RunConfig
from ragas.metrics import (
    AnswerRelevancy,
    faithfulness,
)

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

# Evaluation model (cheap)
evaluation_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=1024,
)

embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
)


# Simple Rule-Based Logic


def rule_based_answer(question):
    data_path = Path(__file__).parent / "data" / "data.txt"
    text = data_path.read_text(encoding="utf-8")

    # naive keyword matching
    for line in text.split("\n"):
        if any(word.lower() in line.lower() for word in question.split()):
            return line

    return "No Information available, Please visit the official website cvr.ac.in"



# Load Evaluation Questions


def load_eval_questions(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "question": row["question"],
                "ground_truth": row["ground_truth"]
            })
    return rows


def main():
    csv_path = Path(__file__).parent / "data" / "eval_questions.csv"
    questions = load_eval_questions(csv_path)

    eval_data = []

    for item in questions:
        answer = rule_based_answer(item["question"])

        eval_data.append({
            "user_input": item["question"],
            "retrieved_contexts": [],   # No retrieval
            "response": answer,
            "reference": item["ground_truth"],
        })

    dataset = EvaluationDataset.from_list(eval_data)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            AnswerRelevancy(strictness=1),
        ],
        llm=evaluation_llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1),
    )

    print(result)


if __name__ == "__main__":
    main()