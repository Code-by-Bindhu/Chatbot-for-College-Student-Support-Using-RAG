import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Groq LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Embeddings
embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)

# Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX")

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


def generate_answer(context_docs, question):
    context = "\n\n".join([d.page_content for d in context_docs])

    prompt = f"""
You are an AI assistant for CVR College of Engineering.

Use ONLY the information in the context to answer.

Return plain text only.
Do NOT use markdown or HTML symbols such as **, __, ##, *, backticks, or tags.

Format:
1) A short title on the first line
2) Then bullet points starting with "- "

Context:
{context}

Question:
{question}

Answer:
"""
    return llm.invoke([HumanMessage(content=prompt)]).content


@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question missing"}), 400

    docs = vectorstore.similarity_search(question, k=12)
    answer = generate_answer(docs, question)

    return jsonify({"answer": answer})


@app.route("/debug-search")
def debug_search():
    query = "cvr.ac.in"
    docs = vectorstore.similarity_search(query, k=7)

    results = []
    for i, d in enumerate(docs):
        results.append({
            "rank": i + 1,
            "content": d.page_content
        })

    return jsonify(results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
