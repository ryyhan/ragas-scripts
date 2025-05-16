from datasets import Dataset
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    AnswerSimilarity,
)
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
# --- Configuration ---
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_API_KEY = "" # Replace with your actual key (use env vars!)
AZURE_OPENAI_API_VERSION = ""
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = "gpt-4.1"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-3-large"
# Configure Azure OpenAI Chat Model via LangChain
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
)
# Configure Azure OpenAI Embedding Model via LangChain
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    chunk_size=1000 # Optional: adjust chunk size if needed
)
# Wrap the LangChain LLM and Embeddings with Ragas wrappers
ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings) # Wrap the embedding model
# LLM-based metrics using the wrapped LLM
faithfulness = Faithfulness(llm=ragas_llm)
answer_relevancy = AnswerRelevancy(llm=ragas_llm)
answer_correctness = AnswerCorrectness(llm=ragas_llm)
# Metrics that don't require LLM (keep as default) - ContextPrecision and ContextRecall
# AnswerSimilarity requires embeddings, so we pass them later to evaluate
context_precision = ContextPrecision()
context_recall = ContextRecall()
answer_similarity = AnswerSimilarity()
# Example dataset
data = {
    "question": [
        "Can I sell equity to customers through self-directed route?",
        "Can you tell me about suitability?",
    ],
    "ground_truth": [
        "Yes, self-directed share dealing is allowed when the share is “non-complex” and the service is execution-only at the client’s initiative",
        "TSuitability means collecting information on the client’s knowledge & experience, financial situation (incl. capacity for loss) and investment objectives/risk-tolerance, then recommending or deciding only what is suitable for that profile",
    ],
    "answer": [
        "### Summary\nThe UK leverage ratio for Coventry Building Society in 2024 was reported as 5.7%, an increase from 5.4% in 2023. However,.........",
        "### Summary The net interest margin for Coventry Building Society in 2024 was reported as **1.07%**. This represents a decrease from.....",
    ],
    "contexts": [
        ["STRATEGIC REPORT: COBS 10A.4.1R  p 696","the Bank of England's target .","the needs of all of our customers and members, and ensuring the financial strength and stability of the combined Group. I believe the ...."],
        ["ounts 2024 5 Strategic Report Capital ratios Sustainable ....","the Bank of England's target range ...","In July, we also welcomed Ewa Kerin, "],
    ],
}
# Convert to HuggingFace dataset
dataset = Dataset.from_dict(data)
# Run evaluation with Ragas metrics, passing the configured embeddings
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        answer_correctness,
        answer_similarity,
        context_precision,
        context_recall,
    ],
    llm=ragas_llm, # Although metrics have LLMs, passing here can sometimes help or be required
    embeddings=ragas_embeddings # Pass the Ragas-wrapped embedding model
)
# Print results
print("RAG Evaluation Results:")
print(result)