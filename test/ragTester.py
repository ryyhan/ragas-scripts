import json
import csv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, answer_relevancy
from dotenv import load_dotenv
import os

load_dotenv()

# Load RAG output
with open('rag_output.json', 'r') as f:
    rag_data = json.load(f)

# Prepare dataset
question = "What is the profit before tax excluding acquisition and integration costs?"
combined_contexts = [
    result['content'] for result in rag_data['vector']['search_results']
] + [
    result['content'] for result in rag_data['keyword']['search_results']
]

dataset = Dataset.from_dict({
    "question": [question],
    "contexts": [combined_contexts],
    "answer": [rag_data['answer']],
    "reference": ["£349 million"]
})

# Run evaluation
results = evaluate(
    dataset,
    metrics=[context_precision, faithfulness, answer_relevancy]
)

# Prepare CSV data
csv_data = {
    "Context Precision": results['context_precision'][0],
    "Faithfulness": results['faithfulness'][0],
    "Answer Relevancy": results['answer_relevancy'][0]
}

# Save to CSV
csv_filename = "rag_evaluation_results.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Metric", "Score"])
    for metric, score in csv_data.items():
        writer.writerow([metric, f"{score:.2f}"])

print(f"Evaluation results saved to {csv_filename}")