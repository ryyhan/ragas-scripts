import json
import os
import pandas as pd

def extract_data_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract the generated answer
    answer = data.get('answer', '')
    
    # Extract contexts from vector search results (higher quality)
    vector_contexts = []
    for result in data.get('vector', {}).get('search_results', []):
        vector_contexts.append(result.get('content', ''))
    
    return {
        "question": "",  # Placeholder for manual input
        "answer": answer,
        "contexts": vector_contexts,  # List of retrieved contexts
        "ground_truth": ""  # Placeholder for manual input
    }

def generate_dataset(input_dir, output_file):
    dataset = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            entry = extract_data_from_json(file_path)
            dataset.append(entry)
    
    # Save as CSV (or JSONL)
    df = pd.DataFrame(dataset)
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")

# Example usage
generate_dataset(
    input_dir="inference_response",
    output_file="rag_dataset.csv"
)