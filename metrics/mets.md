**Metric**	**What It Measures**	**Why It Matters**	**How to Implement**	**References**
**Retrieval Metrics**				
**Context Precision**	Relevance of retrieved documents to the query [[4]][[8]]	Ensures the retriever avoids irrelevant or redundant information	Use Ragas’ `context_precision` or manual relevance scoring	[[4]][[8]]
**Context Recall**	Coverage of all relevant information in retrieved docs [[4]][[8]]	Identifies gaps in retrieval (e.g., missing key figures like £349 million)	Ragas’ `context_recall` (requires ground-truth references)	[[4]][[8]]
**Precision@k**	% of top-k retrieved docs that are relevant [[6]]	Prioritizes high-quality retrieval results	Tools like Pyserini or Hugging Face Evaluate	[[6]]
**Recall@k**	% of total relevant docs retrieved in top-k [[6]]	Ensures critical information isn’t missed	Pyserini or custom benchmarks	[[6]]
**MRR (Mean Reciprocal Rank)**	Rank of the first relevant document retrieved [[3]]	Rewards systems that surface relevant docs earlier	Custom script (e.g., `1/rank` of first relevant doc)	[[3]]
**NDCG (Normalized Discounted Cumulative Gain)**	Graded relevance of retrieved docs [[9]]	Prioritizes nuanced relevance (e.g., exact vs. partial matches)	Galileo AI or custom implementation	[[9]]
**Generation Metrics**				
**Faithfulness**	Whether the answer is grounded in retrieved contexts [[4]][[8]][[10]]	Prevents hallucinations (critical for financial accuracy)	Ragas’ `faithfulness` or DeepEval	[[4]][[8]][[10]]
**Answer Relevancy**	How well the answer addresses the query [[4]][[8]]	Ensures answers aren’t vague or off-topic	Ragas’ `answer_relevancy` or LLM-as-a-judge	[[4]][[8]]
**BLEU/ROUGE**	N-gram overlap with reference answers [[2]]	Measures factual alignment (e.g., "£349 million" vs. generated text)	NLTK or Hugging Face Evaluate	[[2]]
**Hallucination Detection**	Presence of ungrounded claims in the answer [[7]][[10]]	Critical for trust in sensitive domains (e.g., finance)	DeepEval or custom LLM prompts	[[7]][[10]]
**Hybrid Metrics**				
**Semantic Answer Similarity (SAS)**	Semantic alignment between generated and reference answers [[4]][[8]]	Captures meaning beyond keywords (e.g., "£349m" ≈ "£349 million")	Embedding-based similarity (e.g., Sentence Transformers)	[[4]][[8]]
**System-Level Metrics**				
**Latency**	Time taken for retrieval/generation [[9]]	Balances performance with user experience (e.g., keyword search took 4.8s)	Track `execution_time` logs	[[9]]
**User Feedback**	Real-world satisfaction (e.g., thumbs-up/down) [[4]][[9]]	Grounds metrics in actual user needs	Surveys or analytics tools	[[4]][[9]]
**DORA Metrics**	DevOps performance (e.g., deployment frequency, failure rate) [[7]][[9]]	Measures system reliability and agility	Monitoring tools like Prometheus	[[7]][[9]]