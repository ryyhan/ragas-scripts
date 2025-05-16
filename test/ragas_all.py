#!/usr/bin/env python3
from ragas import SingleTurnSample
import ragas.metrics as M

# List of all the non‑LLM metric class names you care about
metric_names = [
    # RAG (non‑LLM retrieval metrics)
    "NonLLMContextPrecisionWithReference",
    "NonLLMContextRecall",

    # NVIDIA‑style
    "AnswerAccuracy",
    "ContextRelevance",
    "ResponseGroundedness",

    # Agent/Tool‑use
    "AgenticOrToolUse",
    "TopicAdherence",
    "ToolCallAccuracy",
    "AgentGoalAccuracy",

    # Natural‑language
    "SemanticSimilarity",
    "NonLLMStringSimilarity",
    "BleuScore",
    "RougeScore",
    "StringPresence",
    "ExactMatch",

    # SQL
    "ExecutionBasedDatacompyScore",
    "SQLQueryEquivalence",

    # General‑purpose (these often require extra args, so will be skipped)
    "AspectCritic",
    "SimpleCriteriaScoring",
    "RubricsBasedScoring",
    "InstanceSpecificRubricsScoring",

    # Other tasks
    "Summarization",
]

metrics = []
for name in metric_names:
    if not hasattr(M, name):
        print(f"Warning: `{name}` not found in ragas.metrics; skipping")
        continue

    cls = getattr(M, name)
    try:
        # try default instantiation
        metrics.append(cls())
    except TypeError as e:
        # likely missing required init args
        print(f"Warning: could not instantiate `{name}` ({e}); skipping")

test_data = {
    "user_input": (
        "summarise given text\n"
        "The company reported an 8% rise in Q3 2024, driven by strong "
        "performance in the Asian market. Sales in this region have significantly "
        "contributed to the overall growth. Analysts attribute this success to "
        "strategic marketing and product localization. The positive trend in the "
        "Asian market is expected to continue into the next quarter."
    ),
    "response": (
        "The company experienced an 8% increase in Q3 2024, largely due to effective "
        "marketing strategies and product adaptation, with expectations of continued "
        "growth in the coming quarter."
    ),
    "reference": (
        "The company reported an 8% growth in Q3 2024, primarily driven by strong "
        "sales in the Asian market, attributed to strategic marketing and localized "
        "products, with continued growth anticipated in the next quarter."
    ),
}
sample = SingleTurnSample(**test_data)

# Compute and dump each score
for metric in metrics:
    try:
        score = metric.single_turn_score(sample)
        print(f"{metric.name:30s}: {score:.4f}")
    except Exception as e:
        print(f"{metric.name:30s}: error ({e})")


"""


from ragas import SingleTurnSample
from ragas.metrics import (
    # Retrieval‑Augmented Generation
    LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference,
    NonLLMContextPrecisionWithReference,
    LLMContextRecall,
    NonLLMContextRecall,
    ContextEntityRecall,
    NoiseSensitivity,
    ResponseRelevancy,
    Faithfulness,
    MultimodalFaithfulness,
    MultimodalRelevance,
    # NVIDIA‑style metrics
    AnswerAccuracy,
    ContextRelevance,
    ResponseGroundedness,
    # Agent/Tool‑use metrics
    AgenticOrToolUse,
    TopicAdherence,
    ToolCallAccuracy,
    AgentGoalAccuracy,
    # Natural‑language comparison
    FactualCorrectness,
    SemanticSimilarity,
    NonLLMStringSimilarity,
    BleuScore,
    RougeScore,
    StringPresence,
    ExactMatch,
    # SQL metrics
    ExecutionBasedDatacompyScore,
    SQLQueryEquivalence,
    # General‑purpose
    AspectCritic,
    SimpleCriteriaScoring,
    RubricsBasedScoring,
    InstanceSpecificRubricsScoring,
    # Other tasks
    Summarization,
)

# --- your LLM and embeddings objects here ---
evaluator_llm = ...            # e.g. OpenAI(temperature=0)
evaluator_embeddings = ...     # e.g. OpenAIEmbeddings()

# build your list of metric instances
metrics = [
    # RAG
    LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    LLMContextPrecisionWithReference(llm=evaluator_llm),
    NonLLMContextPrecisionWithReference(),
    LLMContextRecall(llm=evaluator_llm),
    NonLLMContextRecall(),
    ContextEntityRecall(llm=evaluator_llm),
    NoiseSensitivity(llm=evaluator_llm),
    ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    Faithfulness(llm=evaluator_llm),
    MultimodalFaithfulness(llm=evaluator_llm),
    MultimodalRelevance(llm=evaluator_llm),

    # NVIDIA
    AnswerAccuracy(),
    ContextRelevance(),
    ResponseGroundedness(),

    # Agentic/tool
    AgenticOrToolUse(),
    TopicAdherence(),
    ToolCallAccuracy(),
    AgentGoalAccuracy(),

    # Natural‑language
    FactualCorrectness(llm=evaluator_llm),
    SemanticSimilarity(),
    NonLLMStringSimilarity(),
    BleuScore(),
    RougeScore(),
    StringPresence(),
    ExactMatch(),

    # SQL
    ExecutionBasedDatacompyScore(),
    SQLQueryEquivalence(),

    # General
    AspectCritic(),
    SimpleCriteriaScoring(),
    RubricsBasedScoring(),
    InstanceSpecificRubricsScoring(),

    # Other
    Summarization(),
]

# your sample data
test_data = {
    "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
    "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
    "reference": "The company reported an 8% growth in Q3 2024, primarily driven by strong sales in the Asian market, attributed to strategic marketing and localized products, with continued growth anticipated in the next quarter.",
}
sample = SingleTurnSample(**test_data)

# compute and print each score
for metric in metrics:
    try:
        score = metric.single_turn_score(sample)
        print(f"{metric.name:30s}: {score:.4f}")
    except Exception as e:
        print(f"{metric.name:30s}: skipped ({e})")

"""