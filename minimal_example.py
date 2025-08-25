#!/usr/bin/env python3
"""
MINIMAL EXAMPLE: How to feed SOURCE and SUMMARY texts into faithfulness evaluator.
"""

import asyncio
from faithfulness_evaluator import FaithfulnessEvaluator
from faithfulness_evaluator.data.loaders import GenericDataLoader

async def evaluate_summary_faithfulness(source_text: str, summary_claims: list[str]):
    """
    Simple function to evaluate if summary claims are faithful to source text.
    
    Args:
        source_text: The original document/book text
        summary_claims: List of claims extracted from a summary
    
    Returns:
        List of results with predictions for each claim
    """
    
    # 1. Create evaluator (choose method: "bm25" for speed, "full_text" for accuracy)
    evaluator = FaithfulnessEvaluator(method="bm25")
    
    # 2. Convert claims to proper format
    loader = GenericDataLoader()
    claims_data = loader.create_claims_from_list(summary_claims, "document")
    
    # 3. Load source text and claims
    evaluator.load_document_and_claims(source_text, claims_data)
    
    # 4. Evaluate all claims
    results = await evaluator.evaluate_all_claims()
    
    return results

# Example usage
async def main():
    # Your source document
    source = """
    John went to the store yesterday. He bought apples and oranges.
    The store was busy with many customers. John paid with cash.
    He walked home and ate an apple on the way.
    """
    
    # Claims from a summary you want to check
    claims = [
        "John went shopping yesterday",           # Should be: Yes (faithful)
        "John bought fruit at the store",        # Should be: Yes (apples/oranges are fruit)  
        "John paid with a credit card",          # Should be: No (he paid with cash)
        "John drove to the store",               # Should be: No/Inapplicable (doesn't say how he got there)
    ]
    
    # Evaluate faithfulness
    results = await evaluate_summary_faithfulness(source, claims)
    
    # Print results
    for i, result in enumerate(results):
        print(f"Claim: {claims[i]}")
        print(f"Faithful? {result.predicted_label}")
        print(f"Reasoning: {result.reasoning}")
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(main())