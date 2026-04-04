"""Prompt building functions for scam detection."""

from llm.prompt import generate_prompt


def build_prompt(message: str, strategy: str = "react") -> str:
    """
    Build a prompt for the given message based on strategy.
    
    Args:
        message: The input message to create a prompt for
        strategy: The prompting strategy to use
        
    Returns:
        A formatted prompt string ready for LLM execution
    """
    if strategy == "react":
        return generate_prompt(message)
    if strategy == "few_shot":
        return generate_prompt(message)
    else:
        raise NotImplementedError(f"Strategy '{strategy}' is not supported yet.") 