"""
Token counting implementation using tiktoken.

This module provides utilities for estimating token counts in text
and chat messages using OpenAI's tiktoken library.
"""

from typing import Union, List
import tiktoken

# Model configuration for token counting
model_name = "gpt-3.5-turbo"
encoding = tiktoken.encoding_for_model(model_name)


def approximate_count_tokens(messages: Union[str, List[dict]]) -> int:
    """
    Approximate the number of tokens in text or chat messages.
    
    Args:
        messages: Either a string text or list of message dictionaries
                 with 'content' keys (chat format)
                 
    Returns:
        Estimated token count as an integer
        
    Note:
        Uses GPT-3.5-turbo encoding for approximation.
        For chat messages, includes overhead tokens per message.
    """
    if isinstance(messages, dict):
        messages = [messages]
    # Tokenize and count tokens
    if isinstance(messages, str):
        return len(encoding.encode(messages))

    total_tokens = 0

    for message in messages:
        total_tokens += 4  # Approx token overhead per message
        total_tokens += len(encoding.encode(message.get("content")))

    total_tokens += 2  # Assistant reply overhead
    return total_tokens