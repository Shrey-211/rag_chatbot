"""Prompt template utilities."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Simple template for prompts with variable substitution."""

    def __init__(self, template: str):
        """Initialize prompt template.

        Args:
            template: Template string with {variable} placeholders
        """
        self.template = template

    def format(self, **kwargs) -> str:
        """Format template with provided variables.

        Args:
            **kwargs: Variables to substitute in template

        Returns:
            Formatted prompt string
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            raise ValueError(f"Missing required variable for prompt: {e}")

    @classmethod
    def default_rag_template(cls) -> "PromptTemplate":
        """Get default RAG prompt template.

        Returns:
            Default PromptTemplate for RAG
        """
        template = """Context information is below:
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the question.
If the answer is not in the context, say "I don't have enough information to answer this question."

Question: {query}
Answer:"""
        return cls(template)

    @classmethod
    def with_system_instruction(cls) -> "PromptTemplate":
        """Get RAG template with system instruction.

        Returns:
            PromptTemplate with system instruction
        """
        template = """{system_instruction}

Context information is below:
---------------------
{context}
---------------------

Question: {query}
Answer:"""
        return cls(template)


# Pre-defined templates
DEFAULT_SYSTEM_INSTRUCTION = """You are a helpful AI assistant. Answer questions based ONLY on the provided context.
If the context doesn't contain enough information to answer the question, say so clearly.
Be concise and accurate in your responses."""


RAG_TEMPLATE = PromptTemplate.default_rag_template()

RAG_WITH_SYSTEM = PromptTemplate(
    """You are a helpful AI assistant. Your task is to answer the question using ONLY the information provided in the context below.

IMPORTANT INSTRUCTIONS:
- Use ONLY the information from the context to answer
- Quote relevant parts from the context when possible
- If the context contains relevant information, provide a detailed answer
- If the answer is NOT in the context, respond with: "I don't have enough information in the provided context to answer this question."
- Do NOT use your prior knowledge or make assumptions

CONTEXT:
---------------------
{context}
---------------------

QUESTION: {query}

ANSWER (using only the context above):"""
)

CHAT_TEMPLATE = PromptTemplate(
    """You are a helpful AI assistant.

User: {query}
Assistant:"""
)

