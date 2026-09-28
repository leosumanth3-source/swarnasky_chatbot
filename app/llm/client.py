
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from groq import Groq


# Load variables from .env
load_dotenv()


class GroqLLM:
    """
    Groq LLM client for the Swarnasky chatbot.

    The API key is loaded from the GROQ_API_KEY
    environment variable.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:

        # ---------------------------------------------------------
        # Load API key
        # ---------------------------------------------------------
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured. "
                "Add GROQ_API_KEY to your .env file."
            )

        # ---------------------------------------------------------
        # Load model
        # ---------------------------------------------------------
        self.model = model or os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b",
        )

        self.temperature = temperature
        self.max_tokens = max_tokens

        # ---------------------------------------------------------
        # Create Groq client
        # ---------------------------------------------------------
        self.client = Groq(
            api_key=self.api_key,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Send a prompt to Groq and return the generated text.
        """

        # ---------------------------------------------------------
        # Validate input
        # ---------------------------------------------------------
        if not system_prompt.strip():
            raise ValueError(
                "system_prompt cannot be empty."
            )

        if not user_prompt.strip():
            raise ValueError(
                "user_prompt cannot be empty."
            )

        # ---------------------------------------------------------
        # Call Groq
        # ---------------------------------------------------------
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

        except Exception as exc:
            raise RuntimeError(
                f"Groq API request failed: {exc}"
            ) from exc

        # ---------------------------------------------------------
        # Validate response
        # ---------------------------------------------------------
        if not response.choices:
            raise RuntimeError(
                "Groq returned no response choices."
            )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        return content.strip()

