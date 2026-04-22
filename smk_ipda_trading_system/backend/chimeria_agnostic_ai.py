"""
Supports multiple AI providers for the "Ask a Model" feature:
- Google Gemini (via google-genai)
- OpenAI GPT
- Anthropic Claude
- Local models via Ollama
- Custom endpoints
"""

from typing import Optional, Dict, Any, List
from enum import Enum
import os
import json
import httpx
import asyncio
import concurrent.futures
from dataclasses import dataclass

class AIProvider(Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    CUSTOM = "custom"

@dataclass
class AIModelConfig:
    provider: AIProvider
    api_key: Optional[str] = None
    model_name: str = "gemini-2.0-flash-exp"
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048

class AgnosticAIService:
    """Unified interface for multiple AI providers"""

    def __init__(self, config: AIModelConfig):
        self.config = config
        self._setup_client()

    def _setup_client(self):
        if self.config.provider == AIProvider.GEMINI:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.config.api_key)
            except ImportError:
                self.client = None

        elif self.config.provider == AIProvider.OPENAI:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.config.api_key)
            except ImportError:
                self.client = None

        elif self.config.provider == AIProvider.ANTHROPIC:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.config.api_key)
            except ImportError:
                self.client = None

        elif self.config.provider == AIProvider.OLLAMA:
            self.client = None  # REST API calls
            self.api_base = self.config.api_base or "http://localhost:11434"

        elif self.config.provider == AIProvider.CUSTOM:
            self.client = None
            self.api_base = self.config.api_base

    async def ask(self, prompt: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Send prompt to configured AI model with SMK context"""

        system_prompt = self._build_system_prompt(context)
        full_prompt = f"{system_prompt}\n\nUser: {prompt}"

        if self.config.provider == AIProvider.GEMINI:
            response = await self._ask_gemini(full_prompt)
        elif self.config.provider == AIProvider.OPENAI:
            response = await self._ask_openai(full_prompt)
        elif self.config.provider == AIProvider.ANTHROPIC:
            response = await self._ask_anthropic(full_prompt)
        elif self.config.provider == AIProvider.OLLAMA:
            response = await self._ask_ollama(full_prompt)
        else:
            response = await self._ask_custom(full_prompt)

        return {
            "provider": self.config.provider.value,
            "model": self.config.model_name,
            "response": response,
            "prompt": prompt
        }

    def _build_system_prompt(self, context: Optional[Dict]) -> str:
        """Build context-aware system prompt with SMK data"""
        base = """You are QUIMERIA AI, a trading assistant for the Sovereign Market Kernel (SMK) system.
You have access to real-time market analysis from 18+ detection modules including:
- IPDA AMD phase (Accumulation/Manipulation/Distribution/Retracement)
- Bias detection (Bullish/Bearish/Neutral)
- Fair Value Gaps (FVGs)
- Order blocks
- Volatility decay (λ1 entrapment)
- Harmonic traps (λ3 spectral inversion)
- Displacement detection (λ6)
- KL divergence regime shifts
- Topological fracture detection

Provide concise, actionable trading insights. Always note risk and never guarantee outcomes."""

        if context and context.get("smk_result"):
            smk = context["smk_result"]
            base += f"""

Current SMK State:
- AMD Phase: {smk.get('amd', {}).get('state', 'Unknown')}
- Bias: {smk.get('bias', {}).get('bias', 'Neutral')}
- Fusion Signal: {smk.get('fusion', {}).get('p_fused', 0):+.3f}
- Veto Decision: {smk.get('veto', {}).get('decision', 'Unknown')}
- FVGs Active: {len(smk.get('fvg', {}).get('recent', [])) > 0}
- Manipulation Detected: {smk.get('manipulation', {}).get('active', False)}
- KL Divergence: {smk.get('kl', {}).get('score', 0):.3f}
"""
        return base

    async def _ask_gemini(self, prompt: str) -> str:
        if not self.client: return "Gemini client not initialized. Install google-genai."
        # google-genai-python uses sync or async client. self.client.aio is the async one.
        try:
            response = await self.client.aio.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                }
            )
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"

    async def _ask_openai(self, prompt: str) -> str:
        if not self.client: return "OpenAI client not initialized. Install openai."
        # OpenAI client in this environment is likely sync. Run in thread.
        try:
            loop = asyncio.get_event_loop()
            def sync_call():
                return self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
            response = await loop.run_in_executor(None, sync_call)
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI Error: {str(e)}"

    async def _ask_anthropic(self, prompt: str) -> str:
        if not self.client: return "Anthropic client not initialized. Install anthropic."
        # Anthropic client is also typically sync.
        try:
            loop = asyncio.get_event_loop()
            def sync_call():
                return self.client.messages.create(
                    model=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
            response = await loop.run_in_executor(None, sync_call)
            return response.content[0].text
        except Exception as e:
            return f"Anthropic Error: {str(e)}"

    async def _ask_ollama(self, prompt: str) -> str:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/api/generate",
                    json={
                        "model": self.config.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.config.temperature,
                            "num_predict": self.config.max_tokens
                        }
                    },
                    timeout=30.0
                )
                data = response.json()
                return data.get("response", "")
        except Exception as e:
            return f"Ollama Error: {str(e)}"

    async def _ask_custom(self, prompt: str) -> str:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/chat",
                    json={"prompt": prompt, "temperature": self.config.temperature},
                    timeout=30.0
                )
                return response.json().get("response", "")
        except Exception as e:
            return f"Custom AI Error: {str(e)}"
