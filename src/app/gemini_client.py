"""
Gemini Client - Unified interface for Vertex AI Gemini

Provides:
- Text generation (replaces ChatOllama)
- Vision/multimodal analysis (replaces Qwen2.5-VL)
- Embeddings (replaces nomic-embed-text)
"""

import os
import json
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path

# Vertex AI imports
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from vertexai.language_models import TextEmbeddingModel


# Configuration
GCP_PROJECT = os.getenv("GCP_PROJECT", "meetingmind-ai-483117")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

# Model names
GEMINI_FLASH = "gemini-2.0-flash"
EMBEDDING_MODEL = "text-embedding-004"

# Initialize Vertex AI
_initialized = False


def _ensure_initialized():
    """Initialize Vertex AI if not already done."""
    global _initialized
    if not _initialized:
        vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
        _initialized = True


class GeminiClient:
    """Unified Gemini client for text and vision tasks."""
    
    def __init__(
        self,
        model_name: str = GEMINI_FLASH,
        temperature: float = 0.3,
        max_output_tokens: int = 4096,
    ):
        _ensure_initialized()
        self.model = GenerativeModel(model_name)
        self.config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    
    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Generate text response from prompt.
        Replaces: _ollama_chat() in phase4.py and name_extraction.py
        """
        try:
            # Build content
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=self.config,
            )
            
            return response.text
            
        except Exception as e:
            print(f"Gemini text generation error: {e}")
            raise
    
    def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/jpeg",
    ) -> str:
        """
        Analyze an image with a prompt.
        Replaces: VLMAnalyzer.analyze_frame() in phase5_visual.py
        """
        try:
            # Create image part
            image_part = Part.from_data(image_bytes, mime_type=mime_type)
            
            response = self.model.generate_content(
                [prompt, image_part],
                generation_config=self.config,
            )
            
            return response.text
            
        except Exception as e:
            print(f"Gemini vision error: {e}")
            raise
    
    def analyze_image_base64(
        self,
        base64_image: str,
        prompt: str,
        mime_type: str = "image/jpeg",
    ) -> str:
        """
        Analyze a base64-encoded image.
        Convenience wrapper for existing code that uses base64.
        """
        image_bytes = base64.b64decode(base64_image)
        return self.analyze_image(image_bytes, prompt, mime_type)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
    ) -> str:
        """
        Multi-turn chat interface.
        Replaces: ChatOllama in rag.py
        
        Messages format: [{"role": "user"|"assistant", "content": "..."}]
        """
        try:
            # Convert messages to Gemini format
            chat = self.model.start_chat()
            
            # Process all messages except the last one as history
            for msg in messages[:-1]:
                if msg["role"] == "user":
                    chat.send_message(msg["content"])
                # Note: Gemini chat handles assistant messages automatically
            
            # Send the last message and get response
            if messages:
                last_msg = messages[-1]
                response = chat.send_message(
                    last_msg["content"],
                    generation_config=GenerationConfig(temperature=temperature),
                )
                return response.text
            
            return ""
            
        except Exception as e:
            print(f"Gemini chat error: {e}")
            raise


class GeminiEmbeddings:
    """
    Gemini embeddings for RAG.
    Replaces: OllamaEmbeddings in rag.py
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        _ensure_initialized()
        self.model = TextEmbeddingModel.from_pretrained(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.get_embeddings(texts)
        return [emb.values for emb in embeddings]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embeddings = self.model.get_embeddings([text])
        return embeddings[0].values


# Convenience functions for drop-in replacement
def gemini_chat(
    messages: List[Dict[str, str]],
    model: str = GEMINI_FLASH,
    temperature: float = 0.3,
    **kwargs
) -> str:
    """
    Drop-in replacement for _ollama_chat().
    
    Usage:
        # Before: response = _ollama_chat(model, messages, temperature=0.2)
        # After:  response = gemini_chat(messages, temperature=0.2)
    """
    client = GeminiClient(model_name=model, temperature=temperature)
    return client.chat(messages, temperature=temperature)


def gemini_generate(
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.3,
    **kwargs
) -> str:
    """
    Simple text generation.
    
    Usage:
        response = gemini_generate("Summarize this meeting...")
    """
    client = GeminiClient(temperature=temperature)
    return client.generate_text(prompt, system_instruction)


def gemini_vision(
    image_base64: str,
    prompt: str,
    temperature: float = 0.1,
    **kwargs
) -> str:
    """
    Analyze image with VLM.
    
    Usage:
        # Before: result = vlm_analyzer.analyze_frame(frame, context)
        # After:  result = gemini_vision(base64_image, prompt)
    """
    client = GeminiClient(temperature=temperature)
    return client.analyze_image_base64(image_base64, prompt)


# For LangChain compatibility
class VertexAIEmbeddingsWrapper:
    """
    LangChain-compatible embeddings wrapper.
    Drop-in replacement for OllamaEmbeddings.
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._embeddings = GeminiEmbeddings(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)
