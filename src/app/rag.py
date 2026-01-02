"""
Meeting RAG Service - LangChain LCEL Implementation

Provides Q&A functionality over meeting content using:
- ChromaDB for vector storage
- Vertex AI Gemini for embeddings and LLM
- Simple message history for conversation
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

# LangChain imports - using stable packages
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Configuration
GCP_PROJECT = os.getenv("GCP_PROJECT", "meetingmind-ai-483117")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-1.5-flash-002"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5


class MeetingRAG:
    """RAG service for a single meeting with conversation memory."""
    
    def __init__(self, meeting_id: str, data_root: str = "src/data/meetings"):
        self.meeting_id = meeting_id
        self.meeting_root = Path(data_root) / meeting_id
        self.vectorstore_path = self.meeting_root / "vectorstore"
        
        # Initialize embeddings (Vertex AI)
        self.embeddings = VertexAIEmbeddings(
            model_name=EMBEDDING_MODEL,
            project=GCP_PROJECT,
            location=GCP_LOCATION,
        )
        
        # Initialize LLM (Vertex AI Gemini)
        self.llm = ChatVertexAI(
            model_name=LLM_MODEL,
            project=GCP_PROJECT,
            location=GCP_LOCATION,
            temperature=0.3,
        )
        
        # Simple message history (last 10 messages)
        self.chat_history: List = []
        self.max_history = 10
        
        # Load or build vectorstore
        self.vectorstore = self._load_or_build_vectorstore()
        
        # Build retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K}
        )
        
        # Build RAG chain
        self.chain = self._build_chain()
    
    def _load_meeting_data(self) -> Dict[str, Any]:
        """Load all meeting data files."""
        data = {
            "transcript": [],
            "summary": {},
            "visual_insights": []
        }
        
        # Load transcript
        transcript_path = self.meeting_root / "phase3" / "labeled_transcript.json"
        if not transcript_path.exists():
            transcript_path = self.meeting_root / "phase1" / "transcript.json"
        
        if transcript_path.exists():
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
                data["transcript"] = transcript_data.get("segments", [])
        
        # Load summary/notes
        notes_path = self.meeting_root / "phase4" / "meeting_notes.json"
        if notes_path.exists():
            with open(notes_path, "r", encoding="utf-8") as f:
                data["summary"] = json.load(f)
        
        # Load visual insights
        visual_path = self.meeting_root / "phase5" / "visual_insights.json"
        if visual_path.exists():
            with open(visual_path, "r", encoding="utf-8") as f:
                visual_data = json.load(f)
                data["visual_insights"] = visual_data.get("insights", [])
        
        return data
    
    def _create_documents(self, data: Dict[str, Any]) -> List[Document]:
        """Convert meeting data to LangChain Documents with metadata."""
        documents = []
        
        # Group transcript segments into chunks
        current_chunk = []
        current_length = 0
        
        for seg in data["transcript"]:
            speaker = seg.get("speaker_label", seg.get("speaker", "Unknown"))
            text = seg.get("text", "")
            timestamp = seg.get("start", 0)
            
            line = f"{speaker}: {text}"
            current_chunk.append({
                "text": line,
                "timestamp": timestamp,
                "speaker": speaker
            })
            current_length += len(line)
            
            if current_length >= CHUNK_SIZE:
                chunk_text = "\n".join([c["text"] for c in current_chunk])
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source_type": "transcript",
                        "start_timestamp": current_chunk[0]["timestamp"],
                        "end_timestamp": current_chunk[-1]["timestamp"],
                    }
                )
                documents.append(doc)
                current_chunk = []
                current_length = 0
        
        # Handle remaining transcript
        if current_chunk:
            chunk_text = "\n".join([c["text"] for c in current_chunk])
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source_type": "transcript",
                    "start_timestamp": current_chunk[0]["timestamp"],
                    "end_timestamp": current_chunk[-1]["timestamp"],
                }
            )
            documents.append(doc)
        
        # Process summary sections
        summary = data["summary"]
        summary_sections = [
            ("overview", summary.get("overview", "")),
            ("key_points", "\n".join(f"• {p}" for p in summary.get("key_points", []))),
            ("decisions", "\n".join(f"• {d}" for d in summary.get("decisions", []))),
            ("action_items", "\n".join(
                f"• {a.get('task', a) if isinstance(a, dict) else a}" 
                for a in summary.get("action_items", [])
            )),
            ("next_steps", "\n".join(f"• {n}" for n in summary.get("next_steps", []))),
        ]
        
        for section_name, content in summary_sections:
            if content and content.strip():
                doc = Document(
                    page_content=content,
                    metadata={
                        "source_type": "summary",
                        "section": section_name
                    }
                )
                documents.append(doc)
        
        # Process visual insights
        for insight in data["visual_insights"]:
            content_type = insight.get("content_type", "")
            if content_type in ["camera_only", "duplicate", "unknown"]:
                continue
            
            timestamp = insight.get("timestamp", 0)
            description = insight.get("description", "")
            extracted_text = insight.get("extracted_text", [])
            
            content = f"[Visual at {insight.get('timestamp_formatted', '')}]\n"
            content += f"Type: {content_type}\n"
            content += f"Description: {description}\n"
            if extracted_text:
                if isinstance(extracted_text, list):
                    content += f"Text on screen: {' '.join(extracted_text)}"
                else:
                    content += f"Text on screen: {extracted_text}"
            
            doc = Document(
                page_content=content,
                metadata={
                    "source_type": "visual",
                    "timestamp": timestamp,
                    "content_type": content_type
                }
            )
            documents.append(doc)
        
        return documents
    
    def _get_data_hash(self, data: Dict[str, Any]) -> str:
        """Generate hash of meeting data to detect changes."""
        content = json.dumps(data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _load_or_build_vectorstore(self) -> Chroma:
        """Load existing vectorstore or build new one."""
        data = self._load_meeting_data()
        current_hash = self._get_data_hash(data)
        hash_file = self.vectorstore_path / "data_hash.txt"
        
        should_rebuild = True
        if self.vectorstore_path.exists() and hash_file.exists():
            stored_hash = hash_file.read_text().strip()
            if stored_hash == current_hash:
                should_rebuild = False
        
        if should_rebuild:
            print(f"Building vectorstore for meeting {self.meeting_id}...")
            documents = self._create_documents(data)
            
            self.vectorstore_path.mkdir(parents=True, exist_ok=True)
            
            if not documents:
                vectorstore = Chroma(
                    collection_name=f"meeting_{self.meeting_id[:8]}",
                    embedding_function=self.embeddings,
                    persist_directory=str(self.vectorstore_path)
                )
            else:
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=f"meeting_{self.meeting_id[:8]}",
                    persist_directory=str(self.vectorstore_path)
                )
            
            hash_file.write_text(current_hash)
            print(f"Indexed {len(documents)} documents")
            return vectorstore
        else:
            print(f"Loading existing vectorstore for meeting {self.meeting_id}")
            return Chroma(
                collection_name=f"meeting_{self.meeting_id[:8]}",
                embedding_function=self.embeddings,
                persist_directory=str(self.vectorstore_path)
            )
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents for context."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _format_history(self) -> str:
        """Format chat history for context."""
        if not self.chat_history:
            return "No previous conversation."
        
        lines = []
        for msg in self.chat_history[-self.max_history:]:
            if isinstance(msg, HumanMessage):
                lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Assistant: {msg.content}")
        return "\n".join(lines)
    
    def _build_chain(self):
        """Build the RAG chain using LCEL."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant answering questions about a meeting.
Use the context from the meeting transcript, summary, and visual content to answer.
If you don't know the answer, say so - don't make things up.
Cite specific timestamps when possible (format: X:XX).

CONTEXT:
{context}

PREVIOUS CONVERSATION:
{history}"""),
            ("human", "{question}"),
        ])
        
        chain = (
            {
                "context": self.retriever | self._format_docs,
                "history": lambda _: self._format_history(),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question and get response with sources."""
        # Get relevant documents
        docs = self.retriever.invoke(question)
        
        # Run chain
        answer = self.chain.invoke(question)
        
        # Update history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        
        # Keep history bounded
        if len(self.chat_history) > self.max_history * 2:
            self.chat_history = self.chat_history[-self.max_history * 2:]
        
        # Format response
        response = {
            "answer": answer,
            "sources": []
        }
        
        for doc in docs:
            source = {
                "type": doc.metadata.get("source_type", "unknown"),
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            
            if doc.metadata.get("start_timestamp"):
                source["timestamp"] = doc.metadata["start_timestamp"]
            elif doc.metadata.get("timestamp"):
                source["timestamp"] = doc.metadata["timestamp"]
            
            if doc.metadata.get("section"):
                source["section"] = doc.metadata["section"]
            
            response["sources"].append(source)
        
        return response
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.chat_history = []


# Cache for meeting RAG instances
_rag_cache: Dict[str, MeetingRAG] = {}


def get_meeting_rag(meeting_id: str, data_root: str = "src/data/meetings") -> MeetingRAG:
    """Get or create RAG instance for a meeting."""
    cache_key = f"{meeting_id}:{data_root}"
    
    if cache_key not in _rag_cache:
        _rag_cache[cache_key] = MeetingRAG(meeting_id, data_root)
    
    return _rag_cache[cache_key]


def clear_rag_cache(meeting_id: Optional[str] = None):
    """Clear RAG cache for a specific meeting or all meetings."""
    global _rag_cache
    
    if meeting_id:
        keys_to_remove = [k for k in _rag_cache if k.startswith(meeting_id)]
        for key in keys_to_remove:
            del _rag_cache[key]
    else:
        _rag_cache = {}
