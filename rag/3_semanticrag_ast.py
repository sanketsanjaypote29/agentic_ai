

import ast
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent





# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------


def load_codebase(repo_path: str) -> list:
   docs = []
   for path in Path(repo_path).rglob("*.py"):
       text = path.read_text(encoding="utf-8", errors="ignore")
       docs.append(Document(page_content=text, metadata={"source": str(path)}))
   return docs





# ---------------------------------------------------------------------------
# 2. CHUNK — AST-based (one chunk per class or module-level function)
# ---------------------------------------------------------------------------


def chunk_code(docs: list) -> list:
   chunks = []


   for doc in docs:
       source = doc.metadata.get("source", "unknown")
       code = doc.page_content


       try:
           tree = ast.parse(code)
       except SyntaxError:
           chunks.append(doc)
           continue


       # Add parent references — ast.walk doesn't expose parents by default
       for node in ast.walk(tree):
           for child in ast.iter_child_nodes(node):
               child.parent = node


       for node in ast.walk(tree):
           parent = getattr(node, "parent", None)


           # Top-level class → grab whole class including all methods
           if isinstance(node, ast.ClassDef) and isinstance(parent, ast.Module):
               text = ast.get_source_segment(code, node)
               if text:
                   chunks.append(Document(
                       page_content=text,
                       metadata={"source": source, "type": "class", "name": node.name},
                   ))


           # Module-level function — not a method inside a class
           elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(parent, ast.Module):
               text = ast.get_source_segment(code, node)
               if text:
                   chunks.append(Document(
                       page_content=text,
                       metadata={"source": source, "type": "function", "name": node.name},
                   ))


   return chunks


# ---------------------------------------------------------------------------
# 3. EMBED & STORE
# ---------------------------------------------------------------------------


def build_vector_store(chunks: list) -> Chroma:
   embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
   return Chroma.from_documents(chunks, embedding=embeddings)



# ---------------------------------------------------------------------------
# 4. RETRIEVER + AGENT
# ---------------------------------------------------------------------------


def build_agent(vector_store: Chroma):
   retriever_tool = create_retriever_tool(
       vector_store.as_retriever(search_kwargs={"k": 1}),
       name="search_codebase",
       description="Search the codebase for relevant functions, classes, or logic.",
   )
   return create_agent(
       model="google_genai:gemini-3.5-flash",
       tools=[retriever_tool],
       system_prompt=(
           "You are a senior engineer. Always use search_codebase before answering. "
           "Reference specific file and function names. "
           "If not found say 'I could not find that in the codebase'."
       ),
   )


def format_agent_response(content) -> str:
   if isinstance(content, str):
       return content

   if isinstance(content, list):
       text_parts = []
       for block in content:
           if isinstance(block, str):
               text_parts.append(block)
           elif isinstance(block, dict) and block.get("type") == "text":
               text_parts.append(block.get("text", ""))
       return "\n".join(part for part in text_parts if part)

   return str(content)




# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--repo", default=str(Path(__file__).parent.parent / "sample_project"))
   args = parser.parse_args()
   repo_path = str(Path(args.repo).resolve())


   docs = load_codebase(repo_path)
   chunks = chunk_code(docs)
   print(f"Loaded {len(docs)} files → {len(chunks)} chunks (AST-based)")


   vector_store = build_vector_store(chunks)
   agent = build_agent(vector_store)


   print("Ready. Ask your question. Type 'exit' to quit")
   while True:
       question = input("\nYou: ").strip()
       if not question or question.lower() in ("exit", "quit"):
           break


       for step in agent.stream(
           {"messages": [{"role": "user", "content": question}]},
           stream_mode="values",
       ):
           last_msg = step["messages"][-1]
           if not getattr(last_msg, "tool_calls", None):
               print(f"Agent: {format_agent_response(last_msg.content)}")
