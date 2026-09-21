import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware import ToolCallLimitMiddleware




CHUNK_SIZE = 256


# load code base from the folder
def load_codebase(repo_path: str) -> list:
    docs = []
    for path in Path(repo_path).rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": str(path)}))
    return docs

# chunk code
def chunk_code(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size = CHUNK_SIZE,
        chunk_overlap = 32,
    )
    return splitter.split_documents(docs)


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


# Build vector store
def build_vector_store(chunks: list) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embedding=embeddings)


# Build an agent
def build_agent(vector_sotre: Chroma):
    retriever_tool = create_retriever_tool(
        vector_sotre.as_retriever(search_kwargs={"k":1}),
        name="search_codebase",
        description="Search the codebase for relevant functions, classes, or logic.",
    )

    return create_agent(
        model="google_genai:gemini-3.5-flash-lite",
        tools=[retriever_tool],
        system_prompt=("Your are a senior engineer, Always use search_codebase before answering."
                       "Refrence specific file and function names."
                       "If not found say 'I could not find that in the codeabase'."),
        middleware=[
        ModelCallLimitMiddleware(run_limit = 5,exit_behavior = "end"),
        ToolCallLimitMiddleware(tool_name="search_codebase", run_limit=2, exit_behavior="end")
    ]
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).parent.parent / "sample_project"))
    args = parser.parse_args()
    repo_path = str(Path(args.repo).resolve())

    docs = load_codebase(repo_path)
    chunks = chunk_code(docs)
    print(f"Loaded {len(docs)} files -> {len(chunks)} chunks (chunk_size={CHUNK_SIZE})")



    vector_store = build_vector_store(chunks)
    agent = build_agent(vector_store)

    print("Ready. Ask your question. Type 'exit' to exit")

    while True:
        question = input("\n You: ").strip()
        if not question or question.lower() in ("exit", "quit"):
            break

        final_text = ""
        for step in agent.stream(
            {"messages":[{"role":"user","content":question}]},
            stream_mode="values"
        ):
            for msg in step.get("messages", []):
                if getattr(msg, "tool_calls", None):
                    continue
                text = format_agent_response(getattr(msg, "content", ""))
                if text:
                    final_text = text

        if final_text:
            print(f"Agent: {final_text}")

