import sys
import ast
import types
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os
from google import genai


# ragas 0.4.x imports ChatVertexAI from langchain_community.chat_models.vertexai,
# which was removed in langchain-community 0.4. Inject a shim before ragas loads.
if "langchain_community.chat_models.vertexai" not in sys.modules:
   from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
   _shim = types.ModuleType("langchain_community.chat_models.vertexai")
   _shim.ChatVertexAI = _ChatVertexAI
   sys.modules["langchain_community.chat_models.vertexai"] = _shim
from ragas.llms import llm_factory


load_dotenv()

ssl_cert_file = os.getenv("SSL_CERT_FILE")
if ssl_cert_file and not Path(ssl_cert_file).is_file():
   os.environ.pop("SSL_CERT_FILE", None)

ssl_cert_dir = os.getenv("SSL_CERT_DIR")
if ssl_cert_dir and not Path(ssl_cert_dir).is_dir():
   os.environ.pop("SSL_CERT_DIR", None)

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent

from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.llms import llm_factory




# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------

def load_codebase(repo_path: str) -> list:
   docs =[]
   for path in Path(repo_path).rglob("*.py"):
      text = path.read_text(encoding="utf-8",errors="ignore")
      docs.append(Document(page_content=text,metadata={"source":str(path)}))
   return docs

# ---------------------------------------------------------------------------
# 2. CHUNK — AST-based (one chunk per class or module-level function)
# ---------------------------------------------------------------------------

def chunk_code(docs:list)->list:
   chunks = []
   for doc in docs:
      source = doc.metadata.get("source", "unknown")
      code = doc.page_content

      try:
         tree = ast.parse(code)
      except SyntaxError:
         chunks.append(doc)
         continue

      for node in ast.walk(tree):
         for child in ast.iter_child_nodes(node):
            child.parent = node

      for node in ast.walk(tree):
         parent = getattr(node, "parent", None)

         if isinstance(node, ast.ClassDef) and isinstance(parent, ast.Module):
            text = ast.get_source_segment(code, node)
            if text:
               chunks.append(Document(
                  page_content=text,
                  metadata ={"source":source,"type":"class","name":node.name},
               ))


            elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and isinstance(parent, ast.Module):
               text = ast.get_source_segment(code,node)
               if text:
                  chunks.append(Document(page_content=text,
                                         metadata={"source":source,"type":"funtion","name":node.name}))

   return chunks


# ---------------------------------------------------------------------------
# 3. EMBED & STORE
# ---------------------------------------------------------------------------
def build_vector_store(chunks:list)->Chroma:
   emeddings = HuggingFaceEmbeddings(model = "all-MiniLM-L6-v2")
   return Chroma.from_documents(chunks,embedding=emeddings)

# ---------------------------------------------------------------------------
# 4. RETRIEVER + AGENT  (identical to 3_semanticrag_ast.py)
# ---------------------------------------------------------------------------

def build_agent(vector_store: Chroma):
   retriver_tool = create_retriever_tool(
      vector_store.as_retriever(search_kwargs= {"k":1}),
      name = "search_codebase",
      description="Search the codebase for relevant funtions, classes, or logic"
   )
   return create_agent(
      model= "google_genai:gemini-3.5-flash-lite",
      tools=[retriver_tool],
      system_prompt=(
         "You are the senior engineed, Always use search_codebase before answering."
         "Reference specific file and funtion names."
         "If not found say 'I could not find that in the codebase'."
      )
   )


def run_agent(agent,question:str)->tuple[str, list[str]]:
   result = agent.invoke({"messages":[{"role":"user","context":question}]})
   answer = result["messages"][-1].content

   contexts = []
   for msg in result["messages"]:
      if type(msg).__name__ == "ToolMessage" and isinstance(msg.content,str):
         contexts.append(msg.content)

   return answer,contexts


# ---------------------------------------------------------------------------
# Golden Dataset
# ---------------------------------------------------------------------------

from golden_dataset import TEST_CASES


def save_rag_results(
   output_path: str | Path,
   repo_path: str,
   file_count: int,
   chunk_count: int,
   rag_outputs: list[dict],
   score_df,
   metric_cols: list[str],
) -> Path:
   """Write RAG answers, retrieved contexts, and evaluation scores to a text file."""
   output_file = Path(output_path)
   output_file.parent.mkdir(parents=True, exist_ok=True)

   lines = [
      "RAG EVALUATION RESULTS",
      "=" * 80,
      f"Repository: {repo_path}",
      f"Files loaded: {file_count}",
      f"Chunks created: {chunk_count} (chunk_size={CHUNK_SIZE})",
      "",
      "RAG OUTPUTS",
      "=" * 80,
   ]

   for index, item in enumerate(rag_outputs, 1):
      lines.extend([
         "",
         f"[{index}] QUESTION",
         "-" * 80,
         item["question"],
         "",
         f"RETRIEVED CONTEXTS ({len(item['contexts'])})",
      ])
      for context_index, context in enumerate(item["contexts"], 1):
         lines.extend([f"\nContext {context_index}:", context])
      lines.extend(["", "ANSWER", "-" * 80, item["answer"]])

   lines.extend(["", "RAGAS SCORECARD", "=" * 80])
   for _, row in score_df.iterrows():
      lines.extend(["", f"Question: {row['user_input']}", "-" * 80])
      for metric in metric_cols:
         if metric in row and row[metric] == row[metric]:
            lines.append(f"{metric:<45}: {row[metric]:.3f}")

   lines.extend(["", "AGGREGATE AVERAGES", "=" * 80])
   for metric in metric_cols:
      if metric in score_df.columns:
         average = score_df[metric].mean()
         if average == average:
            lines.append(f"{metric:<45}: {average:.3f}")

   output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
   return output_file


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument(
      "--repo",
      default=str(Path(__file__).parent.parent / "sample_project"),
   )
   parser.add_argument(
      "--output",
      default=str(Path(__file__).with_name(f"{Path(__file__).stem}_results.txt")),
      help="Text file where RAG answers and evaluation scores are saved.",
   )
   args = parser.parse_args()
   repo_path = str(Path(args.repo).resolve())


   docs = load_codebase(repo_path)
   chunks = chunk_code(docs)
   print(f"Loaded {len(docs)} files -> {len(chunks)} chunks (chunk_size={CHUNK_SIZE})")


   vector_store = build_vector_store(chunks)
   agent = build_agent(vector_store)


 # Collect RAG outputs
   print("\n Running agent on test cases...")
   samples = []
   rag_outputs = []
   for tc in TEST_CASES:
      answer, contexts = run_agent(agent,tc["question"])
      rag_outputs.append({
         "question": tc["question"],
         "answer": answer,
         "contexts": contexts,
      })
      samples.append(SingleTurnSample(
         user_input = tc["question"],
         response=answer,
         retrieved_contexts= contexts,
         reference= tc["reference"],
      ))

      print(f"\nQ: {tc['question']}")
      print(f"Contexts retrieved ({len(contexts)}): ")
      for i, ctx in enumerate(contexts, 1):
         print(f"[{i}] {ctx[:200]} {'...' if len(ctx)>200 else ''}")
      print(f"Answer: {answer}")

   #RAGAS Evalution
   print("\n Running RAGAS evalution (GPT-4o as judge)...")

   lc_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
   eval_dataset = EvaluationDataset(samples=samples)

   # Create client with API key
   client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Create LLM - adapter is auto-detected for google provider
   llm = llm_factory(
        "gemini-3.5-flash",
        provider="google",
        client=client
    )

   results = evaluate(
      dataset = eval_dataset,
      metrics = [
         LLMContextPrecisionWithReference(),
         LLMContextRecall(),
         Faithfulness(),
         AnswerRelevancy(embeddings = lc_embeddings),
         FactualCorrectness(),
      ],
      llm=llm
   )

   # Scorecard
   df = results.to_pandas()
   metric_cols =[
      "llm_context_precisiion_with_reference",
      "context_recall",
      "faithfulness",
      "answer_relevancy",
      "factual_correctness",
   ]


   print("\n RAGAS SCORECARD = Semantic RAG on sample_project")
   print("\n Per-question breakdown: ")
   for _, row in df.iterrows():
         print(f"\n  Q: {row['user_input'][:65]}")
         for col in metric_cols:
            if col in row:
                  val = row[col]
                  icon = "✅" if val >= 0.7 else ("⚠️ " if val >= 0.5 else "❌")
                  print(f"    {icon} {col:<42}: {val:.3f}")


   print("\nAggregate averages:")
   for col in metric_cols:
      if col in df.columns:
         avg = df[col].mean()
         icon = "✅" if avg >= 0.7 else ("⚠️ " if avg >= 0.5 else "❌")
         print(f"  {icon} {col:<42}: {avg:.3f}")
   output_file = save_rag_results(
      output_path=args.output,
      repo_path=repo_path,
      file_count=len(docs),
      chunk_count=len(chunks),
      rag_outputs=rag_outputs,
      score_df=df,
      metric_cols=metric_cols,
   )
   print(f"\nSaved formatted results to: {output_file.resolve()}")
   print()
