import sys
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

# Create client with API key
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Create LLM - adapter is auto-detected for google provider
llm = llm_factory(
    "gemini-3.5-flash",
    provider="google",
    client=client
)

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware import ToolCallLimitMiddleware


from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._factual_correctness import FactualCorrectness



# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------


CHUNK_SIZE = 256


def load_codebase(repo_path: str) -> list:
   docs = []
   for path in Path(repo_path).rglob("*.py"):
       text = path.read_text(encoding="utf-8", errors="ignore")
       docs.append(Document(page_content=text, metadata={"source": str(path)}))
   return docs



# ---------------------------------------------------------------------------
# 2. CHUNK
# ---------------------------------------------------------------------------


def chunk_code(docs: list) -> list:
   splitter = RecursiveCharacterTextSplitter.from_language(
       language=Language.PYTHON,
       chunk_size=CHUNK_SIZE,
       chunk_overlap=32,
   )
   return splitter.split_documents(docs)

# ---------------------------------------------------------------------------
# 3. EMBED & STORE
# ---------------------------------------------------------------------------


def build_vector_store(chunks: list) -> Chroma:
   embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
   return Chroma.from_documents(chunks, embedding=embeddings)



# ---------------------------------------------------------------------------
# 4. RETRIEVER + AGENT  (identical to 1_semanticrag.py)
# ---------------------------------------------------------------------------


def build_agent(vector_store: Chroma):
   retriever_tool = create_retriever_tool(
       vector_store.as_retriever(search_kwargs={"k": 4}),
       name="search_codebase",
       description="Search the codebase for relevant functions, classes, or logic.",
   )
   return create_agent(
       model="google_genai:gemini-3.5-flash-lite", tools=[retriever_tool],
       system_prompt=(
           "You are a senior engineer. Always use search_codebase before answering. "
           "Reference specific file and function names. "
           "If not found say 'I could not find that in the codebase'."
       ),
       middleware=[
           ModelCallLimitMiddleware(run_limit=50, exit_behavior="end"),
           ToolCallLimitMiddleware(tool_name="retriever_tool", run_limit=20, exit_behavior="end")
       ]
   )



def content_to_text(content) -> str:
   if isinstance(content, str):
      return content

   if isinstance(content, list):
      text_parts = []
      for block in content:
         if isinstance(block, str):
            text_parts.append(block)
         elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
               text_parts.append(text)
         else:
            text = getattr(block, "text", None)
            if isinstance(text, str):
               text_parts.append(text)
      return "".join(text_parts)

   return str(content)


def run_agent(agent, question: str) -> tuple[str, list[str]]:
   """Invoke the agent and extract (answer, contexts) from the message trace."""
   result = agent.invoke({"messages": [{"role": "user", "content": question}]})
   answer = content_to_text(result["messages"][-1].content)


   contexts = []
   for message in result["messages"]:
      if type(message).__name__ == "ToolMessage":
         contexts.append(content_to_text(message.content))


   return answer, contexts


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
# Golden Dataset - Ground-truth test set (questions about sample_project)
# ---------------------------------------------------------------------------


from golden_dataset import TEST_CASES


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
