# from src.ingest.argo_loader import load_all_years
# from src.warehouse.duck import DuckDBWarehouse

# # Load all NetCDF files into one DataFrame
# df = load_all_years("./data/raw")

# print(df.head())  # check if 'time' and 'year' exist now

# db = DuckDBWarehouse()
# db.register_dataframe(df, "profiles")

# # # Count rows per year
# print(db.query("SELECT year, COUNT(*) FROM profiles GROUP BY year ORDER BY year"))

# # Query by time/region
# result = db.query("""
#     SELECT time, latitude, longitude, temp, psal
#     FROM profiles
#     WHERE year=2016
#       AND latitude BETWEEN 20 AND 30
#       AND longitude BETWEEN -90 AND -70
#     LIMIT 10
# """)
# print(result.head())

# db.close()








from src.ingest.argo_loader import load_all_years
from src.vectorstore.chroma_store import ChromaVectorStore

# Step 1: Load data (already from step 2)
df = load_all_years("./data/raw")

# Step 2: Build vector store
store = ChromaVectorStore(path="./data/chroma")
store.add_dataframe(df)

# Step 3: Test query
question = "Show me temperature trends in 2016"
results = store.query(question)

print("🔍 Query Results:")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"- {doc} | Meta: {meta}")





# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("./models/all-MiniLM-L6-v2")


# from sentence_transformers import SentenceTransformer

# # This will fetch all weights + config files
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# model.save("./models/all-MiniLM-L6-v2")


# from src.vectorstore.chroma_store import get_chroma_retriever
# from src.rag.llms import get_openai_llm, get_local_llm
# from src.rag.pipeline import build_rag_pipeline

# # 1. Get retriever
# retriever = get_chroma_retriever(path="./data/chroma")

# # 2. Choose LLM
# llm = get_openai_llm()       # requires OPENAI_API_KEY in env
# # llm = get_local_llm("mistral")  # if you use Ollama locally

# # 3. Build pipeline
# rag = build_rag_pipeline(retriever, llm)

# # 4. Ask a question
# query = "Show me temperature and salinity profiles in the Indian Ocean in 2017"
# res = rag.invoke({"query": query})

# print("🔍 Answer:", res["result"])
# print("📂 Sources:")
# for doc in res["source_documents"]:
#     print(f"- {doc.metadata}")





