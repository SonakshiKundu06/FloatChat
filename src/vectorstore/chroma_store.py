import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class ChromaVectorStore:
    def __init__(self, path="./data/chroma", collection_name="argo_profiles"):
        # Connect to local Chroma
        self.client = chromadb.PersistentClient(path=path, settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Load embedding model
        model_name = os.getenv("EMBED_MODEL", "./models/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name)

    def build_profile_text(self, df, source_file: str) -> str:
        """
        Summarize one NetCDF profile (many depth levels) into a single text document.
        Example: "On 2016-02-08 float 0042682 at 24.6N, -81.8W recorded 72 depth levels
                  with temperature range 2.1–28.3°C and salinity range 33.5–36.7 PSU."
        """
        time = df["time"].iloc[0] if "time" in df else "unknown time"
        float_id = df["platform_number"].iloc[0] if "platform_number" in df else "unknown float"
        lat = df["latitude"].iloc[0] if "latitude" in df else 0
        lon = df["longitude"].iloc[0] if "longitude" in df else 0

        temp_min, temp_max = df["temp"].min(), df["temp"].max()
        psal_min, psal_max = df["psal"].min(), df["psal"].max()

        text = (f"On {time}, float {float_id} at {lat:.2f}N, {lon:.2f}W "
                f"recorded {len(df)} depth levels "
                f"with temperature range {temp_min:.2f}–{temp_max:.2f} °C "
                f"and salinity range {psal_min:.2f}–{psal_max:.2f} PSU.")

        return text

    def add_dataframe(self, df, batch_size=200):
        """
        Add grouped profiles instead of individual rows → more efficient!
        """
        docs, ids, metas = [], [], []

        # Group by (platform, cycle, file) so each profile is one document
        grouped = df.groupby(["platform_number", "cycle_number", "source_file"])

        for idx, (keys, group) in enumerate(grouped):
            text = self.build_profile_text(group, keys[2])
            docs.append(text)
            ids.append(f"{keys[0]}_{keys[1]}_{idx}")  # unique ID
            metas.append({"year": int(group["year"].iloc[0]) if "year" in group else -1,
                          "file": keys[2]})

            if len(docs) >= batch_size:
                embeddings = self.model.encode(docs, show_progress_bar=False).tolist()
                self.collection.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)
                docs, ids, metas = [], [], []

        # Leftovers
        if docs:
            embeddings = self.model.encode(docs, show_progress_bar=False).tolist()
            self.collection.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)

    def query(self, question: str, n_results=5):
        """ Query the Chroma vector store with a natural language question """
        embedding = self.model.encode([question]).tolist()
        results = self.collection.query(query_embeddings=embedding, n_results=n_results)
        return results

# ✅ Retriever for RAG pipeline
def get_chroma_retriever(path="./data/chroma", collection_name="argo_profiles", k=5):
    embeddings = HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2")
    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=path,
        embedding_function=embeddings
    )
    return vectordb.as_retriever(search_kwargs={"k": k})
