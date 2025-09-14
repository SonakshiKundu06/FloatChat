from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def build_rag_pipeline(retriever, llm):
    """Builds a RetrievalQA chain using retriever + LLM with structured extraction of lat, lon, year."""

    template = """
    You are an intelligent assistant designed to answer questions about ARGO oceanographic data.  
    You have access to detailed observations including latitude, longitude, depth, time of measurement, temperature, and salinity.

    Your tasks:
    1. Answer clearly with temperature, salinity, and location details when available.
    If the answer is not in the context, say "I don’t know".
    2. Answer the user's question using the ARGO dataset context, **or respond helpfully** if the query is not about ARGO data.

    Guidelines:
    - Return a **JSON object** with fields `latitude`, `longitude`, `year`, and `answer`.
    - If the question is casual (e.g., greetings like "hi", "hello", "how are you") or unrelated to ARGO data, respond politely and helpfully in `answer`, keeping `latitude`, `longitude`, `year` as `null`.
    - For ARGO-related queries, answer clearly with temperature, salinity, and location details when available.
    - Cite specific data points or trends from the ARGO dataset if they exist.

    Context:
    {context}

    User Query:
    {question}

    Output Format (JSON):
    {{
      "latitude": <latitude or null>,
      "longitude": <longitude or null>,
      "year": <year or null>,
      "answer": "<your answer here>"
    }}


    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain
