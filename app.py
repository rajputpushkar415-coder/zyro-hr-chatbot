import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Zyro Dynamics HR Help Desk")
st.caption("Ask questions about Zyro Dynamics HR policies.")


@st.cache_resource
def build_rag_pipeline():
    DATA_PATH = "."

    pdf_files = sorted([
        os.path.join(DATA_PATH, file)
        for file in os.listdir(DATA_PATH)
        if file.endswith(".pdf")
    ])

    documents = []

    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = os.path.basename(pdf)

        documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.6
        }
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    def format_docs(docs):
        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        )

    prompt = ChatPromptTemplate.from_template("""
You are Zyro Dynamics HR Help Desk Assistant.

Answer the employee's question using ONLY the HR policy context provided below.

Rules:
1. If the answer is not present in the context, say: "I can only answer questions based on Zyro Dynamics HR policy documents."
2. Do not use outside knowledge.
3. Keep the answer clear and concise.
4. Mention the relevant policy/source name if available.
5. For non-HR or out-of-scope questions, politely refuse.

HR Policy Context:
{context}

Employee Question:
{question}

Final Answer:
""")

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


try:
    rag_chain = build_rag_pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_question = st.chat_input("Ask an HR policy question...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching HR policies..."):
                answer = rag_chain.invoke(user_question)
                st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

except Exception as e:
    st.error("App setup error. Please check API keys and document files.")
    st.exception(e)
