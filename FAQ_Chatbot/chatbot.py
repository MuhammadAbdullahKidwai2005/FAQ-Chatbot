from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

system_prompt = (
    "You are an expert customer support document analyst.\n\n"
    "You have access to company documents including PDFs, emails, and chat logs.\n\n"
    "STRICT RULES:\n"
    "1. Read ALL the provided context carefully.\n"
    "2. If asked about emails — list EVERY email subject you find in the context, numbered.\n"
    "3. If asked to count something — count exactly what appears in the context.\n"
    "4. If asked to list something — list every item found, numbered.\n"
    "5. Be specific: extract names, subjects, dates, numbers directly from context.\n"
    "6. If partial info exists — share it and say it may be incomplete.\n"
    "7. NEVER say not available if the context has anything even slightly related.\n"
    "8. Only say 'not available' if context has zero relevance to the question.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

EMAIL_KEYWORDS = [
    "email", "emails", "subject", "subjects", "mail",
    "received", "inbox", "message", "messages", "how many emails"
]

def build_qa_chain(vectorstore, local_llm, all_chunks=None):

    def smart_retriever(query):
        query_lower = query.lower()
        is_email_query = any(kw in query_lower for kw in EMAIL_KEYWORDS)

        if is_email_query and all_chunks:
            # Directly filter ALL chunks that came from emails.txt — no similarity search
            email_chunks = [
                c for c in all_chunks
                if "emails.txt" in c.metadata.get("source", "")
            ]
            # Also get semantic results for extra context
            semantic_docs = vectorstore.similarity_search(query, k=3)

            # Merge and deduplicate
            seen = set()
            combined = []
            for doc in email_chunks + semantic_docs:
                key = doc.page_content[:80]
                if key not in seen:
                    seen.add(key)
                    combined.append(doc)
            return combined

        return vectorstore.similarity_search(query, k=5)

    qa_chain = (
        {"context": lambda x: format_docs(smart_retriever(x)), "input": RunnablePassthrough()}
        | prompt
        | local_llm
        | StrOutputParser()
    )

    print("Manual LCEL QA chain created successfully without using langchain.chains")
    return qa_chain