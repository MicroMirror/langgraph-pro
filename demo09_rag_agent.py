import os

from pathlib import Path

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.tools import tool

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent


# ============================================================
# 1. Environment
# ============================================================

load_dotenv()


# ============================================================
# 2. Load Documents
# ============================================================

def load_documents():

    documents = []

    docs_dir = Path("docs")

    for file in docs_dir.glob("*.md"):

        text = file.read_text(
            encoding="utf-8"
        )

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file.name
                },
            )
        )

    return documents


documents = load_documents()

print(
    f"Loaded {len(documents)} documents."
)


# ============================================================
# 3. Embedding
# ============================================================

print(
    "Loading embedding model..."
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)


# ============================================================
# 4. Vector Store
# ============================================================

print(
    "Building vector store..."
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="demo09_knowledge",
    persist_directory="./chroma_db",
)


# ============================================================
# 5. Retriever
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# ============================================================
# 6. RAG Tool
# ============================================================

@tool
def search_knowledge(
    query: str,
) -> str:
    """
    查询企业内部知识库。

    当用户的问题涉及：

    - 企业业务
    - 数智化
    - AI战略
    - 产品体系
    - 农业数字化

    时应该调用这个工具。

    query:
        要查询的问题。
    """

    documents = retriever.invoke(
        query
    )

    if not documents:

        return (
            "知识库中没有找到相关信息。"
        )

    results = []

    for i, doc in enumerate(
        documents,
        1,
    ):

        results.append(
            f"""
[文档 {i}]
来源：
{doc.metadata.get("source")}

内容：
{doc.page_content}
"""
        )

    return "\n".join(results)


# ============================================================
# 7. LLM
# ============================================================

llm = ChatOpenAI(
    model=os.getenv(
        "OPENAI_MODEL",
        "gpt-5.5",
    ),
    api_key=os.getenv(
        "OPENAI_API_KEY"
    ),
    base_url=os.getenv(
        "OPENAI_BASE_URL"
    ),
    temperature=0,
    use_responses_api=False,
)


# ============================================================
# 8. RAG Agent
# ============================================================

agent = create_react_agent(
    model=llm,
    tools=[
        search_knowledge
    ],
)


# ============================================================
# 9. Run
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("Demo09 - RAG Agent")
    print("=" * 60)

    while True:

        question = input(
            "\n请输入问题（输入 q 退出）："
        )

        if question.lower() == "q":

            break

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

        print()
        print("-" * 60)
        print("Agent Answer")
        print("-" * 60)

        print(
            result[
                "messages"
            ][-1].content
        )