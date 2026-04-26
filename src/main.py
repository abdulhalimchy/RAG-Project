import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

def load_data(path):
    try:
        df = pd.read_csv(path, encoding="latin1")
        print("✅ Data loaded successfully!\n")
        return df
    except Exception as e:
        print("❌ Error loading data:", e)
        return None


def explore_data(df):
    print("🔹 First 5 rows:")
    print(df.head(), "\n")

    print("🔹 Columns:")
    for col in df.columns:
        print(f"- {col}")
    print()

    print("🔹 Data Info:")
    print(df.info(), "\n")


def create_transaction_documents(df):
    documents = []

    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Year"] = df["Order Date"].dt.year

    for _, row in df.iterrows():
        text = f"""
        Order {row['Order ID']} was placed on {row['Order Date'].date()} by {row['Customer Name']}
        from the {row['Segment']} segment in {row['City']}, {row['State']} ({row['Region']} region).
        The customer purchased {row['Product Name']} from the {row['Category']} category,
        sub-category {row['Sub-Category']}.
        Sales were {row['Sales']:.2f}, quantity was {row['Quantity']},
        discount was {row['Discount']:.2f}, and profit was {row['Profit']:.2f}.
        """

        documents.append(Document(
            page_content=text.strip(),
            metadata={
                "type": "transaction",
                "year": int(row["Year"]),
                "region": row["Region"],
                "category": row["Category"],
                "sub_category": row["Sub-Category"]
            }
        ))

    return documents

def create_summary_documents(df):
    documents = []

    df = df.copy()

    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Year"] = df["Order Date"].dt.year
    df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)

    def add_doc(text, metadata):
        documents.append(Document(
            page_content=text.strip(),
            metadata=metadata
        ))

    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    total_quantity = df["Quantity"].sum()
    avg_discount = df["Discount"].mean()
    profit_margin = (total_profit / total_sales) * 100

    add_doc(f"""
    Overall, the dataset contains {len(df)} transactions.
    Total sales were {total_sales:.2f}, total profit was {total_profit:.2f},
    total quantity sold was {total_quantity}, average discount was {avg_discount:.2f},
    and overall profit margin was {profit_margin:.2f}%.
    """, {
        "type": "overall_summary"
    })

    yearly = df.groupby("Year").agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Discount=("Discount", "mean")
    ).reset_index()

    for _, row in yearly.iterrows():
        margin = (row["Profit"] / row["Sales"]) * 100
        add_doc(f"""
        In {int(row['Year'])}, total sales were {row['Sales']:.2f},
        total profit was {row['Profit']:.2f}, total quantity sold was {int(row['Quantity'])},
        average discount was {row['Discount']:.2f}, and profit margin was {margin:.2f}%.
        """, {
            "type": "yearly_summary",
            "year": int(row["Year"])
        })

    monthly = df.groupby("YearMonth").agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Discount=("Discount", "mean")
    ).reset_index()

    for _, row in monthly.iterrows():
        margin = (row["Profit"] / row["Sales"]) * 100
        add_doc(f"""
        In {row['YearMonth']}, total sales were {row['Sales']:.2f},
        total profit was {row['Profit']:.2f}, total quantity sold was {int(row['Quantity'])},
        average discount was {row['Discount']:.2f}, and profit margin was {margin:.2f}%.
        """, {
            "type": "monthly_summary",
            "year_month": row["YearMonth"]
        })

    category = df.groupby("Category").agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Discount=("Discount", "mean")
    ).reset_index()

    for _, row in category.iterrows():
        margin = (row["Profit"] / row["Sales"]) * 100
        add_doc(f"""
        The {row['Category']} category generated total sales of {row['Sales']:.2f},
        total profit of {row['Profit']:.2f}, total quantity sold of {int(row['Quantity'])},
        average discount of {row['Discount']:.2f}, and profit margin of {margin:.2f}%.
        """, {
            "type": "category_summary",
            "category": row["Category"]
        })

    category_year = df.groupby(["Year", "Category"]).agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Discount=("Discount", "mean")
    ).reset_index()

    for _, row in category_year.iterrows():
        margin = (row["Profit"] / row["Sales"]) * 100
        add_doc(f"""
        In {int(row['Year'])}, the {row['Category']} category had sales of {row['Sales']:.2f},
        profit of {row['Profit']:.2f}, quantity sold of {int(row['Quantity'])},
        average discount of {row['Discount']:.2f}, and profit margin of {margin:.2f}%.
        """, {
            "type": "category_year_summary",
            "year": int(row["Year"]),
            "category": row["Category"]
        })

    sub_category = df.groupby("Sub-Category").agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Discount=("Discount", "mean")
    ).reset_index()

    for _, row in sub_category.iterrows():
        margin = (row["Profit"] / row["Sales"]) * 100
        add_doc(f"""
        The {row['Sub-Category']} sub-category generated sales of {row['Sales']:.2f},
        profit of {row['Profit']:.2f}, quantity sold of {int(row['Quantity'])},
        average discount of {row['Discount']:.2f}, and profit margin of {margin:.2f}%.
        """, {
            "type": "subcategory_summary",
            "sub_category": row["Sub-Category"]
        })

    region = df.groupby("Region").agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Discount=("Discount", "mean")
    ).reset_index()

    for _, row in region.iterrows():
        margin = (row["Profit"] / row["Sales"]) * 100
        add_doc(f"""
        The {row['Region']} region achieved total sales of {row['Sales']:.2f},
        total profit of {row['Profit']:.2f}, total quantity sold of {int(row['Quantity'])},
        average discount of {row['Discount']:.2f}, and profit margin of {margin:.2f}%.
        """, {
            "type": "region_summary",
            "region": row["Region"]
        })

    region_year = df.groupby(["Year", "Region"]).agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Discount=("Discount", "mean")
    ).reset_index()

    for _, row in region_year.iterrows():
        margin = (row["Profit"] / row["Sales"]) * 100
        add_doc(f"""
        In {int(row['Year'])}, the {row['Region']} region had sales of {row['Sales']:.2f},
        profit of {row['Profit']:.2f}, quantity sold of {int(row['Quantity'])},
        average discount of {row['Discount']:.2f}, and profit margin of {margin:.2f}%.
        """, {
            "type": "region_year_summary",
            "year": int(row["Year"]),
            "region": row["Region"]
        })

    state = df.groupby("State").agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    ).reset_index()

    top_states = state.sort_values("Sales", ascending=False).head(10)

    for _, row in top_states.iterrows():
        add_doc(f"""
        The state of {row['State']} is one of the top states by sales.
        Total sales were {row['Sales']:.2f}, total profit was {row['Profit']:.2f},
        and total quantity sold was {int(row['Quantity'])}.
        """, {
            "type": "top_state_summary",
            "state": row["State"]
        })

    city = df.groupby("City").agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    ).reset_index()

    top_cities = city.sort_values("Sales", ascending=False).head(10)

    for _, row in top_cities.iterrows():
        add_doc(f"""
        The city of {row['City']} is one of the top cities by sales.
        Total sales were {row['Sales']:.2f}, total profit was {row['Profit']:.2f},
        and total quantity sold was {int(row['Quantity'])}.
        """, {
            "type": "top_city_summary",
            "city": row["City"]
        })

    top_category_sales = category.sort_values("Sales", ascending=False).iloc[0]
    top_category_profit = category.sort_values("Profit", ascending=False).iloc[0]
    top_region_sales = region.sort_values("Sales", ascending=False).iloc[0]
    top_region_profit = region.sort_values("Profit", ascending=False).iloc[0]
    top_subcat_profit = sub_category.sort_values("Profit", ascending=False).iloc[0]

    add_doc(f"""
    The category with the highest total sales was {top_category_sales['Category']}
    with sales of {top_category_sales['Sales']:.2f}.
    The category with the highest total profit was {top_category_profit['Category']}
    with profit of {top_category_profit['Profit']:.2f}.
    """, {
        "type": "top_performer_summary"
    })

    add_doc(f"""
    The region with the highest total sales was {top_region_sales['Region']}
    with sales of {top_region_sales['Sales']:.2f}.
    The region with the highest total profit was {top_region_profit['Region']}
    with profit of {top_region_profit['Profit']:.2f}.
    """, {
        "type": "top_performer_summary"
    })

    add_doc(f"""
    The sub-category with the highest total profit was {top_subcat_profit['Sub-Category']}
    with profit of {top_subcat_profit['Profit']:.2f}.
    """, {
        "type": "top_performer_summary"
    })

    high_discount = df[df["Discount"] >= 0.30]
    low_discount = df[df["Discount"] < 0.30]

    add_doc(f"""
    Transactions with high discounts of 30% or more had total sales of {high_discount['Sales'].sum():.2f}
    and total profit of {high_discount['Profit'].sum():.2f}.
    Transactions with discounts below 30% had total sales of {low_discount['Sales'].sum():.2f}
    and total profit of {low_discount['Profit'].sum():.2f}.
    This summary helps analyze how discount levels relate to profitability.
    """, {
        "type": "discount_profit_summary"
    })

    return documents

def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    return chunks

def create_embeddings(chunks):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [chunk.page_content for chunk in chunks]

    embeddings = model.encode(texts)

    return embeddings, model

## Store chunks and embeddings in ChromaDB
def store_in_chromadb(chunks, embeddings):
    print("🔄 Initializing ChromaDB...")

    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        client.delete_collection("sales_rag")
    except Exception:
        pass

    collection = client.get_or_create_collection(name="sales_rag")

    print("🔄 Storing documents in ChromaDB...")

    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"id_{i}" for i in range(len(chunks))]
    embeddings_list = embeddings.tolist()

    batch_size = 1000

    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=documents[i:i + batch_size],
            embeddings=embeddings_list[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )

    print("✅ Stored in ChromaDB")

    return collection

## Get or create ChromaDB collection (with option to rebuild)
def get_or_create_chromadb(chunks, embeddings, rebuild=False):
    client = chromadb.PersistentClient(path="./chroma_db")

    if rebuild:
        try:
            client.delete_collection("sales_rag")
        except Exception:
            pass

    collection = client.get_or_create_collection(name="sales_rag")

    if collection.count() > 0 and not rebuild:
        print(f"✅ Loaded existing ChromaDB with {collection.count()} documents")
        return collection

    print("🔄 Storing documents in ChromaDB...")

    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"id_{i}" for i in range(len(chunks))]
    embeddings_list = embeddings.tolist()

    batch_size = 1000

    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=documents[i:i + batch_size],
            embeddings=embeddings_list[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )

    print(f"✅ Stored {collection.count()} documents in ChromaDB")
    return collection

def test_query(collection, model):
    query = "Which region has the best sales performance?"

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    print("\n🔍 Query:", query)
    print("\n🔹 Retrieved Results:")

    for doc in results["documents"][0]:
        print("-", doc[:200])

def ask_rag_question(collection, embedding_model, question):
    query_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=10
    )

    retrieved_docs = results["documents"][0]
    context = "\n\n".join(retrieved_docs)
    print("###CCONTEXT:  ")
    print(context)
    print()
    prompt = f"""
    You are a sales data analyst.

    Use ONLY the provided context to answer the question.
    For questions about "best sales performance", interpret performance mainly using total sales.
    If profit is also available, mention it as supporting evidence.

    Do not say the data is insufficient if the context contains direct sales or profit comparisons.

    Context:
    {context}

    Question:
    {question}

    Answer clearly and briefly:
    """

    llm = OllamaLLM(model="llama3.2:3b")
    response = llm.invoke(prompt)

    return response

def main():
    path = "data/Sample - Superstore.csv"

    df = load_data(path)

    if df is not None:
        explore_data(df)

        transaction_documents = create_transaction_documents(df)
        summary_documents = create_summary_documents(df)

        all_documents = transaction_documents + summary_documents

        chunks = chunk_documents(all_documents)

        embeddings, model = create_embeddings(chunks)

        collection = get_or_create_chromadb(chunks, embeddings, rebuild=False)

        # print(f"✅ Transaction docs: {len(transaction_documents)}")
        # print(f"✅ Summary docs: {len(summary_documents)}\n")

        # print(f"✅ Total documents before chunking: {len(all_documents)}")
        # print(f"✅ Total chunks after chunking: {len(chunks)}\n")

        # # 🔹 Sample transaction document
        # print("🔹 Sample TRANSACTION document:")
        # print(transaction_documents[0].page_content)
        # print("Metadata:", transaction_documents[0].metadata)
        # print()

        # # 🔹 Sample summary document
        # print("🔹 Sample SUMMARY document:")
        # print(summary_documents[0].page_content)
        # print("Metadata:", summary_documents[0].metadata)
        # print()

        # # 🔹 Sample chunk
        # print("🔹 Sample CHUNK:")
        # print(chunks[0].page_content)
        # print("Metadata:", chunks[0].metadata)
        # print()

        question = "Which region has the best sales performance?"

        answer = ask_rag_question(collection, model, question)

        print("\n❓ Question:")
        print(question)

        print("\n🤖 RAG Answer:")
        print(answer)

if __name__ == "__main__":
    main()