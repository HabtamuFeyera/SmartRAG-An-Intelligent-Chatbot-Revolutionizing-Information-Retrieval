import os
import dotenv
from getpass import getpass
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter

# Load environment variables from .env file
dotenv.load_dotenv()

# Get the OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = getpass("Enter your OpenAI API key: ")

# PDF Loader class to load and split PDF documents
class PDFLoader:
    def __init__(self, pdf_paths):
        self.pdf_paths = pdf_paths

    def load_and_split_documents(self):
        pdf_data = []
        for path in self.pdf_paths:
            loader = PyPDFLoader(path)
            pdf_data.extend(loader.load())
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(pdf_data)

# Vector Embedding class to create embeddings and store them in Chroma
class VectorEmbedding:
    def __init__(self, openai_api_key):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    def create_vector_store(self, split_data):
        collection_name = "statics_collection"
        local_directory = "statics_vect_embedding"
        persist_directory = os.path.join(os.getcwd(), local_directory)

        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        vect_db = Chroma.from_documents(
            split_data,
            self.embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        vect_db.persist()
        return vect_db

# Chat Model class to handle query-response using GPT-4
class ChatModel:
    def __init__(self, openai_api_key):
        self.chat_model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4", temperature=0)

    def create_chat_qa(self, vect_db):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chat_qa = ConversationalRetrievalChain.from_llm(
            self.chat_model,
            vect_db.as_retriever(),
            memory=memory
        )
        return chat_qa

# Main RAG system orchestrator
class RAGSystem:
    def __init__(self, pdf_paths, openai_api_key):
        # Load and split documents
        pdf_loader = PDFLoader(pdf_paths)
        split_data = pdf_loader.load_and_split_documents()

        # Create vector store and embeddings
        vector_embedding = VectorEmbedding(openai_api_key)
        self.vect_db = vector_embedding.create_vector_store(split_data)

        # Set up the chat model with retrieval
        chat_model = ChatModel(openai_api_key)
        self.chat_qa = chat_model.create_chat_qa(self.vect_db)

    def query(self, input_query):
        response = self.chat_qa({"question": input_query})
        return response["answer"]

# Main function to interact with the system
def main():
    print("Welcome to the PDF Query System! Type 'exit' to quit.")
    
    # PDF paths (you should place your PDF files in the 'pdfs/' folder)
    pdf_paths = ["pdfs/document1.pdf", "pdfs/document2.pdf"]  # Update with your PDFs

    # Initialize RAG system
    rag_system = RAGSystem(pdf_paths, openai_api_key)

    # Query input loop
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break

        if query.strip():  # Ensure the query is not empty
            response = rag_system.query(query)
            print(f'\nAssistant: {response}')
        else:
            print("Please enter a valid query.")

if __name__ == "__main__":
    main()
