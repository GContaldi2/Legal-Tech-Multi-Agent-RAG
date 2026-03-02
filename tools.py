#Tale file svolge 2 funzioni fondamentali

# --- Funzione 1 : load_vectorstore() ---
# Caricare il database FAISS e il modello emdedding in memoria all'avvio del programma

# --- Funzione 2: ricerca_knowledge_base ---
# Eseguire la ricerca ibrida, questa funzione riceve sia la query (la domanda) sia il filter_dict


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

DB_PATH = "db_giuridica_faiss"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def load_vectorstore():
    """
    Carica il database vettoriale (FAISS) da disco.
    Restituisce l'intero oggetto vectorstore.
    """
    print("Caricamento del modello di embedding (HuggingFace)...")
    
    embedding_function = HuggingFaceEmbeddings( 
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    print(f"Caricamento del database vettoriale da: {DB_PATH}...")
    try:
        vectorstore = FAISS.load_local(
            DB_PATH, 
            embeddings=embedding_function,
            allow_dangerous_deserialization=True 
        )
    except Exception as e:
        print(f"\n--- ERRORE ---")
        print(f"Impossibile caricare il database da '{DB_PATH}'.")
        print("Hai eseguito correttamente lo script 'ingest.py' con i metadati?")
        print(f"Dettaglio errore: {e}")
        return None

    print("Database (Vectorstore) caricato.")
    return vectorstore

def ricerca_knowledge_base(query_utente: str, vectorstore, filter_dict: dict) -> str:
    """
    Esegue la ricerca semantica sul database vettoriale
    APPLICANDO UN FILTRO SUI METADATI.
    """
    print(f"\n--- Attivazione Tool Ricerca con Filtro: {filter_dict} ---")
    print(f"Query ricevuta: '{query_utente}'")
    
    documenti = vectorstore.similarity_search(
        query_utente,
        k=3,
        filter=filter_dict
    )
    
    if not documenti:
        print("Nessun documento trovato con il filtro. Tento ricerca generica...")
        documenti = vectorstore.similarity_search(query_utente, k=2)
        if not documenti:
            return "Nessun documento rilevante trovato."
        
    print(f"Trovati {len(documenti)} documenti rilevanti.")
    
    context = "\n---\n".join([doc.page_content for doc in documenti])
    
    sources = [doc.metadata.get('source', 'Sconosciuta') for doc in documenti]
    context += f"\n\nFonti: {', '.join(set(sources))}"
    
    return context

