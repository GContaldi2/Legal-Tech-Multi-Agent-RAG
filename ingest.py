#viene eseguita una sola volta per preparare la "bibilioteca"
#Obiettivo di ingest.py leggere i file.txt presenti nella cartella "corpus giuridico" e trasformarli in un database vettoriale (FAISS)
# def parse_custom_txt questo script cerca un'intestazione speciale
# estrazione i vari "tag" (es. regime: b2c , tipo:giurisprudenza) e li salva in un dizionario
# dividie il contenuto in pezzi più piccoli per ottimizzare la ricerca ( text_splitter=....)
# attraverso il modello locale paraphrase-multilingual-MiniLM-L12-v2 , il testo diviso in pezzi più piccoli , viene trasforamto in un vettore numerico (embedding) che ne rappresenta il significato semantico
# crea e salva (db_giurisprudenza_faiss) l'indice FAISS che collega i vettori numerici ai loro pezzi di testo e ai loro metadati


import os
import re # Importiamo le regular expression per il parsing
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

CORPUS_PATH = "corpus_giuridico"
DB_PATH = "db_giuridica_faiss"

def parse_custom_txt(file_path):
    """
    Legge un file .txt, estrae i metadati dall'intestazione
    e separa il contenuto.
    Formato atteso:
    ---
    tag: valore
    tag2: valore2
    ---
    Contenuto del documento...
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    metadata = {}
    document_content = content
    

    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL | re.MULTILINE)
    
    if match:
        header_lines = match.group(1).split('\n')
        document_content = match.group(2).strip() # Rimuove spazi bianchi
        
        for line in header_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
    
    
    metadata['source'] = file_path.replace(os.path.sep, '/')
    return Document(page_content=document_content, metadata=metadata)

def ingest_data():
    """
    Funzione per caricare, splittare e indicizzare (Versione con Metadati).
    """
    print("Inizio l'ingestion dei dati con Metadati...")
    
    all_documents = []
    for file_name in os.listdir(CORPUS_PATH):
        if file_name.endswith('.txt'):
            file_path = os.path.join(CORPUS_PATH, file_name)
            doc = parse_custom_txt(file_path)
            all_documents.append(doc)

    if not all_documents:
        print(f"Errore: Nessun documento .txt trovato nella cartella '{CORPUS_PATH}'.")
        return
    print(f"Caricati {len(all_documents)} documenti con metadati.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". "]
    )
    splits = text_splitter.split_documents(all_documents)
    print(f"Documenti divisi in {len(splits)} chunks.")

    print("Avvio del modello di embedding gratuito (HuggingFace)...")
    model_name = "paraphrase-multilingual-MiniLM-L12-v2" 
    embedding_function = HuggingFaceEmbeddings( 
        model_name=model_name,
        model_kwargs={'device': 'cpu'} 
    )
    
    print("Inizio la creazione degli embeddings e del vector store (FAISS)...")
    vectorstore = FAISS.from_documents(
        documents=splits, 
        embedding=embedding_function
    )
    
    vectorstore.save_local(DB_PATH)

    print(f"--- Processo di Ingestion completato! ---")
    print(f"Il tuo database vettoriale (FAISS) è stato salvato in: '{DB_PATH}'")

if __name__ == "__main__":
    ingest_data()