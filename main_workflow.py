#Obiettivo di main_workflow.py è eseguire l'intero processo in un ciclo interattivo. All'avvio carcia tutti i componenti
# attende la domanda (query) dall'utente
# [WF-1] Analisi: passa la query all'analista
# [WF-2] Ricerca: prende l'output dell'analista (filter_dict) e la query e li passa al bibliotecario (ricerca_knoledge_base)
# [WF-3] Ragionamento: prende l'output del bibliotecario (context) e la query e li passa al giurista (reasoning_chain)
# [WF-4] Risposta: stampa a schermo la risposta finale del giurista.
# Loop: ricomincia al commento 2: "Attende la domanda (query) dell'utente"

#Domanda che la v1 sbagliava : "Ho comprato un telefono da un negozio 5 mesi fa. Il venditore dice che devo provare che il difetto esisteva. E' vero?"
# Risposta Sbagliata: "Si, basandosi esclusivamente sul contesto fornito, il venditore ha ragione. Secondo il principio espresso dalla Cassazione Civile,
# Sez.II, 18 Maggio, 2020, n.9139, l'onore della prova dell'esistenza del vizio, e che questo sussiteva già al momento della consegna, grava sul compratore."

#Promemoria 1: attivare il cassetto venv prima di eseguire qualsiasi codice nel progetto. Codcie da inserire " .\venv\Scripts\activate "
#Promemoria 2: per distruggere l'ambiete corrotto codice da usare " rmdir /s /q venv "
#Promemoria 3: per distruggere il database  codice da usare " rmdir /s /q db_giuridica_faiss "


import time
from tools import load_vectorstore, ricerca_knowledge_base
from agents import create_reasoning_chain, create_filter_chain

def run_legal_agent_v2():
    """
    Funzione principale che orchestra il workflow agentico v2
    (Analista -> Bibliotecario -> Giurista).
    """
    
    # --- 1. INIZIALIZZAZIONE ---
    print("Avvio del sistema di ragionamento giuridico (PoC v2)...")
    
    print("(Fase 1/3) Caricamento Vectorstore (FAISS + Embeddings)...")
    try:
        vectorstore = load_vectorstore()
        if vectorstore is None:
            print("Errore critico: il Vectorstore non è stato caricato. Uscita.")
            return
    except Exception as e:
        print(f"Errore fatale durante il caricamento del vectorstore: {e}")
        return
        
    print("\n(Fase 2/3) Caricamento 'Giurista' (Chain di Ragionamento Gemini)...")
    try:
        reasoning_chain = create_reasoning_chain()
    except Exception as e:
        print(f"Errore fatale durante il caricamento di Gemini (Giurista): {e}")
        return
    
    print("\n(Fase 3/3) Caricamento 'Analista' (Filter Chain Gemini)...")
    try:
        filter_chain = create_filter_chain()
    except Exception as e:
        print(f"Errore fatale durante il caricamento di Gemini (Analista): {e}")
        return

    print("\n--- Sistema v2 Pronto. ---")
    
    # --- 2. LOOP DI ESECUZIONE (Il Workflow Interattivo) ---
    while True:
        print("\n----------------------------------")
        query = input("Inserisci il tuo quesito (o scrivi 'esci' per terminare):\n> ")
        
        if query.lower() in ['esci', 'exit', 'quit']:
            break
            
        if not query.strip():
            print("Input vuoto, riprova.")
            continue
            
        start_time = time.time()

        # --- FASE 1: ANALISI  ---
        print("\n...[WF-1] Il 'Supervisore' interroga l''Analista' (Gemini)...")
        try:
            filter_dict = filter_chain.invoke({"query": query})
            print(f"...[WF-1] Filtro determinato: {filter_dict}")
        except Exception as e:
            print(f"Errore dell'Analista: {e}. Procedo senza filtro.")
            filter_dict = {} 

        # --- FASE 2: RICERCA  ---
        print("\n...[WF-2] Il 'Supervisore' interroga il 'Bibliotecario' (FAISS)...")
        context = ricerca_knowledge_base(query, vectorstore, filter_dict)
        print(f"...[WF-2] Contesto recuperato:\n{context}\n")
        
        # --- FASE 3: RAGIONAMENTO  ---
        print("...[WF-3] Il 'Supervisore' invia Contesto e Quesito al 'Giurista' (Gemini)...")
        
        try:
            risposta = reasoning_chain.invoke({
                "context": context,
                "domanda": query
            })
            
            end_time = time.time()

            # --- FASE 4: RISPOSTA ---
            print("\n--- RISPOSTA FINALE DEL SISTEMA ---")
            print(risposta)
            print("----------------------------------")
            print(f"(Analisi completata in {end_time - start_time:.2f} secondi)")
            
        except Exception as e:
            print(f"\n--- ERRORE DURANTE IL RAGIONAMENTO ---")
            print(f"Errore: {e}")
            print("Riprova con un'altra domanda.")

    print("\n--- Sistema disattivato. Arrivederci. ---")

# --- Avvio dello script ---
if __name__ == "__main__":
    run_legal_agent_v2()