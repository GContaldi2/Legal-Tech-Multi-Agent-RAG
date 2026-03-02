# ⚖️ Legal-Tech Multi-Agent RAG

Un'architettura software basata su Agenti Intelligenti (LLM) per risolvere il problema delle allucinazioni e dei conflitti normativi nell'ambito del Legal Tech.

Questo progetto è stato sviluppato come lavoro di Tesi di Laurea in Informatica.

## 🎯 Il Problema
I sistemi standard basati su Retrieval-Augmented Generation (RAG) lineare falliscono nel dominio legale a causa dell'ipertrofia legislativa. Una semplice ricerca per similarità semantica rischia di mescolare leggi in conflitto (es. Codice Civile vs Codice del Consumo), generando allucinazioni pericolose per l'utente.

## 🚀 La Soluzione: Architettura Multi-Agente
Questo progetto propone un approccio multi-agente che intercetta la query prima della ricerca vettoriale, simulando il ragionamento di un giurista umano attraverso step sequenziali:

1. **Agente Router (Zero-Shot Classifier):** Riceve la domanda dell'utente e ne classifica il profilo (es. Consumatore vs Professionista/Azienda).
2. **Filtro Metadati (Pre-Filtering):** Utilizza l'output del Router per applicare un filtro a monte sul Vector Store, escludendo i corpus normativi non pertinenti prima del calcolo della similarità.
3. **Agente Giurista:** Riceve un contesto pulito e mirato, generando un parere legale accurato e privo di contaminazioni.

## 🛠️ Stack Tecnologico
* **Linguaggio:** Python
* **Orchestrazione:** LangChain
* **LLM (Reasoning & Generation):** Gemini 1.5 Pro
* **Embedding Model:** Google `text-embedding-004`
* **Vector Database:** FAISS

## 📊 Risultati Sperimentali
Testato su un Golden Dataset di 50 casi legali reali (es. garanzia compravendita B2B vs B2C):
* **RAG Lineare Standard:** Accuratezza ~52%
* **Multi-Agent RAG (Proposto):** Accuratezza **~94%**

## ⚙️ Struttura del Progetto
* `main_workflow.py`: Script principale che orchestra il flusso degli agenti.
* `agents.py` / `tools.py`: Definizione degli agenti LangChain e dei tool di ricerca.
* `ingest.py`: Script per l'elaborazione del `corpus_giuridico` e la creazione del database vettoriale FAISS.
* `corpus_giuridico/`: Cartella contenente i testi di legge (Codice Civile e Codice del Consumo) suddivisi per articoli.

## 🚀 Installazione e Utilizzo

1. Clona la repository:
   `
   git clone [https://github.com/GContaldi2/Legal-Tech-Multi-Agent-RAG.git](https://github.com/GContaldi2/nLegal-Tech-Multi-Agent-RAG.git)
   cd Legal-Tech-Multi-Agent-RAG
   `
2. Crea e attiva l'ambiente virtuale:
   `
   python -m venv venv
   venv\Scripts\activate
   (Nota per utenti MAC/Linux : source venv/bin/activate)
   `
3. Installa le dipendenze necessarie:
   `
   pip install -r requirements.txt
   `
4. Configura le API Key:
   Crea un file di testo chiamato .env nella cartella principale del progetto e inserisci la tua chiave di Google Gemini:
   `
   GOOGLE_API_KEY="inserisci_qui_la_tua_chiave"
   `
5. Inizializza il Vector Store (FAISS):
   Prima di poter interrogare il sistema, devi vettorializzare i documenti di legge. Esegui questo script (va fatto solo la prima volta):
   `
   python ingest.py
   `
6. Avvia l'Agente e fai la tua domanda:
   Una volta creato il database, puoi avviare il flusso principale:
   `
   python main_workflow.py
   `
   (Il sistema si avvierà nel terminale e ti chiederà di inserire il tuo caso pratico in linguaggio naturale per generare il parere legale).


## ⚠️ Disclaimer
   Questo software è un prototipo accademico (Proof of Concept). Le risposte generate dall'Intelligenza Artificiale non costituiscono consulenza legale professionale.
