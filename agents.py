#Obiettivo di agents.py definire le personalità e le istruzioni per i 2 agenti LLM che usano Gemini Analista e Giurista

# --- Agente Analista ---
# il suo obiettivo è quello di classificare la domanda  dell'utente attraverso il prompt specifico che gli dice di analizzare la domanda e di restituire ,
# solo un output JSON (es. {'regime': 'b2c'}). JsonOutputParser forza LLM a restituire JSON strutturato basato su RegimeFilter invece di testo libero

# --- Agente Giurista ---
# il suo obiettivo è quello di formulare la risposta finale. Attraverso il prompt del Sillogismo Giuridico (template_sillogismo = ....)  lo istruisce a  seguire
# la struttura  "Norme Rilevanti", "Analisi del fatto", "Conclusione". Ha l'istruzione " Basa la tua risposta **ESCLUSIVAMENTE** sul contesto fornito" questo previene
# le "allucinazioni" dell'IA




import os
from dotenv import load_dotenv
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.output_parsers.json import JsonOutputParser
from pydantic import BaseModel, Field

# Carica la chiave API segreta dal file .env
load_dotenv()

# --- 1. Definizione dello schema di output (Pydantic) ---
class RegimeFilter(BaseModel):
    regime: str = Field(description="Il regime giuridico da applicare. Deve essere 'b2c' (per consumatori/negozi) o 'cc' (per vendite tra privati/aziende).")

# --- 2. "Analista" ---
def create_filter_chain():
    """
    Crea un agente "Analista" che legge la query
    e decide quale regime legale (filtro) applicare.
    """
    print("Caricamento 'Analista' (Filter Chain)...")
    

    parser = JsonOutputParser(pydantic_object=RegimeFilter)

    filter_prompt_template = """
    Sei un assistente legale esperto. Il tuo unico compito è analizzare la seguente
    domanda di un utente e determinare il regime giuridico applicabile.

    Devi restituire un output ESCLUSIVAMENTE in formato JSON.
    
    -   Se la domanda menziona un 'negozio', 'professionista', 'azienda', 'online',
        'consumatore' o un contesto B2C, imposta il regime su 'b2c'.
    -   Se la domanda menziona 'privato', 'auto usata tra privati' o un contesto
        C2C/B2B, imposta il regime su 'cc'.
    -   Se la domanda è generica (es. "parlami dei vizi"), 
        imposta il regime su 'cc' come default.

    Domanda:
    {query}

    Formato JSON di output:
    {format_instructions}
    """

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.0)

    prompt = ChatPromptTemplate.from_template(
        template=filter_prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Creiamo la chain: Prompt -> LLM -> Parser JSON
    filter_chain = prompt | llm | parser
    
    return filter_chain


# --- 3. Giurista ---
def create_reasoning_chain():
    """
    Crea la "catena" di ragionamento che combina
    il prompt del sillogismo e il modello LLM (Gemini).
    """
    
    template_sillogismo = """
    Sei un assistente legale esperto di diritto italiano.
    Il tuo compito è applicare un rigoroso ragionamento giuridico
    basato sul sillogismo (Norma -> Fatto -> Conclusione).

    Basa la tua risposta **ESCLUSIVAMENTE** sul contesto fornito.
    Non inventare informazioni, norme o scadenze.
    Se il contesto non contiene la risposta, dichiara di non poter rispondere.

    **Contesto (Premessa Maggiore - La Norma):**
    {context}

    **Quesito (Premessa Minore - Il Fatto):**
    {domanda}

    **Output Strutturato (Conclusione - Il Ragionamento):**
    Fornisci una risposta chiara e strutturata in 3 parti:
    
    1.  **Norme Rilevanti:** (Identifica e cita gli articoli chiave dal contesto)
    2.  **Analisi del Fatto:** (Applica le norme al quesito)
    3.  **Conclusione:** (Fornisci la risposta pratica alla domanda)
    """

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.1)

    # 4. Crea la "Chain" di Ragionamento
    prompt = ChatPromptTemplate.from_template(template_sillogismo)
    output_parser = StrOutputParser()

    ragionamento_chain = prompt | llm | output_parser
    
    return ragionamento_chain

