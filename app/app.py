import streamlit as st
import os
import re
import gdown
import psutil
from llama_cpp import Llama

# Configurações da página
st.set_page_config(page_title="Seu Amiguinho Virtual!", page_icon="🤖", layout="centered")

# Variáveis do modelo
file_id = "1KLdCXc8s37SjmPENKbTntQB5VMFNSuEH"
file_name = "qwen2.5-1.5b-5yoModel-Q4_K_M.gguf"

# 1. Download e Carregamento do modelo
@st.cache_resource
def load_model():
    if not os.path.exists(file_name):
        with st.spinner("Baixando os arquivos mágicos do amiguinho... Isso pode demorar um pouquinho na primeira vez."):
            gdown.download(id=file_id, output=file_name, quiet=False)

    return Llama(
        model_path=file_name,
        n_ctx=2048,
        n_threads=2, # Limita as threads para evitar sobrecarga no servidor gratuito
        verbose=False
    )

llm = load_model()

# --- Sidebar: Monitoramento de RAM ---
st.sidebar.title("📊 Monitoramento")
# Captura o uso de memória do processo atual (que engloba o modelo carregado)
process = psutil.Process(os.getpid())
mem_mb = process.memory_info().rss / (1024 * 1024)
st.sidebar.metric("Uso de RAM (App + Modelo)", f"{mem_mb:.2f} MB")
st.sidebar.caption("Nota: O `llama.cpp` utiliza MMAP. O sistema operacional aloca memória RAM dinamicamente conforme necessário com base no tamanho do GGUF.")
# -------------------------------------

# 2. Construindo a Interface Visual Amigável
st.title("🤖 Seu Amiguinho Virtual!")
st.write("Olá! Eu sou seu amiguinho inteligente! Adoro brincar de aprender coisas novas e responder suas curiosidades. O que você quer saber hoje?")

def exibir_mensagem_formatada(texto):
    """Separa os pensamentos da resposta final e exibe usando componentes do Streamlit."""
    # Encontra todos os blocos de pensamento
    pensamentos = re.findall(r'<pensamento>(.*?)</pensamento>', texto, flags=re.DOTALL)
    # Remove os pensamentos para obter apenas a resposta final
    resposta_final = re.sub(r'<pensamento>.*?</pensamento>', '', texto, flags=re.DOTALL).strip()
    
    # Se houver pensamentos, os coloca dentro de um Expander (Acordeão)
    if pensamentos:
        with st.expander("🧠 Cadeia de Pensamento (CoT)"):
            for p in pensamentos:
                st.markdown(f"<div style='color: #4af626; background-color: #1e1e1e; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.85em; margin-bottom: 10px;'>{p.strip()}</div>", unsafe_allow_html=True)
    
    # Exibe a resposta final com destaque principal
    if resposta_final:
        st.markdown(resposta_final)

# 3. Gerenciamento de Histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            exibir_mensagem_formatada(msg["content"])
        else:
            st.markdown(msg["content"])

# 4. Campo de entrada e resposta
if prompt := st.chat_input("Digite sua perguntinha aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        system_msg = {"role": "system", "content": "Você é um amiguinho virtual de 5 anos. Responda de forma simples, amigável e educativa."}
        llm_messages = [system_msg] + st.session_state.messages

        with st.spinner("Pensando..."):
            output = llm.create_chat_completion(
                messages=llm_messages,
                max_tokens=250,
                temperature=0.7,
            )

        resposta = output["choices"][0]["message"]["content"].strip()

        # Exibe a resposta final e os pensamentos
        exibir_mensagem_formatada(resposta)

    # Adiciona ao histórico do Streamlit
    st.session_state.messages.append({"role": "assistant", "content": resposta})
