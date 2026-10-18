import streamlit as st
import os
import re
import gdown
from llama_cpp import Llama

# Configuração inicial da página
st.set_page_config(page_title="Amiguinho Virtual", page_icon="🤖", layout="centered")

# Variáveis do modelo
file_id = "1KLdCXc8s37SjmPENKbTntQB5VMFNSuEH"
file_name = "qwen2.5-1.5b-5yoModel-Q4_K_M.gguf"

# Função de download com cache
@st.cache_resource(show_spinner=False)
def download_model():
    if os.path.exists(file_name):
        return True

    if file_id == "COLOQUE_O_ID_DO_ARQUIVO_AQUI" or file_id == "":
        st.error("⚠️ Por favor, atualize o arquivo app.py com o ID real do modelo!")
        return False

    try:
        gdown.download(id=file_id, output=file_name, quiet=False)
        return os.path.exists(file_name)
    except Exception as e:
        st.error(f"Erro ao baixar o arquivo: {e}")
        return False

# Função de carregamento do modelo com cache (usando llama.cpp)
@st.cache_resource(show_spinner=False)
def load_model():
    # O llama-cpp-python é otimizado e poupa muita RAM
    llm = Llama(
        model_path=file_name,
        n_ctx=2048,  # Limite de contexto para não estourar a memória
        n_threads=2, # Número de threads leve para o Streamlit Cloud
        verbose=False
    )
    return llm

# Interface amigável
st.title("🤖 Seu Amiguinho Virtual!")

# Placeholder para imagem 200x200
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://placehold.co/200x200/ffcc00/ffffff?text=Mascote", width=100)
with col2:
    st.markdown("### Olá! Eu sou seu amiguinho inteligente! Adoro brincar de aprender coisas novas e responder suas curiosidades. O que você quer saber hoje?")

# Sugestões de perguntas
with st.expander("💡 Ideias do que me perguntar:"):
    st.markdown("""
    - *Por que o céu é azul?*
    - *Como os passarinhos conseguem voar?*
    - *O que é um arco-íris?*
    - *De onde vem a chuva?*
    """)

# Fluxo principal de carregamento
with st.spinner("Preparando a mágica... (pode demorar um pouquinho na primeira vez)"):
    is_downloaded = download_model()

if is_downloaded:
    with st.spinner("Acordando o amiguinho..."):
        llm = load_model()

    # Inicializa o histórico de chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "Você é um amiguinho virtual de 5 anos. Responda de forma simples, amigável e educativa."}
        ]

    # Exibe as mensagens anteriores
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Entrada de texto do usuário
    if prompt := st.chat_input("Digite sua perguntinha aqui..."):
        # Mostra a mensagem do usuário
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Resposta do modelo
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                output = llm.create_chat_completion(
                    messages=st.session_state.messages,
                    max_tokens=250,
                    temperature=0.7,
                )
                
                resposta = output["choices"][0]["message"]["content"].strip()

                # Remove tags de <pensamento> para a interface ficar limpa para as crianças
                resposta_limpa = re.sub(r'<pensamento>.*?</pensamento>', '', resposta, flags=re.DOTALL).strip()

                st.markdown(resposta_limpa)

        # Salva a resposta no histórico
        st.session_state.messages.append({"role": "assistant", "content": resposta_limpa})
else:
    st.error("Puxa, não consegui carregar meus arquivos mágicos. Verifique o ID do arquivo!")
