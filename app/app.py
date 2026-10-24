import streamlit as st
import os
import re
import gdown
from llama_cpp import Llama

# Configurações da página
st.set_page_config(page_title="Seu Amiguinho Virtual!", page_icon="🤖", layout="centered")

# Variáveis do modelo
file_id = "1KLdCXc8s37SjmPENKbTntQB5VMFNSuEH"
file_name = "qwen2.5-1.5b-5yoModel-Q4_K_M.gguf"

# Função para formatar o pensamento na interface
def formatar_resposta(texto):
    # Substitui a tag <pensamento> por um bloco com visual bonito (fundo escuro, texto verde estilo terminal)
    estilo_pensamento = r'''
    <div style="background-color: #1e1e1e; color: #4af626; padding: 12px; border-radius: 8px; font-family: monospace; font-size: 0.9em; margin-bottom: 15px; border-left: 4px solid #4af626; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);">
        <span style="color: #ffffff; font-weight: bold; font-family: sans-serif;">🧠 Pensando:</span><br>
        \1
    </div>
    '''
    # Usa regex para encontrar e substituir todas as ocorrências
    texto_formatado = re.sub(r'<pensamento>(.*?)</pensamento>', estilo_pensamento, texto, flags=re.DOTALL)
    return texto_formatado

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

# 2. Construindo a Interface Visual Amigável
st.title("🤖 Seu Amiguinho Virtual!")
st.write("Olá! Eu sou seu amiguinho inteligente! Adoro brincar de aprender coisas novas e responder suas curiosidades. O que você quer saber hoje?")

# 3. Gerenciamento de Histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(formatar_resposta(msg["content"]), unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# 4. Campo de entrada e resposta
if prompt := st.chat_input("Digite sua perguntinha aqui..."):
    # Adiciona e exibe a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepara a resposta do assistente
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Configura o comportamento do modelo (persona de 5 anos)
        system_msg = {"role": "system", "content": "Você é um amiguinho virtual de 5 anos. Responda de forma simples, amigável e educativa."}
        llm_messages = [system_msg] + st.session_state.messages
        
        with st.spinner("Pensando..."):
            output = llm.create_chat_completion(
                messages=llm_messages,
                max_tokens=250,
                temperature=0.7,
            )
        
        resposta = output["choices"][0]["message"]["content"].strip()
        
        # Formata a resposta exibindo o pensamento estilizado e a resposta final
        resposta_formatada = formatar_resposta(resposta)
        
        message_placeholder.markdown(resposta_formatada, unsafe_allow_html=True)
    
    # Adiciona a resposta bruta ao histórico (para ser reformatada se a tela recarregar)
    st.session_state.messages.append({"role": "assistant", "content": resposta})
