import streamlit as st
import os
import re
import gdown
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configuração inicial da página
st.set_page_config(page_title="Amiguinho Virtual", page_icon="🤖", layout="centered")

# Variáveis do modelo
file_id = "1KLdCXc8s37SjmPENKbTntQB5VMFNSuEH"
file_name = "qwen2.5-1.5b-5yoModel-Q4_K_M.gguf"
model_id = file_name
diretorio_modelo = "."

# Função de download com cache
@st.cache_resource(show_spinner=False)
def download_model():
    if os.path.exists(file_name):
        return True

    if file_id == "COLOQUE_O_ID_DO_ARQUIVO_AQUI" or file_id == "":
        st.error("⚠️ Por favor, atualize o arquivo app.py com o ID real do modelo no Google Drive!")
        return False

    try:
        gdown.download(id=file_id, output=file_name, quiet=False)
        return os.path.exists(file_name)
    except Exception as e:
        st.error(f"Erro ao baixar o arquivo: {e}")
        return False

# Função de carregamento do modelo com cache
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        diretorio_modelo,
        gguf_file=model_id,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return tokenizer, pipe

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
        tokenizer, pipe = load_model()

    # Inicializa o histórico de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibe as mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada de texto do usuário
    if prompt := st.chat_input("Digite sua perguntinha aqui..."):
        # Mostra a mensagem do usuário
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Gera o prompt para o modelo
        prompt_gguf = tokenizer.apply_chat_template(
            st.session_state.messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Resposta do modelo
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                output = pipe(
                    prompt_gguf,
                    max_new_tokens=250,
                    temperature=0.7,
                    do_sample=True,
                    return_full_text=False
                )
                resposta = output[0]['generated_text'].strip()

                # Remove tags de <pensamento> para a interface ficar limpa para as crianças
                resposta_limpa = re.sub(r'<pensamento>.*?</pensamento>', '', resposta, flags=re.DOTALL).strip()

                st.markdown(resposta_limpa)

        # Salva a resposta no histórico
        st.session_state.messages.append({"role": "assistant", "content": resposta_limpa})
else:
    st.error("Puxa, não consegui carregar meus arquivos mágicos. Verifique o ID do arquivo!")
