"""
Seu Amiguinho Virtual — comparador de modelos SLM (SFT + RLAIF/KD + DPO)

Estrutura:
  1. CATÁLOGO        -> lista de modelos (fácil de editar)
  2. GERENCIADOR     -> baixa, carrega/descarrega e mede RAM (singleton compartilhado)
  3. ESTILO          -> CSS (sidebar compacta + visual infantil)
  4. SIDEBAR         -> seleção, ficha do modelo, métricas, ajustes, disco
  5. CHAT            -> histórico separado por modelo, streaming e métricas por resposta
"""

import ctypes
import gc
import glob
import html
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import gdown
import psutil
import streamlit as st
from llama_cpp import Llama

# =============================================================================
# 1. CATÁLOGO DE MODELOS
# =============================================================================
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class ModelSpec:
    key: str               # id interno (nome do arquivo sem extensão)
    file: str              # nome do .gguf
    drive_id: str | None   # id do Google Drive (None = pendente)
    teacher: str
    student: str
    fmt: str               # descrição do formato final
    short_fmt: str         # rótulo curto (FP16 / Q4_K_M / Q2_K)
    approx_mb: int         # tamanho aproximado, usado só na barra de progresso

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.file

    @property
    def family(self) -> str:
        return "0.5B" if "0.5b" in self.file else "1.5B"

    @property
    def label(self) -> str:
        return f"Qwen2.5 {self.family} · {self.short_fmt}"


_T14 = "Qwen2.5-14B-Instruct"
_S05 = "Qwen2.5-0.5B-Instruct"
_T72 = "Qwen2.5-72B-Instruct"
_S15 = "Qwen2.5-1.5B-Instruct"

MODELS: list[ModelSpec] = [
    ModelSpec("0.5b-f16", "qwen2.5-0.5b-child-f16.gguf", "1dH9Tzbm0ITiMCWEceq5lmQIt3Ox_aAOj",
              _T14, _S05, "FP16 (precisão original)", "FP16", 1000),
    ModelSpec("0.5b-q4", "qwen2.5-0.5b-child-Q4_K_M.gguf", "1WhNjVh4TGXNj086_ELSG8taiUEOOVlb6",
              _T14, _S05, "Quantização 4 bits (Q4_K_M)", "Q4_K_M", 400),
    ModelSpec("0.5b-q2", "qwen2.5-0.5b-child-Q2_K.gguf", "13OD_oOBdc-i5Rk8aKzrIuQI24ig9FgC7",
              _T14, _S05, "Quantização extrema 2 bits (Q2_K)", "Q2_K", 340),
    ModelSpec("1.5b-f16", "qwen2.5-1.5b-5yoModel-f16.gguf", "1bnjNFOToXqyJQHD-Mx2sv5beoQ6tslHg",
              _T72, _S15, "FP16 (precisão original)", "FP16", 3100),
    ModelSpec("1.5b-q4", "qwen2.5-1.5b-5yoModel-Q4_K_M.gguf", "1KLdCXc8s37SjmPENKbTntQB5VMFNSuEH",
              _T72, _S15, "Quantização 4 bits (Q4_K_M)", "Q4_K_M", 990),
    # ATENÇÃO: na lista original o Q2_K do 1.5B tinha o MESMO id do Q4_K_M.
    # Baixar assim geraria um arquivo "Q2_K" que na verdade é Q4_K_M e invalidaria a comparação.
    # Preencha com o id correto do Drive:
    ModelSpec("1.5b-q2", "qwen2.5-1.5b-5yoModel-Q2_K.gguf", "1I4MKo_b8QdbMSTqx6nhmkMKKsvoal2eA",
              _T72, _S15, "Quantização extrema 2 bits (Q2_K)", "Q2_K", 680),
]
MODELS_BY_KEY = {m.key: m for m in MODELS}

SYSTEM_PROMPT = (
    "Você é um amiguinho virtual de 5 anos. "
    "Responda de forma simples, amigável e educativa."
)
MAX_HISTORY_MSGS = 12  # evita estourar o n_ctx de 2048 tokens


# =============================================================================
# 2. GERENCIADOR DE MODELO (download + load + RAM)
# =============================================================================
def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def trim_memory() -> None:
    """Pede à glibc para devolver páginas livres ao SO (sem isso o RSS 'não desce')."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class ModelManager:
    """Mantém UM modelo na RAM por vez e registra métricas honestas a cada troca."""

    def __init__(self):
        self.llm: Llama | None = None
        self.spec: ModelSpec | None = None
        self.signature: tuple | None = None
        self.stats: dict = {}
        self.gen_lock = threading.Lock()  # um prompt por vez no servidor

    def unload(self) -> None:
        if self.llm is not None:
            try:
                self.llm.close()
            except Exception:
                pass
        self.llm = None
        self.spec = None
        self.signature = None
        gc.collect()
        trim_memory()

    def load(self, spec: ModelSpec, n_threads: int, use_mmap: bool) -> None:
        signature = (spec.key, n_threads, use_mmap)
        if self.signature == signature and self.llm is not None:
            return

        with self.gen_lock:
            self.unload()                    # 1) libera o anterior
            time.sleep(0.2)
            ram_before = rss_mb()            # 2) baseline limpo (app sem modelo)

            self.llm = Llama(                # 3) carrega o novo
                model_path=str(spec.path),
                n_ctx=2048,
                n_threads=n_threads,
                use_mmap=use_mmap,
                verbose=False,
            )
            ram_after = rss_mb()

            self.spec = spec
            self.signature = signature
            self.stats = {
                "file_mb": spec.path.stat().st_size / (1024 * 1024),
                "ram_before": ram_before,
                "ram_after": ram_after,
                "ram_model": max(ram_after - ram_before, 0.0),
                "mmap": use_mmap,
            }


@st.cache_resource
def get_manager() -> ModelManager:
    return ModelManager()


def download_model(spec: ModelSpec, bar, text: str) -> None:
    """Baixa via gdown em thread e mostra progresso olhando o arquivo parcial."""
    if spec.drive_id is None:
        raise RuntimeError(f"O id do Google Drive de {spec.file} não foi informado.")

    result: dict = {}

    def worker():
        try:
            gdown.download(id=spec.drive_id, output=str(spec.path), quiet=True)
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    expected = spec.approx_mb * 1024 * 1024
    while thread.is_alive():
        # gdown grava num arquivo temporário na mesma pasta antes de renomear
        parts = glob.glob(str(MODELS_DIR / f"{spec.file}*"))
        done = sum(os.path.getsize(p) for p in parts if os.path.exists(p))
        bar.progress(min(done / expected, 0.99), text=f"{text} · {done / 1024**2:,.0f} MB")
        time.sleep(0.3)

    if "error" in result or not spec.path.exists():
        for p in glob.glob(str(MODELS_DIR / f"{spec.file}*")):
            if not p.endswith(".gguf"):
                os.remove(p)
        raise RuntimeError(f"Falha no download: {result.get('error', 'arquivo não encontrado')}")

    bar.progress(1.0, text=f"{text} · concluído")


# =============================================================================
# 3. ESTILO
# =============================================================================
st.set_page_config(page_title="Seu Amiguinho Virtual!", page_icon="🤖", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;800&family=Nunito:wght@400;600;700&display=swap');

:root{
  --ink:#1F2A44; --sky:#3E8EF7; --sun:#FFC83D; --mint:#34C79A; --coral:#FF7A6B;
  --bg:#EEF5FF; --card:#FFFFFF; --line:#D9E6FA;
}
html, body, [class*="css"], .stApp { font-family:'Nunito',sans-serif; }
.stApp { background:var(--bg); }

/* ---------- Topo ---------- */
.hero{
  display:flex; align-items:center; gap:18px;
  background:linear-gradient(120deg,#3E8EF7 0%,#6FB2FF 60%,#9AD0FF 100%);
  border-radius:28px; padding:22px 26px; margin-bottom:18px;
  box-shadow:0 10px 0 -4px #2C6FCB;
}
.hero .bot{ font-size:56px; line-height:1; filter:drop-shadow(0 4px 0 rgba(0,0,0,.12)); }
.hero h1{ font-family:'Baloo 2',cursive; font-weight:800; color:#fff; margin:0; font-size:2.1rem; line-height:1.05; padding:0; }
.hero p{ color:#EAF3FF; margin:4px 0 0; font-size:1.02rem; }

/* ---------- Chat ---------- */
[data-testid="stChatMessage"]{
  background:var(--card); border:2px solid var(--line); border-radius:22px;
  padding:14px 18px; margin-bottom:10px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  background:#FFF6DA; border-color:#FFE59A;
}
.meta-chip{
  display:inline-block; background:#EEF5FF; border:1px solid var(--line); color:#4B5B7C;
  border-radius:999px; padding:2px 10px; font-size:.74rem; margin:6px 6px 0 0;
}
.cot-box{
  background:#17203A; color:#7CF0C0; padding:10px 12px; border-radius:12px;
  font-family:ui-monospace,monospace; font-size:.82rem; margin-bottom:8px; white-space:pre-wrap;
}

/* ---------- Sidebar compacta ---------- */
section[data-testid="stSidebar"]{ background:#fff; border-right:2px solid var(--line); }
section[data-testid="stSidebar"] *{ font-size:.82rem; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3{
  font-family:'Baloo 2',cursive; font-size:1.05rem !important; margin:.2rem 0 .4rem; padding:0; color:var(--ink);
}
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{ padding-top:1.2rem; }

.card{ background:#F7FAFF; border:1px solid var(--line); border-radius:14px; padding:10px 12px; margin:6px 0 10px; }
.card .row{ display:flex; justify-content:space-between; gap:8px; padding:3px 0; border-bottom:1px dashed var(--line); }
.card .row:last-child{ border-bottom:none; }
.card .k{ color:#6B7A99; }
.card .v{ color:var(--ink); font-weight:700; text-align:right; word-break:break-word; }

.grid{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:6px 0 10px; }
.tile{ background:#F7FAFF; border:1px solid var(--line); border-radius:14px; padding:8px 10px; }
.tile .k{ color:#6B7A99; font-size:.7rem !important; line-height:1.2; }
.tile .v{ color:var(--ink); font-weight:800; font-size:1.02rem !important; margin-top:2px; }
.tile.hot{ background:#FFF4D6; border-color:#FFE08A; }
.tile.cool{ background:#E7FAF3; border-color:#BDEBDC; }

.badge{ display:inline-block; border-radius:999px; padding:2px 10px; font-weight:700; font-size:.72rem !important; }
.badge.ok{ background:#E7FAF3; color:#14876A; }
.badge.off{ background:#FFE9E6; color:#C4473A; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="bot">🤖</div>
  <div>
    <h1>Seu Amiguinho Virtual!</h1>
    <p>Eu adoro aprender coisas novas! O que você quer saber hoje?</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# 4. SIDEBAR
# =============================================================================
def human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} PB"


def info_card(rows: list[tuple[str, str]]) -> str:
    body = "".join(f'<div class="row"><span class="k">{k}</span><span class="v">{html.escape(v)}</span></div>' for k, v in rows)
    return f'<div class="card">{body}</div>'


def tiles(items: list[tuple[str, str, str]]) -> str:
    body = "".join(f'<div class="tile {cls}"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v, cls in items)
    return f'<div class="grid">{body}</div>'


manager = get_manager()

with st.sidebar:
    st.markdown("### 🧪 Laboratório de modelos")

    # --- Seleção -------------------------------------------------------------
    default_idx = 1  # 0.5B Q4_K_M, o modelo que você já usava
    selected_key = st.selectbox(
        "Modelo",
        options=[m.key for m in MODELS],
        index=default_idx,
        format_func=lambda k: ("✅ " if MODELS_BY_KEY[k].path.exists() else "☁️ ") + MODELS_BY_KEY[k].label,
        help="✅ já está no disco · ☁️ ainda precisa ser baixado",
    )
    spec = MODELS_BY_KEY[selected_key]

    # --- Ajustes de execução (afetam a medição, por isso vêm antes do load) ---
    with st.expander("⚙️ Ajustes", expanded=False):
        n_threads = st.slider("Threads de CPU", 1, max(os.cpu_count() or 2, 2), min(2, os.cpu_count() or 2))
        use_mmap = st.toggle(
            "Usar mmap",
            value=True,
            help="Ligado: o SO carrega o arquivo sob demanda (RAM sobe aos poucos). "
                 "Desligado: o modelo inteiro é copiado para a RAM ao carregar, e a medição reflete o custo real.",
        )
        temperature = st.slider("Temperatura", 0.0, 1.5, 0.7, 0.05)
        max_tokens = st.slider("Máx. de tokens", 50, 600, 250, 25)
        use_system = st.toggle(
            "Enviar system prompt",
            value=False,
            help="Desligado: testa se o treino (SFT/DPO) já alinhou o modelo sem precisar de instrução extra.",
        )

    # --- Ficha do modelo -----------------------------------------------------
    on_disk = spec.path.exists()
    badge = '<span class="badge ok">no disco</span>' if on_disk else '<span class="badge off">não baixado</span>'
    st.markdown(badge, unsafe_allow_html=True)
    st.markdown(
        info_card([
            ("Student", spec.student),
            ("Teacher", spec.teacher),
            ("Formato final", spec.fmt),
            ("Arquivo", spec.file),
        ]),
        unsafe_allow_html=True,
    )

    # --- Download sob demanda ------------------------------------------------
    if not on_disk:
        if spec.drive_id is None:
            st.warning("O id do Google Drive deste modelo está pendente. Edite o catálogo em `MODELS`.")
        else:
            st.caption(f"Tamanho aproximado: {spec.approx_mb:,} MB")
            if st.button("⬇️ Baixar este modelo", use_container_width=True, type="primary"):
                bar = st.progress(0.0, text="Preparando…")
                try:
                    download_model(spec, bar, "Baixando")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
                else:
                    st.rerun()

    # --- Carregamento (automático ao trocar) ---------------------------------
    if spec.path.exists():
        if manager.signature != (spec.key, n_threads, use_mmap):
            with st.spinner(f"Carregando {spec.label}…"):
                manager.load(spec, n_threads, use_mmap)

    # --- Métricas do modelo ativo --------------------------------------------
    if manager.llm is not None and manager.spec is not None and manager.spec.key == spec.key:
        s = manager.stats
        st.markdown("### 📊 Monitoramento")
        st.markdown(
            tiles([
                ("Arquivo GGUF", f"{s['file_mb']:,.0f} MB", ""),
                ("RAM antes do modelo", f"{s['ram_before']:,.0f} MB", ""),
                ("RAM do modelo", f"{s['ram_model']:,.0f} MB", "hot"),
                ("RAM atual (app + modelo)", f"{rss_mb():,.0f} MB", "cool"),
            ]),
            unsafe_allow_html=True,
        )
        if s["mmap"]:
            st.caption("Com mmap, parte do modelo só entra na RAM quando é usada. "
                       "O valor “atual” sobe depois das primeiras respostas.")
        else:
            st.caption("mmap desligado: “RAM do modelo” mostra o custo completo.")

    # --- Baixar todos --------------------------------------------------------
    missing = [m for m in MODELS if not m.path.exists() and m.drive_id]
    if missing:
        with st.expander(f"📦 Baixar todos ({len(missing)} faltando)"):
            total_mb = sum(m.approx_mb for m in missing)
            st.caption(f"Aprox. {total_mb / 1024:.1f} GB no total.")
            if st.button("Baixar todos os modelos", use_container_width=True):
                overall = st.progress(0.0, text="Iniciando…")
                current = st.progress(0.0, text="")
                try:
                    for i, m in enumerate(missing, start=1):
                        overall.progress((i - 1) / len(missing), text=f"Modelo {i} de {len(missing)}")
                        download_model(m, current, m.short_fmt + " " + m.family)
                    overall.progress(1.0, text="Tudo pronto!")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
                else:
                    st.rerun()

    # --- Disco (uma única vez, só onde ficam os modelos) ---------------------
    du = shutil.disk_usage(MODELS_DIR)
    models_bytes = sum(p.stat().st_size for p in MODELS_DIR.glob("*.gguf"))
    st.markdown("### 💾 Disco")
    st.progress(du.used / du.total, text=f"{human(du.free)} livres de {human(du.total)}")
    st.caption(f"Modelos baixados ocupam {human(models_bytes)}")

    if st.button("🧹 Limpar conversa deste modelo", use_container_width=True):
        st.session_state.setdefault("histories", {})[spec.key] = []
        st.rerun()

# Sem arquivo local não há o que conversar ainda.
if not spec.path.exists() or manager.llm is None:
    st.info("👈 Escolha um modelo na barra lateral e baixe-o para começar a conversar.")
    st.stop()


# =============================================================================
# 5. CHAT
# =============================================================================
def render_message(content: str, meta: dict | None = None) -> None:
    """Mostra a resposta; <pensamento>…</pensamento> vai para um expander."""
    thoughts = re.findall(r"<pensamento>(.*?)</pensamento>", content, flags=re.DOTALL)
    answer = re.sub(r"<pensamento>.*?</pensamento>", "", content, flags=re.DOTALL).strip()

    if thoughts:
        with st.expander("🧠 Cadeia de pensamento"):
            for t in thoughts:
                st.markdown(f'<div class="cot-box">{html.escape(t.strip())}</div>', unsafe_allow_html=True)

    st.markdown(answer or content)

    if meta:
        chips = (
            f'<span class="meta-chip">{html.escape(meta["model"])}</span>'
            f'<span class="meta-chip">{meta["tokens"]} tokens</span>'
            f'<span class="meta-chip">{meta["tps"]:.1f} tok/s</span>'
            f'<span class="meta-chip">{meta["secs"]:.1f} s</span>'
        )
        st.markdown(chips, unsafe_allow_html=True)


histories: dict = st.session_state.setdefault("histories", {})
history: list = histories.setdefault(spec.key, [])  # conversa separada por modelo

for msg in history:
    avatar = "🧒" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant":
            render_message(msg["content"], msg.get("meta"))
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Digite sua perguntinha aqui…"):
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧒"):
        st.markdown(prompt)

    # Só role/content vão para o modelo (o campo "meta" fica de fora)
    llm_messages = [{"role": m["role"], "content": m["content"]} for m in history[-MAX_HISTORY_MSGS:]]
    if use_system:
        llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + llm_messages

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown("Pensando… 🤔")

        text, n_tokens = "", 0
        start = time.perf_counter()
        with manager.gen_lock:
            stream = manager.llm.create_chat_completion(
                messages=llm_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    text += delta
                    n_tokens += 1
                    placeholder.markdown(text + "▌")
        elapsed = time.perf_counter() - start

        text = text.strip()
        meta = {
            "model": spec.label,
            "tokens": n_tokens,
            "secs": elapsed,
            "tps": n_tokens / elapsed if elapsed > 0 else 0.0,
        }
        placeholder.empty()
        render_message(text, meta)

    history.append({"role": "assistant", "content": text, "meta": meta})
    st.rerun()  # atualiza a RAM "atual" na sidebar já com o modelo aquecido