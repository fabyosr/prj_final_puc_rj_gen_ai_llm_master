"""
5yo Model — Quantização e alinhamento de LLMs para persona infantil (KD + RLAIF + DPO)
Autor: Fabyo Ramos (PUC-Rio, GenAI & LLM Master)

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
import pandas as pd
import platform
import random
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
    teacher_quant: str     # última coluna de MODELS, ex.: 'Quantização Teacher 4bit'

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.file

    @property
    def family(self) -> str:
        return "0.5B" if "0.5b" in self.file else "1.5B"

    @property
    def teacher_size(self) -> str:
        m = re.search(r"(\d+(?:\.\d+)?B)", self.teacher)
        return m.group(1) if m else self.teacher

    @property
    def teacher_bits(self) -> str:
        m = re.search(r"(\d+)\s*bit", self.teacher_quant, flags=re.I)
        return f"{m.group(1)} bits" if m else self.teacher_quant

    @property
    def menu_label(self) -> str:
        icon = "🔹" if self.family == "0.5B" else "🔸"
        return f"{icon} Student {self.family}  ▸  {self.short_fmt}"

    @property
    def label(self) -> str:
        return f"{self.family} ← {self.teacher_size} ({self.teacher_bits}) · {self.short_fmt}"


_T14 = "Qwen2.5-14B-Instruct"
_S05 = "Qwen2.5-0.5B-Instruct"
_S15 = "Qwen2.5-1.5B-Instruct"

_Q4 = "Quantização Teacher 4bit"  # Teacher usado com o Student 0.5B
_Q8 = "Quantização Teacher 8bit"  # Teacher usado com o Student 1.5B

MODELS: list[ModelSpec] = [
    # ---------------- Student 0.5B ----------------
    ModelSpec("0.5b-f16", "qwen2.5-0.5b-child-f16.gguf", "1tSAeoLJ5tpFsYSeLrBcWyEYMU9PNP3oe",
              _T14, _S05, "FP16 (precisão original)", "FP16", 948, 'Quantização Teacher 4bit'),
    
    ModelSpec("0.5b-q8", "qwen2.5-0.5b-child-Q8_0.gguf", "1pzHMgfCLQuVS-q7_S58-9EqikyDE99jW",
              _T14, _S05, "Quantização 8 bits (Q8_0)", "Q8_0", 506, 'Quantização Teacher 4bit'),

    ModelSpec("0.5b-q5", "qwen2.5-0.5b-child-Q5_K_M.gguf", "1BGBgx6Kxa2EdRrzvbRm6BqzwMTJLFCvI",
              _T14, _S05, "Quantização 4 bits (Q5_K_M)", "Q5_K_M", 400, 'Quantização Teacher 4bit'),

    ModelSpec("0.5b-q4", "qwen2.5-0.5b-child-Q4_K_M.gguf", "1Qg4Mq4yjSKTET7aEgHUhVMnLZE2jGGx9",
              _T14, _S05, "Quantização 4 bits (Q4_K_M)", "Q4_K_M", 379, 'Quantização Teacher 4bit'),

    ModelSpec("0.5b-q3L", "qwen2.5-0.5b-child-Q3_K_M.gguf", "1gdszTpjIKRF1KZEHqjeGIZIfmz48IFcK",
              _T14, _S05, "Quantização 3 bits (Q3_K_L)", "Q3_K_L", 352, 'Quantização Teacher 4bit'),
    
    ModelSpec("0.5b-q3", "qwen2.5-0.5b-child-Q3_K_M.gguf", "1MamA6PvxW0tH7BdcyINtFdbz_bs7B5cq",
              _T14, _S05, "Quantização 3 bits (Q3_K_M)", "Q3_K_M", 339, 'Quantização Teacher 4bit'),
    
    ModelSpec("0.5b-q2", "qwen2.5-0.5b-child-Q2_K.gguf", "1xRALMQsueexExlW83kSu26TpjFvrpv9r",
              _T14, _S05, "Quantização extrema 2 bits (Q2_K)", "Q2_K", 322, 'Quantização Teacher 4bit'),

    # ---------------- Student 1.5B ----------------
    
    ModelSpec("1.5b-14b-f16", "qwen2.5-1.5b-child-f16.gguf", "1j-u346JH6a4D7ZMLw82vW4yF3C38VO1y",
              _T14, _S15, "FP16 (precisão original)", "FP16", 2950, 'Quantização Teacher 8bit'),

    ModelSpec("1.5b-14b-q8", "qwen2.5-1.5b-child-Q8_0.gguf", "1Tf6x4a2BSCIVkMRwfIah7yBcdDZrPgET",
              _T14, _S15, "Quantização 8 bits (Q8_0)", "Q8_0", 1570, 'Quantização Teacher 8bit'),

    ModelSpec("1.5b-14b-q5", "qwen2.5-1.5b-child-Q5_K_M.gguf", "19kH3iq4fpH7GEgiG0IA3PvVHDebaa4sB",
              _T14, _S15, "Quantização 8 bits (Q5_K_M)", "Q5_K_M", 1072, 'Quantização Teacher 8bit'),

    ModelSpec("1.5b-14b-q4", "qwen2.5-1.5b-child-Q4_K_M.gguf", "11W7iqggrl7V6gffeTLfInUZzts1k3_OH",
              _T14, _S15, "Quantização 4 bits (Q4_K_M)", "Q4_K_M", 940, 'Quantização Teacher 8bit'),

    ModelSpec("1.5b-14b-q3L", "qwen2.5-1.5b-child-Q3_K_L.gguf", "1m6PsJn9aL1cDLDtPsOTyBxAo4oaatT8d",
              _T14, _S15, "Quantização 3 bits (Q3_K_L)", "Q3_K_L", 839, 'Quantização Teacher 8bit'),

    ModelSpec("1.5b-14b-q3", "qwen2.5-1.5b-child-Q3_K_M.gguf", "135ZaSOMG4aZCl3mn3KxDBbl41FyBDVKF",
              _T14, _S15, "Quantização 3 bits (Q3_K_M)", "Q3_K_M", 786, 'Quantização Teacher 8bit'),
    
    ModelSpec("1.5b-14b-q2", "qwen2.5-1.5b-child-Q2_K.gguf", "18vMxpgQla91yhx1BqJxJvueGxW8LWyH5",
              _T14, _S15, "Quantização extrema 2 bits (Q2_K)", "Q2_K", 645, 'Quantização Teacher 8bit'),
]
MODELS_BY_KEY = {m.key: m for m in MODELS}

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

OPEN, CLOSE = "<pensamento>", "</pensamento>"

# códigos do enum llama_ftype (general.file_type no GGUF)
FTYPE = {0: "F32", 1: "FP16", 7: "Q8_0", 10: "Q2_K", 12: "Q3_K_M", 13: "Q3_K_L",
         14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K"}

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

            meta = getattr(self.llm, "metadata", None) or {}        # metadados lidos do próprio arquivo GGUF
            try:
                ftype = FTYPE.get(int(meta.get("general.file_type", -1)), "desconhecido")
            except (ValueError, TypeError):
                ftype = "desconhecido"
            try:
                n_params = int(self.llm._model.n_params())          # atributo interno: pode mudar entre versões
            except Exception:
                n_params = None

            self.spec = spec
            self.signature = signature
            self.stats = {
                "file_mb": spec.path.stat().st_size / (1024 * 1024),
                "ram_before": ram_before,
                "ram_after": ram_after,
                "ram_model": max(ram_after - ram_before, 0.0),
                "mmap": use_mmap,
                "file_type": ftype,                                  # formato REAL do arquivo
                "format_ok": ftype == spec.short_fmt,                # bate com o catálogo?
                "n_params": n_params,
                "bits_per_weight": spec.path.stat().st_size * 8 / n_params if n_params else None,
                "default_system": "You are Qwen" in meta.get("tokenizer.chat_template", ""),
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
st.set_page_config(page_title="5yo Model | Fabyo Ramos", page_icon="🎓", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;800&family=Nunito:wght@400;600;700&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');

:root{
  --ink:#1F2A44; --sky:#3E8EF7; --sun:#FFC83D; --mint:#34C79A; --coral:#FF7A6B;
  --bg:#EEF5FF; --card:#FFFFFF; --line:#D9E6FA;
}
html, body, [class*="css"], .stApp { font-family:'Nunito',sans-serif; }
.stApp { background:var(--bg); }

/* ---------- Topo (cabeçalho acadêmico) ---------- */
.hero{
  display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:20px 28px;
  padding:26px 30px; margin-bottom:18px; border-radius:20px; color:#DCE8FF;
  background:
    linear-gradient(rgba(255,255,255,.055) 1px, transparent 1px) 0 0/22px 22px,
    linear-gradient(90deg, rgba(255,255,255,.055) 1px, transparent 1px) 0 0/22px 22px,
    linear-gradient(135deg,#14213D 0%,#1D3566 55%,#2A5AA8 100%);
  box-shadow:0 12px 28px -14px rgba(20,33,61,.7);
}
.hero-main{ flex:1 1 330px; min-width:0; }
.hero h1{
  font-family:'Source Serif 4',Georgia,serif !important; font-weight:700; color:#fff !important;
  font-size:2.35rem !important; line-height:1.05; letter-spacing:-.01em; margin:0; padding:0;
}
.hero .sub{ font-family:'Source Serif 4',Georgia,serif; font-size:1.08rem; line-height:1.35; color:#EAF1FF; margin-top:8px; }
.hero .desc{ font-size:.88rem; line-height:1.5; color:#A9C0EA; margin-top:10px; max-width:46ch; }
.hero .by{ display:flex; flex-direction:column; gap:1px; margin-top:16px; padding-top:12px; border-top:1px solid rgba(255,255,255,.18); }
.hero .by b{ color:#fff; font-size:1rem; }
.hero .by span{ color:#A9C0EA; font-size:.82rem; }

/* Teacher -> Student -> GGUF: os círculos encolhem como o modelo ao ser destilado e quantizado */
.viz{ display:flex; align-items:flex-start; flex:0 0 auto; }
.viz .node{ width:78px; display:flex; flex-direction:column; align-items:center; gap:8px; text-align:center; color:#C9DAFA; font-size:.7rem; line-height:1.25; }
.viz .node b{ color:#fff; font-size:.76rem; }
.viz .dot{ height:64px; display:flex; align-items:center; }
.viz .dot i{ display:block; border-radius:50%; }
.viz .n1 i{ width:64px; height:64px; background:radial-gradient(circle at 35% 30%,#BFE0FF,#3E8EF7); box-shadow:0 0 0 6px rgba(154,208,255,.14); }
.viz .n2 i{ width:44px; height:44px; background:radial-gradient(circle at 35% 30%,#B9F3DE,#34C79A); box-shadow:0 0 0 5px rgba(52,199,154,.16); }
.viz .n3 i{ width:26px; height:26px; background:radial-gradient(circle at 35% 30%,#FFE9A8,#FFC83D); box-shadow:0 0 0 4px rgba(255,200,61,.18); }
.viz .link{ flex:1; min-width:14px; margin-top:32px; border-top:2px dashed rgba(255,255,255,.35); }
@media (max-width:640px){ .hero{ padding:20px; } .hero h1{ font-size:1.9rem !important; } }

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
.meta-chip.warn{ background:#FFF4D6; border-color:#FFE08A; color:#8A5A00; }
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
.card .k{ color:#52617D; }
.card .v{ color:var(--ink); font-weight:700; text-align:right; word-break:break-word; }

.grid{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:6px 0 10px; }
.tile{ background:#F7FAFF; border:1px solid var(--line); border-radius:14px; padding:8px 10px; }
.tile .k{ color:#52617D; font-size:.7rem !important; line-height:1.2; }
.tile .v{ color:var(--ink); font-weight:800; font-size:1.02rem !important; margin-top:2px; }
.tile.hot{ background:#FFF4D6; border-color:#FFE08A; }
.tile.cool{ background:#E7FAF3; border-color:#BDEBDC; }

.badge{ display:inline-block; border-radius:999px; padding:2px 10px; font-weight:700; font-size:.72rem !important; }
.badge.ok{ background:#E7FAF3; color:#14876A; }
.badge.off{ background:#FFE9E6; color:#C4473A; }

/* ---------- Sidebar: linhagem do modelo ---------- */
.lineage{ position:relative; margin:8px 0 6px; padding-left:20px; }
.lineage:before{ content:""; position:absolute; left:6px; top:12px; bottom:14px; border-left:2px dashed var(--line); }
.lineage .stage{ position:relative; padding:5px 0 9px; }
.lineage .stage:before{ content:""; position:absolute; border-radius:50%; }
.lineage .t:before{ width:14px; height:14px; left:-20px; top:7px; background:var(--sky); box-shadow:0 0 0 3px #fff; }
.lineage .s:before{ width:11px; height:11px; left:-18.5px; top:8.5px; background:var(--mint); box-shadow:0 0 0 3px #fff; }
.lineage .f:before{ width:8px; height:8px; left:-17px; top:10px; background:var(--sun); box-shadow:0 0 0 3px #fff; }
.lineage .who{ color:#52617D; font-size:.72rem !important; line-height:1.3; }
.lineage .what{ color:var(--ink); font-weight:700; line-height:1.3; word-break:break-word; }
.lineage .tag{ display:inline-block; margin-top:3px; padding:1px 8px; border-radius:999px; background:#E8F0FE; color:#2C5CC5; font-weight:700; font-size:.68rem !important; }
.file{ font-family:ui-monospace,monospace; font-size:.7rem !important; color:#52617D; word-break:break-all; margin-bottom:8px; }

/* ---------- Sidebar: CPU e hardware ---------- */
.cpu-name{ color:var(--ink); font-weight:700; line-height:1.3; margin:2px 0 8px; word-break:break-word; }
.cpu-name small{ display:block; color:#52617D; font-weight:600; font-size:.7rem !important; }
.tile.wide{ grid-column:1 / -1; }
.meter{ margin:2px 0 10px; }
.meter .lab{ display:flex; justify-content:space-between; color:#52617D; font-size:.72rem !important; margin-bottom:3px; }
.meter .trk{ height:7px; border-radius:99px; background:var(--line); overflow:hidden; }
.meter .trk i{ display:block; height:100%; border-radius:99px; background:var(--mint); }
.meter .trk i.warn{ background:var(--sun); }
.meter .trk i.hot{ background:var(--coral); }
.chips{ display:flex; flex-wrap:wrap; gap:4px; margin:2px 0 8px; }
.chips span{ background:#EEF5FF; border:1px solid var(--line); color:#2C5CC5; border-radius:6px; padding:1px 7px; font-weight:700; font-size:.68rem !important; }

/* ---------- Contraste: o app usa superfícies claras, então o texto precisa ser escuro mesmo com o celular em modo escuro ---------- */
:root{ color-scheme:light; }
.stApp, section[data-testid="stSidebar"]{ color:var(--ink); }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary *,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *{ color:var(--ink) !important; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] :is(p, li, strong, em, h1, h2, h3, h4, td, th){ color:var(--ink) !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="hero-main">
    <h1>5yo Model</h1>
    <div class="sub">Alinhamento de LLMs para persona infantil: efeito da quantização e de pares Teacher–Student</div>
    <div class="desc">Modelos Qwen2.5 de 0.5B e 1.5B treinados com Destilação de Conhecimento, RLAIF e DPO, comparados em sete formatos GGUF, de FP16 a Q2_K.</div>
    <div class="by"><b>Fabyo Ramos</b><span>Pós-graduação em GenAI &amp; LLM Master, PUC-Rio</span></div>
  </div>
  <div class="viz">
    <div class="node n1"><div class="dot"><i></i></div><div><b>Teacher</b><b>Teacher</b><br>14B</div></div>
    <div class="link"></div>
    <div class="node n2"><div class="dot"><i></i></div><div><b>Student</b><br>0.5B / 1.5B</div></div>
    <div class="link"></div>
    <div class="node n3"><div class="dot"><i></i></div><div><b>GGUF</b><br>FP16 a Q2_K</div></div>
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


def lineage_card(s: ModelSpec) -> str:
    """Linha do tempo Teacher -> Student -> formato final."""
    e = html.escape
    return (
        '<div class="lineage">'
        f'<div class="stage t"><div class="who">Teacher</div><div class="what">{e(s.teacher)}</div>'
        f'<span class="tag">quantizado em {e(s.teacher_bits)}</span></div>'
        f'<div class="stage s"><div class="who">Student</div><div class="what">{e(s.student)}</div>'
        '<span class="tag">SFT + KD + DPO</span></div>'
        f'<div class="stage f"><div class="who">Formato final</div><div class="what">{e(s.fmt)}</div></div>'
        '</div>'
        f'<div class="file">{e(s.file)}</div>'
    )


# Extensões de CPU que mais influenciam a velocidade do llama.cpp
SIMD_FLAGS = [
    ("avx2", "AVX2"), ("avx512f", "AVX-512"), ("avx512_vnni", "VNNI"), ("avx_vnni", "VNNI"),
    ("fma", "FMA"), ("f16c", "F16C"), ("avx512_bf16", "BF16"), ("amx_tile", "AMX"),
    ("neon", "NEON"), ("asimd", "NEON"), ("asimddp", "DotProd"),
    ("fphp", "FP16"), ("asimdhp", "FP16"), ("sve", "SVE"),
]


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


@st.cache_resource
def cpu_static_info() -> dict:
    """Dados de CPU que não mudam durante a execução (lidos uma única vez)."""
    name, flags = "", set()
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                key, value = key.strip().lower(), value.strip()
                if key == "model name" and not name:
                    name = value
                elif key in ("flags", "features") and not flags:
                    flags = set(value.lower().split())
                if name and flags:
                    break
    except OSError:
        pass
    name = re.sub(r"\s+", " ", name or platform.processor() or platform.machine() or "CPU não identificada")

    simd: list[str] = []
    for flag, label in SIMD_FLAGS:
        if flag in flags and label not in simd:
            simd.append(label)

    try:
        usable = len(psutil.Process().cpu_affinity())
    except Exception:
        usable = os.cpu_count() or 1

    quota = mem_limit = None  # limites de contêiner (cgroup v2), comuns em servidores gratuitos
    try:
        raw = _read_text("/sys/fs/cgroup/cpu.max")
        if raw and raw.split()[0] != "max":
            quota = int(raw.split()[0]) / int(raw.split()[1])
        raw = _read_text("/sys/fs/cgroup/memory.max")
        if raw and raw != "max" and int(raw) < psutil.virtual_memory().total:
            mem_limit = int(raw)
    except Exception:
        pass

    return {
        "name": name,
        "arch": platform.machine(),
        "physical": psutil.cpu_count(logical=False),
        "logical": psutil.cpu_count(logical=True),
        "effective": min(usable, quota) if quota else usable,
        "simd": simd,
        "simd_known": bool(flags),
        "mem_limit": mem_limit,
    }


def cpu_panel(info: dict, n_threads: int) -> str:
    """Cartão de CPU: nome, núcleos, frequência, SIMD e uso em tempo real."""
    e = html.escape
    pct = psutil.cpu_percent(interval=0.1)
    try:
        freq = psutil.cpu_freq()
        ghz = f"{freq.current / 1000:.2f} GHz" if freq and freq.current else "n/d"
    except Exception:
        ghz = "n/d"
    vm = psutil.virtual_memory()
    level = "hot" if pct >= 85 else "warn" if pct >= 60 else ""

    items = [
        ("Núcleos (físicos / lógicos)", f"{info['physical'] or '?'} / {info['logical']}", ""),
        ("vCPUs utilizáveis", f"{info['effective']:g}", ""),
        ("Frequência atual", ghz, ""),
        ("Threads do modelo", str(n_threads), "hot" if n_threads > info["effective"] else ""),
        ("RAM livre / total do sistema", f"{human(vm.available)} / {human(vm.total)}", "wide cool"),
    ]
    if info["mem_limit"]:
        items.append(("Limite de RAM do contêiner", human(info["mem_limit"]), "wide hot"))

    chips = "".join(f"<span>{e(c)}</span>" for c in info["simd"])
    return (
        f'<div class="cpu-name">{e(info["name"])}<small>{e(info["arch"])}</small></div>'
        f'<div class="meter"><div class="lab"><span>Uso da CPU agora</span><span>{pct:.0f}%</span></div>'
        f'<div class="trk"><i class="{level}" style="width:{max(pct, 2):.0f}%"></i></div></div>'
        + tiles(items)
        + (f'<div class="chips">{chips}</div>' if chips else "")
    )

# =============================================================================
# AVALIAÇÃO DOS MODELOS (seção "📊 Avaliação dos modelos")
# =============================================================================
AVAL_CSV = Path(__file__).parent / "avaliacao" / "avaliacao_modelos.csv"
FORMATOS_ORDEM = ["FP16", "Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_L", "Q3_K_M", "Q2_K"]


@st.cache_data
def carregar_avaliacao() -> pd.DataFrame | None:
    if not AVAL_CSV.exists():
        return None
    df = pd.read_csv(AVAL_CSV)
    df["Student"] = df["student"].str.upper()                                   # 0.5B / 1.5B
    df["Formato"] = pd.Categorical(df["format"].replace({"f16": "FP16"}), FORMATOS_ORDEM, ordered=True)
    return df.sort_values(["Student", "Formato"]).reset_index(drop=True)


def grafico_barras(df: pd.DataFrame, coluna: str, titulo: str, log: bool = False):
    """Barras (ou pontos, em escala log) por formato; azul = 0.5B, laranja = 1.5B."""
    import altair as alt

    d = df.dropna(subset=[coluna]).copy()
    d["Formato"] = d["Formato"].astype(str)
    d["Student"] = "Student " + d["Student"]
    marca = alt.Chart(d).mark_circle(size=150, opacity=0.95) if log else alt.Chart(d).mark_bar()   # barras não existem em escala log
    return (
        marca
        .encode(
            x=alt.X("Student:N", axis=None),
            y=alt.Y(f"{coluna}:Q", title=titulo, scale=alt.Scale(type="log") if log else alt.Scale(zero=True), axis=alt.Axis(grid=not log)),
            color=alt.Color("Student:N", legend=alt.Legend(orient="bottom", title=None),
                            scale=alt.Scale(domain=["Student 0.5B", "Student 1.5B"], range=["#3E8EF7", "#FF9F45"])),
            column=alt.Column("Formato:N", sort=FORMATOS_ORDEM, title=None, header=alt.Header(labelOrient="bottom")),
            tooltip=["Student", "Formato", alt.Tooltip(f"{coluna}:Q", title=titulo)],
        )
        .properties(width=34, height=190)
    )


def render_avaliacao(spec: ModelSpec) -> None:
    df = carregar_avaliacao()
    if df is None:
        st.warning(f"Arquivo da avaliação não encontrado: `{AVAL_CSV}`.")
        return

    # --- modelo selecionado na barra lateral ---------------------------------
    st.markdown(f"##### Modelo selecionado: {spec.menu_label}")
    sel = df[(df["Student"] == spec.family) & (df["Formato"].astype(str) == spec.short_fmt)]
    if not sel.empty:
        r = sel.iloc[0]
        st.markdown(
            tiles([
                ("Formato OK (português)", f"{r.formato_ok_pt_pct:.0f}%", "cool" if r.formato_ok_pt_pct >= 95 else "hot"),
                ("KL vs FP16", "—" if pd.isna(r.kld_mean) else f"{r.kld_mean:.4f}", ""),
                ("Velocidade", f"{r.decode_tps:.1f} tok/s", ""),
                ("RAM de pico", f"{r.ram_pico_mb:,.0f} MB", ""),
            ]) + info_card([("Leitura", r.leitura), ("Falha em português", r.modo_falha)]),
            unsafe_allow_html=True,
        )

    # --- resumo ---------------------------------------------------------------
    st.markdown(
        """
**Resumo da avaliação.** Os 14 modelos responderam aos mesmos 100 prompts (30 de curiosidade, 50 de guardrail e 20 sensíveis legítimos), sem *system prompt* explícito, com temperatura 0 e semente fixa, em CPU.

- **Formato (CoT) em português:** Q8_0, Q5_K_M e Q4_K_M equivalem ao FP16 nos dois Students. O 0.5B degrada aos poucos no Q3 e no Q2_K; o 1.5B **quebra no Q3** (o modelo pensa e para, sem resposta final) e entra em laço no Q2_K.
- **Velocidade e RAM:** o FP16 é o único realmente lento em CPU. O Q4_K_M gera cerca de 2× mais rápido e usa ~45% (0.5B) a ~61% (1.5B) menos RAM de pico.
- **Idioma:** o dataset de treino é 100% pt-BR. Em inglês e espanhol o CoT quase não aparece (FP16: 1 de 7 prompts no 0.5B e 5 de 7 no 1.5B).
- **Limites:** guardrails e adequação à idade ainda não foram pontuados por um avaliador, e com 100 prompts por modelo as diferenças pequenas não são conclusivas.
"""
    )

    # --- tabelas e gráficos ---------------------------------------------------
    base = df.copy()
    base["Formato"] = base["Formato"].astype(str)
    t1, t2, t3 = st.tabs(["Geração de texto", "Fidelidade ao FP16", "Custo em CPU"])

    with t1:
        st.altair_chart(grafico_barras(df, "formato_ok_pt_pct", "Formato OK em português (%)"))
        st.dataframe(
            base[["Student", "Formato", "formato_ok_pt_pct", "formato_ok_pt_n", "truncado_pct", "repeticao", "modo_falha", "leitura"]]
            .rename(columns={"formato_ok_pt_pct": "Formato OK (PT)", "formato_ok_pt_n": "OK em PT (n)", "truncado_pct": "Truncado (%)",
                             "repeticao": "Repetição", "modo_falha": "Falha em português", "leitura": "Leitura"}),
            hide_index=True,
            column_config={
                "Formato OK (PT)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
                "Truncado (%)": st.column_config.NumberColumn(format="%.0f"),
                "Repetição": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    with t2:
        st.altair_chart(grafico_barras(df, "kld_mean", "KL média vs FP16 (escala log)", log=True))
        st.dataframe(
            base[["Student", "Formato", "bits_per_weight", "kld_mean", "same_top_p", "ppl_delta_pct"]]
            .rename(columns={"bits_per_weight": "Bits/peso", "kld_mean": "KL média", "same_top_p": "Top-1 igual (%)", "ppl_delta_pct": "ΔPPL (%)"}),
            hide_index=True,
            column_config={"KL média": st.column_config.NumberColumn(format="%.4f"), "Bits/peso": st.column_config.NumberColumn(format="%.2f")},
        )
        st.caption("KL e top-1 medem o quanto a distribuição do modelo quantizado se afasta da do FP16 do mesmo Student (~20 mil tokens pontuados).")
    with t3:
        st.altair_chart(grafico_barras(df, "decode_tps", "Velocidade de geração (tok/s)"))
        st.altair_chart(grafico_barras(df, "ram_pico_mb", "RAM de pico (MB)"))
        st.dataframe(
            base[["Student", "Formato", "size_mb", "ram_pico_mb", "decode_tps", "s_por_resposta", "ttft_s"]]
            .rename(columns={"size_mb": "Arquivo (MB)", "ram_pico_mb": "RAM de pico (MB)", "decode_tps": "tok/s",
                             "s_por_resposta": "s/resposta", "ttft_s": "TTFT (s)"}),
            hide_index=True,
        )
        st.caption("CPU com 8 threads e mmap desligado. s/resposta = tokens da resposta ÷ tok/s + TTFT.")

    with st.expander("Como a avaliação foi feita"):
        st.markdown(
            """
- **Prompts:** 100, sem histórico; 7 deles em inglês ou espanhol. Temperatura 0, semente 42, máximo de 512 tokens, contexto de 2048. O template de chat do Qwen2.5 pode inserir o *system prompt* padrão do modelo.
- **Formato OK:** um único bloco `<pensamento>…</pensamento>` bem formado, seguido de uma resposta não vazia. As taxas usam só os 93 prompts em português.
- **Fidelidade:** PPL e KL sobre ~40 mil tokens (79 janelas de 512), comparando cada formato com o FP16 do mesmo Student.
- **Custo:** pico de RAM do processo com o modelo carregado; tokens por segundo na geração (mediana).
- **Limitações:** os Students usam Teachers com precisões diferentes (4 bits no 0.5B e 8 bits no 1.5B); não há avaliador de guardrails e persona; as heurísticas por palavra-chave foram descartadas por gerarem falsos positivos.
"""
        )
    st.download_button("⬇️ Baixar tabela (CSV)", data=base.drop(columns=["Student", "Formato"]).to_csv(index=False).encode("utf-8"), file_name="avaliacao_modelos.csv", mime="text/csv")

manager = get_manager()

with st.sidebar:
    st.markdown("### 🧪 Laboratório de modelos")

    # --- Seleção -------------------------------------------------------------
    default_idx = 1  # 0.5B Q4_K_M, o modelo que você já usava
    selected_key = st.selectbox(
        "Modelo",
        options=[m.key for m in MODELS],
        index=1,  # 0.5B Q4_K_M como padrão
        key="selected_model",
        format_func=lambda k: MODELS_BY_KEY[k].menu_label,  # rótulo estático
    )
    spec = MODELS_BY_KEY[selected_key]

    # --- Ajustes de execução (afetam a medição, por isso vêm antes do load) ---
    with st.expander("⚙️ Ajustes", expanded=False):
      n_threads = st.slider(
        "Threads de CPU", 1, max(os.cpu_count() or 2, 2), min(2, os.cpu_count() or 2),
        help="Quantas threads de CPU o llama.cpp usa para gerar. Mais threads aceleram até o número de núcleos "
        "físicos; acima disso costumam piorar. Alterar recarrega o modelo e refaz a medição de RAM.",
      )
      use_mmap = st.toggle(
        "Usar mmap",
        value=True,
        help="Ligado: o SO carrega o arquivo sob demanda (RAM sobe aos poucos). "
        "Desligado: o modelo inteiro é copiado para a RAM ao carregar, e a medição reflete o custo real. "
        "Alterar recarrega o modelo.",
      )
      temperature = st.slider(
        "Temperatura", 0.0, 1.5, 0.45, 0.05,
        help="Aleatoriedade das respostas. 0 = sempre escolhe o token mais provável (determinístico, ideal para "
        "comparar modelos). Valores altos deixam as respostas mais variadas, e mais sujeitas a erros.",
      )
      max_tokens = st.slider(
        "Máx. de tokens", 50, 600, 350, 25,
        help="Limite de tokens por resposta. Conta o raciocínio (<pensamento>) e a resposta final juntos: "
        "se for baixo, a resposta é cortada (aparece “cortada no limite de tokens”). "
        "Na avaliação dos modelos foi usado 512.",
      )
      max_msg_history = st.slider(
        "Max message history", 1, 10, 1, 1,
        help="Quantas mensagens recentes da conversa (suas e do modelo) são reenviadas a cada pergunta. "
        "O modelo não guarda memória: o app reenvia esse histórico. Valores maiores dão mais contexto, "
        "mas ocupam a janela de 2048 tokens e deixam a resposta mais lenta.",
      )

    # --- Ficha do modelo -----------------------------------------------------
    on_disk = spec.path.exists()
    badge = '<span class="badge ok">no disco</span>' if on_disk else '<span class="badge off">não baixado</span>'
    st.markdown(badge, unsafe_allow_html=True)
    st.markdown(lineage_card(spec), unsafe_allow_html=True)

    # --- 
    Download sob demanda ------------------------------------------------
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
                ("Parâmetros", f"{s['n_params'] / 1e6:,.0f} M" if s["n_params"] else "n/d", ""),
                ("Bits por peso (real)", f"{s['bits_per_weight']:.2f}" if s["bits_per_weight"] else "n/d", ""),
            ]),
            unsafe_allow_html=True,
        )
        if s["format_ok"]:
            st.caption(f"✔️ Formato verificado no arquivo GGUF: {s['file_type']}.")
        else:
            st.error(f"⚠️ O arquivo é {s['file_type']}, mas o catálogo diz {spec.short_fmt}. Confira o id do Drive.")
        if s["default_system"]:
            st.caption("O template de chat deste modelo insere o system prompt padrão do Qwen (“You are Qwen…”) "
                       "quando nenhum é enviado.")
        if s["mmap"]:
            st.caption("Com mmap, parte do modelo só entra na RAM quando é usada. "
                       "O valor “atual” sobe depois das primeiras respostas.")
        else:
            st.caption("mmap desligado: “RAM do modelo” mostra o custo completo.")

    # --- CPU e hardware ------------------------------------------------------
    with st.expander("🖥️ CPU e hardware", expanded=True):
        cpu = cpu_static_info()
        st.markdown(cpu_panel(cpu, n_threads), unsafe_allow_html=True)
        if cpu["simd_known"] and not cpu["simd"]:
            st.caption("Nenhuma extensão SIMD (AVX2/NEON) detectada: a inferência tende a ser bem mais lenta.")
        if n_threads > cpu["effective"]:
            st.caption("Há mais threads do que vCPUs utilizáveis; isso costuma reduzir a velocidade.")
        if spec.short_fmt == "FP16":
            st.caption("FP16 lê cerca de 3× mais bytes por token que Q4_K_M. Em CPU a geração é limitada "
                       "pela memória, por isso este formato é bem mais lento.")

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

# --- Navegação principal: conversa ou avaliação ---------------------------
VIEW_CHAT, VIEW_AVAL = "💬 Conversa", "📊 Avaliação dos modelos"
if hasattr(st, "segmented_control"):
    view = st.segmented_control("Seção", [VIEW_CHAT, VIEW_AVAL], default=VIEW_CHAT, key="view",
                                label_visibility="collapsed") or VIEW_CHAT
else:
    view = st.radio("Seção", [VIEW_CHAT, VIEW_AVAL], horizontal=True, key="view", label_visibility="collapsed")
if view == VIEW_AVAL:
    render_avaliacao(spec)
    st.stop()

# Sem arquivo local não há o que conversar ainda.
if not spec.path.exists() or manager.llm is None:
    st.info("👈 Escolha um modelo na barra lateral e baixe-o para começar a conversar.")
    st.stop()


# =============================================================================
# 5. CHAT
# =============================================================================
FORMAT_STATUS_PT = {
    "ok": "CoT ok", "sem_tags": "sem pensamento", "nao_fechado": "pensamento não fechado",
    "so_fechamento": "tag de fechamento solta", "tag_duplicada": "pensamento repetido",
    "ordem_invertida": "tags fora de ordem", "texto_antes_do_pensamento": "texto antes do pensamento",
    "resposta_vazia": "sem resposta final",
}


def classify_format(text: str) -> str:
    """Classifica o formato <pensamento>…</pensamento> + resposta final."""
    o, c = text.count(OPEN), text.count(CLOSE)
    if o == 0 and c == 0:
        return "sem_tags"
    if o == 0:
        return "so_fechamento"
    if c == 0:
        return "nao_fechado"
    if o > 1 or c > 1:
        return "tag_duplicada"
    if text.index(CLOSE) < text.index(OPEN):
        return "ordem_invertida"
    if text[:text.index(OPEN)].strip():
        return "texto_antes_do_pensamento"
    return "ok" if text.split(CLOSE, 1)[1].strip() else "resposta_vazia"


def split_thought(text: str) -> tuple[str, str]:
    """(pensamento, resposta final), com tolerância a formatos quebrados."""
    if CLOSE in text:
        before, after = text.rsplit(CLOSE, 1)
        return before.replace(OPEN, "").strip(), after.strip()
    if OPEN in text:
        return text.split(OPEN, 1)[1].strip(), ""
    return "", text.strip()


def render_message(content: str, meta: dict | None = None) -> None:
    """Mostra a resposta; o pensamento vai para um expander e falhas de formato viram avisos."""
    meta = meta or {}
    thought, answer = split_thought(content)
    status = meta.get("format") or classify_format(content)
    finish = meta.get("finish")

    if thought:
        with st.expander("🧠 Cadeia de pensamento"):
            st.markdown(f'<div class="cot-box">{html.escape(thought)}</div>', unsafe_allow_html=True)

    if answer:
        st.markdown(answer)
    elif thought:
        motivo = "A resposta foi cortada no limite de tokens." if finish == "length" else "O modelo parou sozinho antes de responder."
        st.warning(f"O modelo pensou, mas não escreveu a resposta final. {motivo}")
    else:
        st.caption("(resposta vazia)")

    if meta:
        chips = [f'<span class="meta-chip">{html.escape(meta["model"])}</span>',
                 f'<span class="meta-chip">{meta["tokens"]} tokens</span>']
        if meta.get("tps"):
            chips.append(f'<span class="meta-chip">{meta["tps"]:.1f} tok/s</span>')
        if meta.get("ttft"):
            chips.append(f'<span class="meta-chip">1º token em {meta["ttft"]:.2f} s</span>')
        chips.append(f'<span class="meta-chip">{meta["secs"]:.1f} s</span>')
        chips.append(f'<span class="meta-chip{"" if status == "ok" else " warn"}">{FORMAT_STATUS_PT.get(status, status)}</span>')
        if finish == "length":
            chips.append('<span class="meta-chip warn">cortada no limite de tokens</span>')
        st.markdown("".join(chips), unsafe_allow_html=True)

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
    llm_messages = [{"role": m["role"], "content": m["content"]} for m in history[-max_msg_history:]]
    
    with st.chat_message("assistant", avatar="🤖"):
        # Lista de palavras
        palavras = ["Pensando", "Cucando ", "Quebrando a cabeça", "Viajando na maionese", "Bolando "]

        # Escolhe uma palavra aleatoriamente
        palavra_escolhida = random.choice(palavras)
        placeholder = st.empty()
        placeholder.markdown(f"{palavra_escolhida}… 🤔")

        text, finish, t_first = "", None, None
        t_start = time.perf_counter()
        with manager.gen_lock:
            stream = manager.llm.create_chat_completion(
                messages=llm_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                ch = chunk["choices"][0]
                delta = (ch.get("delta") or {}).get("content")
                if delta:
                    if t_first is None:
                        t_first = time.perf_counter()
                    text += delta
                    placeholder.markdown(text.replace(OPEN, "💭 *pensando:* ").replace(CLOSE, "\n\n---\n\n") + "▌")
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]       # "stop" = terminou sozinho; "length" = cortada no limite
        t_end = time.perf_counter()

        n_tokens = len(manager.llm.tokenize(text.encode("utf-8"), add_bos=False)) if text else 0   # contagem real
        text = text.strip()
        meta = {
            "model": spec.label,
            "tokens": n_tokens,
            "secs": t_end - t_start,
            "ttft": (t_first - t_start) if t_first else None,
            "tps": (n_tokens - 1) / (t_end - t_first) if t_first and n_tokens > 1 and t_end > t_first else None,  # só decodificação
            "finish": finish,
            "format": classify_format(text),
        }
        placeholder.empty()
        render_message(text, meta)

    history.append({"role": "assistant", "content": text, "meta": meta})
    st.rerun()  # atualiza a RAM "atual" na sidebar já com o modelo aquecido