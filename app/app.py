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
import platform
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
    def label(self) -> str:
        return f"{self.family} ← {self.teacher_size} ({self.teacher_bits}) · {self.short_fmt}"


_T14 = "Qwen2.5-14B-Instruct"
_S05 = "Qwen2.5-0.5B-Instruct"
_S15 = "Qwen2.5-1.5B-Instruct"
_T72 = "Qwen2.5-72B-Instruct"
_S15 = "Qwen2.5-1.5B-Instruct"

MODELS: list[ModelSpec] = [
    ModelSpec("0.5b-f16", "qwen2.5-0.5b-child-f16.gguf", "1dH9Tzbm0ITiMCWEceq5lmQIt3Ox_aAOj",
              _T14, _S05, "FP16 (precisão original)", "FP16", 1000, 'Quantização Teacher 4bit'),
    ModelSpec("0.5b-q8", "qwen2.5-0.5b-child-Q8_0.gguf", "1zRFFc7JGIebKetEkd0lvz7Ob9f5rTXjU",
              _T14, _S05, "Quantização 8 bits (Q8_0)", "Q8_0", 506, 'Quantização Teacher 4bit'),
    ModelSpec("0.5b-q3", "qwen2.5-0.5b-child-Q3_K_M.gguf", "1bSwhBSpq0ZqmGOCcg3gFG6NPaSFReNlm",
              _T14, _S05, "Quantização 3 bits (Q3_K_M)", "Q3_K_M", 339, 'Quantização Teacher 4bit'),
    ModelSpec("0.5b-q4", "qwen2.5-0.5b-child-Q4_K_M.gguf", "1WhNjVh4TGXNj086_ELSG8taiUEOOVlb6",
              _T14, _S05, "Quantização 4 bits (Q4_K_M)", "Q4_K_M", 400, 'Quantização Teacher 4bit'),
    ModelSpec("0.5b-q2", "qwen2.5-0.5b-child-Q2_K.gguf", "13OD_oOBdc-i5Rk8aKzrIuQI24ig9FgC7",
              _T14, _S05, "Quantização extrema 2 bits (Q2_K)", "Q2_K", 340, 'Quantização Teacher 4bit'),
    
    ModelSpec("1.5b-f16", "qwen2.5-1.5b-5yoModel-f16.gguf", "1bnjNFOToXqyJQHD-Mx2sv5beoQ6tslHg",
              _T72, _S15, "FP16 (precisão original)", "FP16", 3100, 'Quantização Teacher 4bit'),
    ModelSpec("1.5b-q8", "qwen2.5-1.5b-5yoModel-Q8_0.gguf", "1ukOMhDm_Ch-rjLXVI5oG4rqgWCcUVdIx",
              _T72, _S15, "Quantização 8 bits (Q8_0)", "Q8_0", 1536, 'Quantização Teacher 4bit'),
    ModelSpec("1.5b-q3", "qwen2.5-1.5b-5yoModel-Q3_K_M.gguf", "1DUZBotvAcSHwmsYDPuiqHIqW0tUlwPWa",
              _T72, _S15, "Quantização 3 bits (Q3_K_M)", "Q3_K_M", 786, 'Quantização Teacher 4bit'),
    ModelSpec("1.5b-q4", "qwen2.5-1.5b-5yoModel-Q4_K_M.gguf", "1KLdCXc8s37SjmPENKbTntQB5VMFNSuEH",
              _T72, _S15, "Quantização 4 bits (Q4_K_M)", "Q4_K_M", 990, 'Quantização Teacher 4bit'),
    ModelSpec("1.5b-q2", "qwen2.5-1.5b-5yoModel-Q2_K.gguf", "1I4MKo_b8QdbMSTqx6nhmkMKKsvoal2eA",
              _T72, _S15, "Quantização extrema 2 bits (Q2_K)", "Q2_K", 680, 'Quantização Teacher 4bit'),

    ModelSpec("1.5b-14b-f16", "qwen2.5-1.5b-child-f16.gguf", "12XajOl4i-GuMx7OcEnH4W71gDviBqsI3",
              _T14, _S15, "FP16 (precisão original)", "FP16", 2880, 'Quantização Teacher 8bit'),
    ModelSpec("1.5b-14b-q8", "qwen2.5-1.5b-child-Q8_0.gguf", "1_IcwltLjLUjT93zbEiZqm6xvyrtOYcXl",
              _T14, _S15, "Quantização 8 bits (Q8_0)", "Q8_0", 1536, 'Quantização Teacher 8bit'),
    ModelSpec("1.5b-14b-q3", "qwen2.5-1.5b-child-Q3_K_M.gguf", "1Mc1dDuYEk_AfGk7WIZ5bi99t1G3tUDJq",
              _T14, _S15, "Quantização 3 bits (Q3_K_M)", "Q3_K_M", 786, 'Quantização Teacher 8bit'),
    ModelSpec("1.5b-14b-q4", "qwen2.5-1.5b-child-Q4_K_M.gguf", "1yTDiDxq7_CLWt54mHnTwXWNk0vsSYW4_",
              _T14, _S15, "Quantização 4 bits (Q4_K_M)", "Q4_K_M", 940, 'Quantização Teacher 8bit'),
    ModelSpec("1.5b-14b-q2", "qwen2.5-1.5b-child-Q2_K.gguf", "1pP9TKyFnkr5_pbwIvci7o7wly0ZSiBS1",
              _T14, _S15, "Quantização extrema 2 bits (Q2_K)", "Q2_K", 645, 'Quantização Teacher 8bit'),
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

/* ---------- Sidebar: linhagem do modelo ---------- */
.lineage{ position:relative; margin:8px 0 6px; padding-left:20px; }
.lineage:before{ content:""; position:absolute; left:6px; top:12px; bottom:14px; border-left:2px dashed var(--line); }
.lineage .stage{ position:relative; padding:5px 0 9px; }
.lineage .stage:before{ content:""; position:absolute; border-radius:50%; }
.lineage .t:before{ width:14px; height:14px; left:-20px; top:7px; background:var(--sky); box-shadow:0 0 0 3px #fff; }
.lineage .s:before{ width:11px; height:11px; left:-18.5px; top:8.5px; background:var(--mint); box-shadow:0 0 0 3px #fff; }
.lineage .f:before{ width:8px; height:8px; left:-17px; top:10px; background:var(--sun); box-shadow:0 0 0 3px #fff; }
.lineage .who{ color:#6B7A99; font-size:.72rem !important; line-height:1.3; }
.lineage .what{ color:var(--ink); font-weight:700; line-height:1.3; word-break:break-word; }
.lineage .tag{ display:inline-block; margin-top:3px; padding:1px 8px; border-radius:999px; background:#E8F0FE; color:#2C5CC5; font-weight:700; font-size:.68rem !important; }
.file{ font-family:ui-monospace,monospace; font-size:.7rem !important; color:#6B7A99; word-break:break-all; margin-bottom:8px; }

/* ---------- Sidebar: CPU e hardware ---------- */
.cpu-name{ color:var(--ink); font-weight:700; line-height:1.3; margin:2px 0 8px; word-break:break-word; }
.cpu-name small{ display:block; color:#6B7A99; font-weight:600; font-size:.7rem !important; }
.tile.wide{ grid-column:1 / -1; }
.meter{ margin:2px 0 10px; }
.meter .lab{ display:flex; justify-content:space-between; color:#6B7A99; font-size:.72rem !important; margin-bottom:3px; }
.meter .trk{ height:7px; border-radius:99px; background:var(--line); overflow:hidden; }
.meter .trk i{ display:block; height:100%; border-radius:99px; background:var(--mint); }
.meter .trk i.warn{ background:var(--sun); }
.meter .trk i.hot{ background:var(--coral); }
.chips{ display:flex; flex-wrap:wrap; gap:4px; margin:2px 0 8px; }
.chips span{ background:#EEF5FF; border:1px solid var(--line); color:#2C5CC5; border-radius:6px; padding:1px 7px; font-weight:700; font-size:.68rem !important; }
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
    <div class="desc">Modelos Qwen2.5 de 0.5B e 1.5B treinados com Destilação de Conhecimento, RLAIF e DPO, comparados em cinco formatos GGUF, de FP16 a Q2_K.</div>
    <div class="by"><b>Fabyo Ramos</b><span>Pós-graduação em GenAI &amp; LLM Master, PUC-Rio</span></div>
  </div>
  <div class="viz">
    <div class="node n1"><div class="dot"><i></i></div><div><b>Teacher</b><br>14B / 72B</div></div>
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
        format_func=lambda k: MODELS_BY_KEY[k].label,  # rótulo estático
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
        max_msg_history = st.slider("Max message history", 1, 10, 5, 1)
        use_system = st.toggle(
            "Enviar system prompt",
            value=False,
            help="Desligado: testa se o treino (SFT/DPO) já alinhou o modelo sem precisar de instrução extra.",
        )

    # --- Ficha do modelo -----------------------------------------------------
    on_disk = spec.path.exists()
    badge = '<span class="badge ok">no disco</span>' if on_disk else '<span class="badge off">não baixado</span>'
    st.markdown(badge, unsafe_allow_html=True)
    st.markdown(lineage_card(spec), unsafe_allow_html=True)

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
    llm_messages = [{"role": m["role"], "content": m["content"]} for m in history[-max_msg_history:]]
    
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