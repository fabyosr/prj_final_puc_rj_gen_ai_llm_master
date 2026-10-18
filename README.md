<!-- antes de enviar a versão final, solicitamos que todos os comentários, colocados para orientação ao aluno, sejam removidos do arquivo -->
# 5yo Model - Alinhamento de LLMs para Persona Infantil utilizando RLAIF, Chain of Thought (CoT), Destilação de Logits (KD) e Direct Preference Optimization (DPO)

#### Aluno: [Fabyo Sousa Ramos](https://github.com/fabyosr)
#### Orientador: 

---

Trabalho apresentado ao curso [GenAI e LLM MASTER](https://ica.ele.puc-rio.br/cursos/ia-generativa-large-language-models/) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

<!-- para os links a seguir, caso os arquivos estejam no mesmo repositório que este README, não há necessidade de incluir o link completo: basta incluir o nome do arquivo, com extensão, que o GitHub completa o link corretamente -->
- [Link para o código](https://github.com/fabyosr/prj_final_puc_rj_gen_ai_llm_master/blob/main/Child_Model_KNOWLEDGE_DISTILLATION_vFinal.ipynb). <!-- caso não aplicável, remover esta linha -->

---

### Resumo

<!-- trocar o texto abaixo pelo resumo do trabalho, em português -->

Este projeto apresenta o desenvolvimento e alinhamento de um modelo de linguagem (LLM) especializado em atuar como um assistente educacional seguro e lúdico para crianças de 5 anos. Utilizando a arquitetura Qwen2.5-3B como Estudante e o Qwen2.5-7B como Professor, o pipeline combina a geração de dados sintéticos estruturados com *Chain of Thought* (CoT), *Knowledge Distillation* (Destilação de Logits) e *Direct Preference Optimization* (DPO). O objetivo principal foi internalizar guardrails de segurança e adaptar a persona do modelo sem comprometer a eficiência computacional. O modelo final foi consolidado via *merge* de pesos LoRA e quantizado para o formato GGUF, permitindo sua execução local e off-line em dispositivos com recursos limitados.

### Abstract <!-- Opcional! Caso não aplicável, remover esta seção -->

<!-- trocar o texto abaixo pelo resumo do trabalho, em inglês -->

This project presents the development and alignment of a Large Language Model (LLM) specialized as a safe and playful educational assistant for 5-year-old children. Using the Qwen2.5-3B architecture as the Student and Qwen2.5-7B as the Teacher, the pipeline combines the generation of structured synthetic data with Chain of Thought (CoT), Knowledge Distillation (KD), and Direct Preference Optimization (DPO). The main objective was to internalize safety guardrails and adapt the model's persona without compromising computational efficiency. The final model was consolidated via LoRA weight merging and quantized to the GGUF format, enabling local, off-line execution on resource-constrained devices.

### 1. Introdução

Com o avanço rápido da IA Generativa, controlar a persona e a segurança das respostas (Guardrails) tornou-se um desafio essencial. Modelos genéricos frequentemente produzem respostas excessivamente complexas ou inadequadas para públicos sensíveis, como o infantil. Este projeto de conclusão de pós-graduação endereça esse problema construindo um pipeline de *Reinforcement Learning from AI Feedback* (RLAIF).
A justificativa central do trabalho baseia-se na necessidade de criar IA segura e acessível. Ao ensinar um modelo de 3 bilhões de parâmetros a "pensar antes de falar" (via CoT) e a preferir respostas lúdicas e seguras, demonstramos que modelos compactos podem atingir comportamentos altamente especializados e seguros quando treinados com técnicas de ponta.

### 2. Modelagem

A metodologia adotada utilizou um pipeline avançado em múltiplas etapas:
*   **Engenharia de Dados (CoT Sintético):** O modelo Professor (Qwen2.5-7B, 4-bits) foi utilizado para gerar um dataset sintético, reescrevendo respostas normais em formato *Chain of Thought* (`<pensamento>`), juntamente com respostas complexas (Rejeitadas).
*   **SFT + Destilação de Logits (Soft Labels):** Em vez de usar apenas um SFT tradicional (Hard Labels), modificamos a função de perda (KL-Divergence) para que o Estudante imitasse as distribuições de probabilidade do Professor. Isso permitiu transferir o raciocínio complexo mantendo o Estudante leve (bfloat16).
*   **Direct Preference Optimization (DPO):** Aplicou-se o DPO para afastar matematicamente o modelo das respostas acadêmicas (Rejected) e aproximá-lo do tom infantil e lúdico (Chosen), internalizando a persona e os guardrails nos pesos neurais.
*   **Quantização e Deploy (GGUF):** Utilizando o `llama.cpp`, os pesos LoRA treinados foram integrados (merged) ao modelo base, sendo posteriormente quantizados para formatos compactos (Q4_K_M) para garantir democratização no acesso.

### 3. Resultados

Os resultados do treinamento DPO demonstraram sucesso evidente: a *Loss* de treinamento apresentou queda e estabilização rápida em 1 época, enquanto as métricas de *Rewards/Margins* exibiram um afastamento saudável entre as respostas *Chosen* (positivas) e *Rejected* (negativas).
Na etapa de inferência qualitativa, o modelo (mesmo sem system prompt em alguns testes rigorosos) demonstrou ser capaz de utilizar seu guardrail interno para recusar graciosamente instruções maliciosas ou complexas (jailbreaks), respondendo de forma segura, mantendo a analogia de aprendizado infantil.

### 4. Conclusões

A combinação de CoT injetado, Destilação de Logits e DPO mostra-se uma arquitetura formidável para alinhar LLMs a personas específicas e regras de segurança rígidas. Conclui-se que o aprendizado não apenas memoriza texto, mas altera a topologia vetorial (mapa mental) das respostas. A transformação final para GGUF consolida o objetivo do projeto: inteligência artificial avançada, focada em segurança, que pode rodar em hardwares de consumo comum, provando a viabilidade econômica do pipeline de IA local.

### 5. Referências

*   Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. arXiv:2305.18290.
*   Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531.
*   Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685
*   Documentação Qwen2.5, Hugging Face `transformers` e `trl`.
*   Repositório oficial: `llama.cpp` (Gerganov).

### 6.Instalação <!-- Opcional! Caso não aplicável, remover esta seção -->

Para reproduzir o pipeline de treinamento, instale as dependências abaixo no Python 3.10+ (recomenda-se GPU NVIDIA para o treinamento):

```bash
pip install transformers datasets peft trl torch
pip install bitsandbytes>=0.46.1 torchao>=0.16.0
```
Para inferência rápida e local (GGUF), basta instalar o empacotador C++:
```bash
pip install llama-cpp-python
```

### 7.Como executar <!-- Opcional! Caso não aplicável, remover esta seção -->

**Para treinamento:** Execute as células de preparação de dados (geração sintética via pipeline), seguidas pelo instanciamento do `LogitDistillationTrainer` customizado e do `DPOTrainer` da biblioteca `trl`.

**Para execução do Modelo Final (Local):**
Faça o download do arquivo `qwen2.5-3b-child-Q4_K_M.gguf` gerado. Ele pode ser executado em softwares de interface gráfica como **LM Studio** ou **GPT4All**, ou através de código puro:

```python
from llama_cpp import Llama

llm = Llama(model_path="qwen2.5-3b-child-Q4_K_M.gguf")
resposta = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Por que chove?"}]
)
print(resposta['choices'][0]['message']['content'])
```
---

Matrícula: 252.100.183

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Inteligência Artificial Generativa & Large Language Models*
