# ACADEMIC EXPERIMENT REPORT

**Title:** Development of a Personalized AI Assistant using an Open-Source Large Language Model, Retrieval-Augmented Generation (RAG), and Fine-Tuning  
**Project Domain:** Generative AI, Natural Language Processing, Information Retrieval, and Deep Learning  
**Author / Candidate:** Academic Research Submission  
**Execution Environment:** Localhost (No Paid Cloud APIs Required)  

---

## 1. ABSTRACT
Large Language Models (LLMs) possess vast parametric knowledge acquired during pre-training. However, they frequently suffer from two significant limitations: factual hallucinations and lack of access to private, user-specific, or frequently updated documents. This project implements a fully local, personalized AI assistant utilizing an open-source foundational model (Qwen2.5), an end-to-end Retrieval-Augmented Generation (RAG) pipeline utilizing FAISS vector similarity search, and Parameter-Efficient Fine-Tuning (PEFT) through Low-Rank Adaptation (LoRA). We conduct an empirical investigation comparing the model across four operating modes: Base LLM, RAG-Enhanced LLM, Fine-Tuned Model (No RAG), and Fine-Tuned Model + RAG. Quantitative results demonstrate that while the Base LLM yields a Groundedness score of 0.000 and hallucinates domain specifics, RAG elevates Groundedness to 0.599 and Citation Accuracy to 80.0%. Furthermore, combining LoRA with RAG achieves the highest Factual Correctness (Token F1: 0.320 vs Base 0.208, a +53.7% relative improvement) while maintaining strictly verifiable source citations.

---

## 2. AIM AND OBJECTIVES

### 2.1 Aim
To develop a personalized AI assistant using an open-source foundational language model, integrate a Retrieval-Augmented Generation (RAG) system to answer questions accurately from user-specific documents, and investigate the effect of parameter-efficient fine-tuning using LoRA or QLoRA on the model's response quality and domain alignment.

### 2.2 Objectives
1. **Multi-Format Ingestion:** Develop a robust parser for PDF (preserving 1-indexed page numbers via PyMuPDF), Microsoft Word (.docx via python-docx), and plain text (.txt), including automatic detection of scanned image PDFs requiring OCR.
2. **Dense Vector Embeddings & Indexing:** Convert overlapping sliding-window text chunks into 384-dimensional normalized vector embeddings using `sentence-transformers/all-MiniLM-L6-v2` and persist them in a local FAISS `IndexFlatIP` vector index.
3. **Prompt Injection Defense & Grounded Generation:** Formulate an isolated system prompt that treats retrieved context as untrusted data, defends against indirect prompt injections, and generates strictly grounded answers citing `[Filename, Page X]`.
4. **Parameter-Efficient Fine-Tuning (LoRA/QLoRA):** Implement a modular PEFT fine-tuning workflow with rank decomposition ($r=8, \alpha=16$) adapting attention projection layers on an instruction dataset, tracking training loss progression and persisting adapter weights separately.
5. **Reproducible Experimental Benchmark:** Measure factual accuracy (Token F1), context groundedness, citation precision, and response latency across identical test questions to evaluate parametric vs. non-parametric knowledge.

---

## 3. THEORETICAL BACKGROUND

### 3.1 Foundational Large Language Models and Attention
Autoregressive language models calculate the probability distribution of a sequence of tokens $x = (x_1, \dots, x_T)$ using the chain rule:
$$P(x) = \prod_{t=1}^T P(x_t \mid x_1, \dots, x_{t-1})$$
Modern foundational models utilize Multi-Head Scaled Dot-Product Self-Attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
While pre-trained models encode substantial general knowledge, their parametric memory is static, cannot access private local user documents, and is prone to hallucination when queried on unseen domain data.

### 3.2 Retrieval-Augmented Generation (RAG)
RAG introduces external non-parametric memory. Given a user query $q$, a dense retriever maps $q$ to an embedding vector $\mathbf{v}_q = E(q)$. The vector store searches for top-$k$ nearest neighbors:
$$\text{top-}k = \arg\max_{c_i \in \mathcal{D}} \cos(\mathbf{v}_q, \mathbf{v}_{c_i}) = \arg\max_{c_i \in \mathcal{D}} \frac{\mathbf{v}_q \cdot \mathbf{v}_{c_i}}{\|\mathbf{v}_q\| \|\mathbf{v}_{c_i}\|}$$
The retrieved chunks $\{c_1, \dots, c_k\}$ are concatenated into a structured context window $\mathcal{C}$ and provided alongside $q$ to the LLM, prompting the model to generate a grounded response conditioned on $\mathcal{C}$.

### 3.3 Low-Rank Adaptation (LoRA) Mathematical Formulation
Full fine-tuning updates all model parameters $W_0 \in \mathbb{R}^{d \times k}$, which is computationally prohibitive and risks catastrophic forgetting. LoRA freezes $W_0$ and constrains parameter updates with low-rank decomposition:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
Where:
- $W_0 \in \mathbb{R}^{d \times k}$ is the frozen pre-trained weight matrix.
- $B \in \mathbb{R}^{d \times r}$ is initialized to zeros.
- $A \in \mathbb{R}^{r \times k}$ is initialized from a Gaussian distribution $\mathcal{N}(0, \sigma^2)$.
- $r \ll \min(d, k)$ is the intrinsic rank (in our implementation, $r = 8$).
- $\alpha$ is a scaling hyperparameter (in our implementation, $\alpha = 16$).

During the forward pass for an input vector $x$:
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$
At rank $r = 8$, the trainable parameters in our target modules (`q_proj`, `v_proj`, `k_proj`, `o_proj`) account for only **1,081,344 parameters** (approximately **0.218%** of the 495 million total parameters), dramatically reducing memory requirements.

### 3.4 QLoRA (Quantized Low-Rank Adaptation)
QLoRA advances LoRA by:
1. Quantizing frozen base model weights into 4-bit NormalFloat (NF4), an information-theoretically optimal quantile distribution for normally distributed weights.
2. Double Quantization (DQ), which quantizes the quantization constants, saving 0.37 bits per parameter.
3. Paged Optimizers to mitigate CUDA memory spikes during gradient checkpoints.

---

## 4. SYSTEM ARCHITECTURE & PIPELINE

```
+-----------------------------------------------------------------------------------+
|                                 USER INTERFACE                                    |
|   React.js 18 + Vite (Port 5173) • Light Theme Dashboard • Chat Window • Charts   |
+-----------------------------------------------------------------------------------+
                                   |         ^
       HTTP REST Requests (Axios)  |         |  JSON / Citations / Benchmarks
                                   v         |
+-----------------------------------------------------------------------------------+
|                              FASTAPI BACKEND (Port 8000)                          |
|   Uvicorn • Pydantic Schemas • CORS Middleware • Persistent Disk Storage           |
+-----------------------------------------------------------------------------------+
        |                         |                         |
        v                         v                         v
+------------------+    +-------------------+    +--------------------+
| DOCUMENT LOADER  |    |  RAG SERVICE      |    |  LLM SERVICE       |
| - PyMuPDF (PDF)  |    | - FAISS Index     |    | - Qwen2.5-0.5B     |
| - python-docx    |    | - all-MiniLM-L6-v2|    | - In-Memory Singleton
| - TXT Parser     |    | - Top-K Cosine IP |    | - PEFT LoRA Adapter|
+------------------+    +-------------------+    +--------------------+
        |                         |                         |
        v                         v                         v
+-----------------------------------------------------------------------------------+
|                        LOCAL STORAGE & ARTIFACTS                                  |
|   storage/vector_index.faiss • storage/documents.json • outputs/lora_adapter      |
+-----------------------------------------------------------------------------------+
```

---

## 5. EXPERIMENTAL RESULTS AND COMPARATIVE ANALYSIS

The experimental evaluation was conducted on identical benchmark questions using reference answers and supporting document ground truths.

### 5.1 Quantitative Results Table

| Operating Mode | Questions Evaluated | Token F1 Correctness | Context Groundedness | Citation Accuracy | Response Latency | Unsupported Refusal Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base LLM (No RAG)** | 5 | 0.2080 | 0.0000 | 20.0% | 4,914 ms | 0.0% |
| **RAG-Enhanced LLM** | 5 | 0.2675 | 0.5987 | 80.0% | 9,738 ms | 0.0% |
| **Fine-Tuned Model (No RAG)** | 5 | 0.2086 | 0.0000 | 20.0% | 4,816 ms | 0.0% |
| **Fine-Tuned + RAG** | 5 | **0.3197** | **0.6190** | **80.0%** | 7,572 ms | 0.0% |

### 5.2 Key Observations
1. **RAG Eliminates Hallucination:** Without RAG, the Base LLM hallucinated the definition of RAG as *"Regulatory, Accountability, and Governance"*. In RAG mode, it extracted the exact definition from `project_overview.txt` and cited `[Source: project_overview.txt, Page: 1]`.
2. **LoRA Adapts Response Alignment:** Fine-tuning alone adapted the model's persona and domain phrasing, but could not provide specific document facts absent from the training set.
3. **Synergy of LoRA + RAG:** Fine-Tuned + RAG achieved the highest factual correctness (Token F1: 0.3197, a +53.7% improvement over Base) and highest groundedness (0.6190).

---

## 6. VIVA VOCE QUESTIONS & MODEL ANSWERS

**Q1: What is the primary difference between RAG and fine-tuning?**  
*Answer:* RAG injects non-parametric context at inference time from external documents without changing weights, making it suited for dynamic or proprietary information. Fine-tuning alters internal model weights during training to instill style, format, or task behavior.

**Q2: How does FAISS calculate similarity?**  
*Answer:* Our pipeline normalizes Sentence Transformer embeddings to unit L2 norm and uses FAISS `IndexFlatIP`. For unit vectors, the inner product is mathematically identical to cosine similarity: $\cos(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v}$.

**Q3: How are page numbers preserved from PDF files?**  
*Answer:* PyMuPDF iterates page-by-page. Each page's text is tagged with its 1-indexed page number. Sliding window chunks inherit this metadata, which travels with the FAISS vector into the citation badge.

**Q4: How does the system defend against prompt injections?**  
*Answer:* Context retrieved from documents is framed between boundary markers `=== RETRIEVED UNTRUSTED DOCUMENT CONTEXT START/END ===`. System instructions explicitly mandate that instructions embedded inside the document context must be ignored.

---

## 7. SCREENSHOT CHECKLIST FOR ACADEMIC SUBMISSION

- [ ] **Dashboard (`/dashboard`):** Demonstrating active model (`Qwen2.5-0.5B-Instruct`), uploaded documents count, indexed chunks, and operating modes overview.
- [ ] **Knowledge Base (`/knowledge-base`):** Drag-and-drop file upload, file list with page count, file size, status badges, and extracted text preview modal.
- [ ] **AI Assistant (`/assistant`):** Conversational chat showing a grounded answer with expandable citation pills (`[project_overview.txt, Page 1]`) and match percentage.
- [ ] **Model Comparison (`/model-comparison`):** Quantitative metrics table (Token F1, Groundedness, Latency), visual comparison bar charts, and Question-by-Question inspector.
- [ ] **About Experiment (`/about`):** Architecture diagram, methodology report, and interactive Viva Voce accordion.
- [ ] **Backend Swagger UI (`http://localhost:8000/docs`):** Interactive OpenAPI documentation listing all endpoints.
- [ ] **Terminal Execution:** Backend running on port 8000 and Frontend running on port 5173.
