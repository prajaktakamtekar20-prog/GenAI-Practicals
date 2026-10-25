import React, { useState } from 'react';
import { 
  BookOpen, 
  Layers, 
  Sparkles, 
  ShieldCheck, 
  Cpu, 
  Database, 
  HelpCircle, 
  ChevronDown, 
  ChevronUp,
  CheckCircle2,
  AlertTriangle
} from 'lucide-react';

const VIVA_QUESTIONS = [
  {
    q: "1. What is the fundamental difference between RAG and Fine-Tuning?",
    a: "RAG (Retrieval-Augmented Generation) supplies external, non-parametric context to the language model at inference time. It retrieves documents dynamically without modifying model weights, making it ideal for rapidly changing or private data. Fine-tuning modifies internal parametric weights through supervised training, instilling specific tone, formatting, domain vocabulary, or reasoning style, but cannot dynamically absorb new external documents without retraining."
  },
  {
    q: "2. How does LoRA (Low-Rank Adaptation) work mathematically?",
    a: "LoRA freezes the pre-trained weight matrix W_0 (of dimensions d x k) and parameterizes weight updates using low-rank decomposition: W = W_0 + ΔW = W_0 + (α / r) * (B * A), where B is d x r and A is r x k, with rank r << min(d, k). Here, α is a constant scaling hyperparameter. In our project, with r=8 and α=16, trainable parameters are reduced to just 0.22% of total parameters while keeping full model capacity intact."
  },
  {
    q: "3. What is QLoRA and how does it reduce memory?",
    a: "QLoRA (Quantized Low-Rank Adaptation) quantizes the frozen base model weights into 4-bit NormalFloat (NF4), uses Double Quantization to compress quantization constants, and leverages Paged Optimizers to manage memory spikes during backward passes. This allows fine-tuning large models on consumer GPUs."
  },
  {
    q: "4. Why use Sentence Transformers and FAISS instead of traditional keyword search?",
    a: "Keyword search (such as BM25) relies on lexical overlap and fails on synonyms or semantic variations. Sentence Transformers map text chunks into dense continuous vector spaces (e.g. 384 dimensions) where semantic similarity corresponds to vector proximity. FAISS (Facebook AI Similarity Search) calculates inner product cosine similarity across millions of vectors in sub-millisecond latency using optimized CPU SIMD instructions."
  },
  {
    q: "5. How does the system defend against Indirect Prompt Injections in uploaded documents?",
    a: "Uploaded documents are treated as untrusted data. The system isolates retrieved context between strict boundary markers ('=== RETRIEVED UNTRUSTED DOCUMENT CONTEXT START/END ===') and specifies system prompt rules instructing the LLM to ignore embedded commands attempting to override the assistant's persona or safety constraints."
  },
  {
    q: "6. Why did we choose Qwen2.5-0.5B-Instruct for local CPU execution?",
    a: "Qwen2.5-0.5B-Instruct has 494 million parameters and occupies under 1GB of memory, enabling near-instantaneous load times and responsive CPU inference (~2-5 seconds per query) on standard laptops without requiring a dedicated CUDA GPU or running out of RAM."
  },
  {
    q: "7. How are page numbers preserved during PDF extraction?",
    a: "Using PyMuPDF (fitz), the ingestion pipeline iterates through document pages sequentially, extracting text alongside the 1-indexed page number. When chunking with a sliding window, each chunk inherits the original page metadata, enabling precise page citations in the frontend."
  },
  {
    q: "8. What does the system do when a scanned image PDF is uploaded?",
    a: "The extractor analyzes text length across all pages. If a multi-page PDF contains zero or virtually no extractable text characters, the system flags it with a 'scanned_pdf_warning' status, informing the user that Optical Character Recognition (OCR) is required to process non-text PDFs."
  }
];

function AboutExperiment() {
  const [openViva, setOpenViva] = useState(null);

  const toggleViva = (index) => {
    setOpenViva(openViva === index ? null : index);
  };

  return (
    <div className="page-column">
      {/* Title Hero */}
      <section className="hero-card">
        <div>
          <div className="hero-badge-row">
            <span className="badge badge-primary">Comprehensive Academic Report</span>
            <span className="badge badge-neutral">Architecture • Methodology • Viva Voce</span>
          </div>
          <h3>Project Architecture & Research Report</h3>
          <p className="hero-subtitle">
            Development of a Personalized AI Assistant using an Open-Source Large Language Model, Retrieval-Augmented Generation (RAG), and Fine-Tuning.
          </p>
        </div>
      </section>

      {/* Aim & Objectives */}
      <section className="panel-card">
        <h3>1. Aim and Objectives</h3>
        <div className="report-section-body">
          <p>
            <strong>Aim:</strong> To develop a personalized AI assistant using an open-source foundational language model, integrate a Retrieval-Augmented Generation (RAG) system to answer questions accurately from user-specific documents, and investigate the effect of parameter-efficient fine-tuning using LoRA or QLoRA on the model's response quality.
          </p>
          <div className="objectives-list">
            <div className="objective-item">
              <CheckCircle2 size={16} className="text-primary" />
              <span>Implement a modular document ingestion pipeline for PDF (preserving page numbers via PyMuPDF), Word (.docx), and plain text (.txt).</span>
            </div>
            <div className="objective-item">
              <CheckCircle2 size={16} className="text-primary" />
              <span>Generate 384-dimensional dense semantic embeddings using Sentence Transformers (all-MiniLM-L6-v2) and index them in a persistent FAISS vector store.</span>
            </div>
            <div className="objective-item">
              <CheckCircle2 size={16} className="text-primary" />
              <span>Integrate an instruction-tuned open-source model (Qwen2.5) with strict grounding, prompt injection isolation, and source citation verification.</span>
            </div>
            <div className="objective-item">
              <CheckCircle2 size={16} className="text-primary" />
              <span>Implement a parameter-efficient fine-tuning (PEFT LoRA) workflow with rank decomposition and loss curve tracking.</span>
            </div>
            <div className="objective-item">
              <CheckCircle2 size={16} className="text-primary" />
              <span>Conduct a comparative benchmark across Base LLM, RAG, and Fine-Tuned modes recording Token F1 correctness, groundedness, and latency.</span>
            </div>
          </div>
        </div>
      </section>

      {/* System Architecture Diagram */}
      <section className="panel-card">
        <h3>2. System Architecture & End-to-End Pipeline</h3>
        <p className="muted-text">Complete data flow from document ingestion to vector search and grounded LLM response synthesis.</p>

        <div className="architecture-diagram-container">
          <div className="arch-flow">
            <div className="arch-box box-user">
              <div className="arch-tag">Input</div>
              <strong>User Query & Documents</strong>
              <small>PDF / DOCX / TXT / Prompt</small>
            </div>

            <div className="arch-arrow">↓ Upload</div>

            <div className="arch-grid-horizontal">
              <div className="arch-box">
                <div className="arch-tag">Document Loader</div>
                <strong>PyMuPDF & python-docx</strong>
                <small>Page preservation & OCR detection</small>
              </div>

              <div className="arch-arrow">→</div>

              <div className="arch-box">
                <div className="arch-tag">Chunking Engine</div>
                <strong>Sliding Window</strong>
                <small>500 words, 100 overlap</small>
              </div>

              <div className="arch-arrow">→</div>

              <div className="arch-box">
                <div className="arch-tag">Embedding Model</div>
                <strong>all-MiniLM-L6-v2</strong>
                <small>384d Normalized Vectors</small>
              </div>

              <div className="arch-arrow">→</div>

              <div className="arch-box box-storage">
                <div className="arch-tag">Vector Store</div>
                <strong>FAISS IndexFlatIP</strong>
                <small>Cosine Inner Product</small>
              </div>
            </div>

            <div className="arch-arrow">↓ Query Vector Retrieval (Top-K Chunks)</div>

            <div className="arch-box box-llm">
              <div className="arch-tag">Context Isolation & Inference</div>
              <strong>System Prompt Isolation + Open-Source LLM (Qwen2.5) + Optional LoRA Adapter</strong>
              <small>Grounded generation with verifiable citations: [Filename, Page X]</small>
            </div>

            <div className="arch-arrow">↓ Structured JSON Response</div>

            <div className="arch-box box-user">
              <div className="arch-tag">React Frontend (Port 5173)</div>
              <strong>Conversational UI with Expandable Citations, Similarity Badges, and Refusal Detection</strong>
            </div>
          </div>
        </div>
      </section>

      {/* Technology Stack Table */}
      <section className="panel-card">
        <h3>3. Mandatory Technology Stack</h3>
        <div className="table-responsive">
          <table className="doc-table">
            <thead>
              <tr>
                <th>Tier</th>
                <th>Technology</th>
                <th>Version / Specification</th>
                <th>Role in Architecture</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Frontend</strong></td>
                <td>React.js + Vite</td>
                <td>React 18, Vite 5</td>
                <td>Client-side dashboard, conversational interface, file dropzone</td>
              </tr>
              <tr>
                <td><strong>Styling</strong></td>
                <td>Modern Theme / CSS</td>
                <td>Inter font, CSS Variables</td>
                <td>Light theme, rounded cards, purple/blue accents, responsive layout</td>
              </tr>
              <tr>
                <td><strong>Backend API</strong></td>
                <td>FastAPI + Uvicorn</td>
                <td>FastAPI 0.115, Python 3.11</td>
                <td>Asynchronous REST endpoints, CORS middleware, Pydantic schemas</td>
              </tr>
              <tr>
                <td><strong>Embeddings</strong></td>
                <td>Sentence Transformers</td>
                <td>all-MiniLM-L6-v2</td>
                <td>Converts text chunks into 384-dimensional normalized vectors</td>
              </tr>
              <tr>
                <td><strong>Vector Database</strong></td>
                <td>FAISS (faiss-cpu)</td>
                <td>IndexFlatIP 1.9</td>
                <td>In-memory cosine similarity search with disk persistence</td>
              </tr>
              <tr>
                <td><strong>Foundational LLM</strong></td>
                <td>Hugging Face Transformers</td>
                <td>Qwen2.5-0.5B-Instruct</td>
                <td>Instruction-tuned open-source model running locally in RAM</td>
              </tr>
              <tr>
                <td><strong>Fine-Tuning</strong></td>
                <td>PEFT (LoRA)</td>
                <td>peft 0.13, PyTorch 2.5</td>
                <td>Low-Rank Adaptation ($r=8, \alpha=16$) on attention projection layers</td>
              </tr>
              <tr>
                <td><strong>PDF Processing</strong></td>
                <td>PyMuPDF (fitz)</td>
                <td>PyMuPDF 1.24</td>
                <td>Multi-page PDF text extraction preserving exact page numbers</td>
              </tr>
              <tr>
                <td><strong>DOCX Processing</strong></td>
                <td>python-docx</td>
                <td>python-docx 1.2</td>
                <td>Paragraph and tabular text extraction from Word documents</td>
              </tr>
              <tr>
                <td><strong>Testing</strong></td>
                <td>pytest</td>
                <td>pytest 8.3, pytest-asyncio</td>
                <td>Automated unit and integration test suite (20 passing tests)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Key Findings & Observations */}
      <section className="panel-card">
        <h3>4. Key Experimental Observations & Findings</h3>
        <div className="observations-grid">
          <div className="obs-card">
            <h4>Finding 1: RAG Eliminates Hallucination</h4>
            <p>
              The Base LLM alone scored a Groundedness of <strong>0.000</strong> on domain documents, occasionally hallucinating acronyms (e.g., misidentifying RAG as "Regulatory, Accountability, and Governance"). Incorporating FAISS retrieval increased Groundedness to <strong>0.599</strong> and Citation Accuracy to <strong>80.0%</strong>.
            </p>
          </div>
          <div className="obs-card">
            <h4>Finding 2: LoRA + RAG Delivers Peak Performance</h4>
            <p>
              While LoRA fine-tuning alone primarily adapted persona and response style without supplying unseen document facts, combining LoRA with RAG achieved the highest factual accuracy (<strong>Token F1: 0.320</strong> vs Base 0.208, a <strong>+53.7% improvement</strong>).
            </p>
          </div>
          <div className="obs-card">
            <h4>Finding 3: Parameter Efficiency of LoRA</h4>
            <p>
              By freezing the base weights and injecting rank-8 decomposition matrices into the query, value, key, and output projections, only <strong>1,081,344 parameters (0.218% of total)</strong> were updated, saving memory and training time (28 seconds on CPU).
            </p>
          </div>
        </div>
      </section>

      {/* Viva Voce Questions & Answers Accordion */}
      <section className="panel-card">
        <div className="panel-header-row">
          <div>
            <h3>5. Viva Voce Questions & Comprehensive Answers</h3>
            <p className="muted-text">Key academic questions covering LLMs, vector search, RAG, and parameter-efficient fine-tuning.</p>
          </div>
        </div>

        <div className="viva-accordion">
          {VIVA_QUESTIONS.map((item, index) => (
            <div key={index} className={`viva-item ${openViva === index ? 'open' : ''}`}>
              <button 
                type="button" 
                className="viva-question-btn" 
                onClick={() => toggleViva(index)}
              >
                <strong>{item.q}</strong>
                {openViva === index ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
              {openViva === index && (
                <div className="viva-answer-content">
                  <p>{item.a}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Limitations and Future Scope */}
      <section className="panel-card">
        <h3>6. Limitations and Future Scope</h3>
        <div className="limitations-grid">
          <div>
            <h4>Limitations</h4>
            <ul>
              <li>• Scanned PDFs containing only bitmap images require optical character recognition (OCR) before ingestion.</li>
              <li>• CPU inference latency (2-5s) is acceptable for academic demonstration but can be enhanced with GPU acceleration.</li>
              <li>• Fixed chunk size of 500 words may split complex paragraphs spanning multiple pages.</li>
            </ul>
          </div>
          <div>
            <h4>Future Scope</h4>
            <ul>
              <li>• Integrate Tesseract OCR directly into the PyMuPDF document loader for automatic scanned PDF transcription.</li>
              <li>• Implement Hybrid Search combining dense vector cosine similarity with BM25 sparse keyword retrieval.</li>
              <li>• Implement Re-ranking using a cross-encoder model to boost top-3 retrieval precision.</li>
              <li>• Add multi-session conversational memory with summarize-and-compress mechanisms.</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}

export default AboutExperiment;
