import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  FileText, 
  Layers3, 
  Sparkles, 
  ArrowUpRight, 
  Cpu, 
  Database, 
  Bot, 
  BarChart3, 
  ShieldCheck,
  CheckCircle2,
  Clock
} from 'lucide-react';
import { getDocuments, getDocumentStats, getModelStatus, getEvaluationResults, getChatHistory } from '../api/client';

function Dashboard() {
  const navigate = useNavigate();
  const [stats, setStats] = useState({
    documents: 0,
    chunks: 0,
    pages: 0,
    modelName: 'Qwen/Qwen2.5-0.5B-Instruct',
    device: 'cpu',
    ramGb: 0,
    hasAdapter: false,
    evalCompleted: false,
    avgLatency: '4.8s',
    recentChats: []
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [docStats, modelStatus, evalResults, historyData] = await Promise.allSettled([
          getDocumentStats(),
          getModelStatus(),
          getEvaluationResults(),
          getChatHistory()
        ]);

        const ds = docStats.status === 'fulfilled' ? docStats.value : {};
        const ms = modelStatus.status === 'fulfilled' ? modelStatus.value : {};
        const es = evalResults.status === 'fulfilled' ? evalResults.value : {};
        const hs = historyData.status === 'fulfilled' ? historyData.value : {};

        let avgLat = '4.9s';
        if (es.mode_summaries && es.mode_summaries.length > 0) {
          const ragSummary = es.mode_summaries.find(m => m.mode === 'rag');
          if (ragSummary) {
            avgLat = `${(ragSummary.average_latency_ms / 1000).toFixed(1)}s`;
          }
        }

        setStats({
          documents: ds.total_documents ?? 0,
          chunks: ds.vector_stats?.total_chunks ?? 0,
          pages: ds.total_pages ?? 0,
          modelName: ms.model_name || 'Qwen/Qwen2.5-0.5B-Instruct',
          device: ms.device || 'CPU',
          ramGb: ms.system_ram_available_gb || 0,
          hasAdapter: ms.loaded_adapter ? true : false,
          evalCompleted: es.status === 'completed',
          avgLatency: avgLat,
          recentChats: (hs.history || []).slice(-4).reverse()
        });
      } catch (err) {
        console.error('Failed to load dashboard data:', err);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="page-column">
      {/* Project Hero Card */}
      <section className="hero-card">
        <div className="hero-content">
          <div className="hero-badge-row">
            <span className="badge badge-primary">M.Tech / B.Tech Major Project</span>
            <span className="badge badge-neutral">FAISS + PyTorch + React + FastAPI</span>
          </div>
          <h3>Development of a Personalized AI Assistant</h3>
          <p className="hero-subtitle">
            Using an Open-Source Large Language Model, Retrieval-Augmented Generation (RAG), and Parameter-Efficient Fine-Tuning (LoRA/QLoRA).
          </p>
          <p className="muted-text">
            <strong>Aim:</strong> To develop a local personalized AI assistant using an open-source foundational model, integrate a RAG pipeline to answer questions from user documents with verifiable citations, and evaluate the effect of LoRA fine-tuning on response accuracy and domain groundedness.
          </p>
        </div>

        <div className="hero-actions">
          <button className="primary-button" onClick={() => navigate('/assistant')}>
            Open Assistant <ArrowUpRight size={16} />
          </button>
          <button className="ghost-button" onClick={() => navigate('/knowledge-base')}>
            Manage Knowledge Base
          </button>
        </div>
      </section>

      {/* Metrics Cards Grid */}
      <section className="stats-grid">
        <div className="stat-card">
          <div className="icon-box icon-box-blue">
            <FileText size={20} />
          </div>
          <div>
            <p className="stat-label">Uploaded Documents</p>
            <strong className="stat-value">{stats.documents}</strong>
            <span className="stat-detail">{stats.pages} total pages extracted</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="icon-box icon-box-purple">
            <Layers3 size={20} />
          </div>
          <div>
            <p className="stat-label">Indexed Vector Chunks</p>
            <strong className="stat-value">{stats.chunks}</strong>
            <span className="stat-detail">FAISS 384d Cosine Similarity</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="icon-box icon-box-green">
            <Sparkles size={20} />
          </div>
          <div>
            <p className="stat-label">Foundational LLM</p>
            <strong className="stat-value text-sm">{stats.modelName.split('/')[1] || stats.modelName}</strong>
            <span className="stat-detail">Device: {stats.device.toUpperCase()} • {stats.ramGb} GB RAM Free</span>
          </div>
        </div>
      </section>

      {/* Operating Modes Breakdown */}
      <section className="panel-card">
        <div className="panel-header-row">
          <div>
            <h3>Three Operating Modes Investigated</h3>
            <p className="muted-text">Compare non-parametric retrieval against parametric weights adaptation.</p>
          </div>
          <button className="ghost-button text-sm" onClick={() => navigate('/model-comparison')}>
            View Comparative Benchmark <ArrowUpRight size={14} />
          </button>
        </div>

        <div className="modes-overview-grid">
          <div className="mode-info-card">
            <div className="mode-card-header">
              <span className="badge badge-neutral">Mode 1</span>
              <h4>Base Foundational LLM</h4>
            </div>
            <p>
              Answers user questions using only pre-trained internal weights. Lacks access to your private documents; may exhibit hallucination or knowledge cutoffs.
            </p>
            <ul className="mode-features-list">
              <li>• Pure parametric inference</li>
              <li>• Zero document retrieval</li>
              <li>• Baseline for comparative study</li>
            </ul>
          </div>

          <div className="mode-info-card mode-info-highlight">
            <div className="mode-card-header">
              <span className="badge badge-primary">Mode 2</span>
              <h4>RAG-Enhanced LLM</h4>
            </div>
            <p>
              Dynamically retrieves relevant chunks from FAISS vector store. Grounds the model in verifiable evidence and cites filenames and page numbers.
            </p>
            <ul className="mode-features-list">
              <li>• Dynamic non-parametric context</li>
              <li>• Verifiable page citations</li>
              <li>• Defends against prompt injection</li>
            </ul>
          </div>

          <div className="mode-info-card">
            <div className="mode-card-header">
              <span className="badge badge-secondary">Mode 3</span>
              <h4>LoRA Fine-Tuned Model</h4>
            </div>
            <p>
              Adapts foundational model weights using Low-Rank Adaptation (PEFT). Learns specialized domain vocabulary, refusal behaviors, and assistant style.
            </p>
            <ul className="mode-features-list">
              <li>• Parameter-efficient adapter ($r=8$)</li>
              <li>• 0.22% trainable parameters</li>
              <li>• Combined with RAG for peak accuracy</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Quick Actions & Recent Sessions */}
      <div className="dashboard-columns-grid">
        <section className="panel-card">
          <h3>Quick Project Actions</h3>
          <div className="action-buttons-stack">
            <button className="action-row-btn" onClick={() => navigate('/knowledge-base')}>
              <Database size={18} className="text-primary" />
              <div className="action-row-text">
                <strong>Upload & Ingest Documents</strong>
                <p>Add PDF, DOCX, or TXT files to build your personalized knowledge base.</p>
              </div>
              <ArrowUpRight size={16} className="chevron" />
            </button>

            <button className="action-row-btn" onClick={() => navigate('/assistant')}>
              <Bot size={18} className="text-secondary" />
              <div className="action-row-text">
                <strong>Query Document-Grounded Assistant</strong>
                <p>Interact with the conversational interface and inspect citation excerpts.</p>
              </div>
              <ArrowUpRight size={16} className="chevron" />
            </button>

            <button className="action-row-btn" onClick={() => navigate('/model-comparison')}>
              <BarChart3 size={18} className="text-success" />
              <div className="action-row-text">
                <strong>Model Comparison & Benchmark</strong>
                <p>View real quantitative metrics (Token F1, Groundedness, Latency) across modes.</p>
              </div>
              <ArrowUpRight size={16} className="chevron" />
            </button>
          </div>
        </section>

        <section className="panel-card">
          <div className="panel-header-row">
            <h3>Recent Assistant Activity</h3>
            {stats.recentChats.length > 0 && (
              <button className="ghost-button text-xs" onClick={() => navigate('/assistant')}>
                View All
              </button>
            )}
          </div>

          {stats.recentChats.length === 0 ? (
            <div className="empty-subtle">
              <Clock size={28} className="text-muted" />
              <p>No recent questions recorded yet.</p>
              <button className="primary-button text-xs" onClick={() => navigate('/assistant')}>
                Start First Conversation
              </button>
            </div>
          ) : (
            <div className="recent-messages-list">
              {stats.recentChats.map((msg, i) => (
                <div key={i} className="recent-message-item">
                  <div className="recent-msg-header">
                    <span className="badge badge-neutral text-xs">
                      {msg.role === 'user' ? 'Question' : 'Response'}
                    </span>
                    {msg.mode && (
                      <span className="mode-pill-subtle">{msg.mode.toUpperCase()}</span>
                    )}
                  </div>
                  <p className="recent-msg-text">{msg.content.slice(0, 110)}...</p>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

export default Dashboard;
