import React, { useState, useEffect } from 'react';
import DocumentUploader from '../components/DocumentUploader';
import DocumentList from '../components/DocumentList';
import { getDocuments, getDocumentStats } from '../api/client';
import { Database, Info, ShieldAlert, Sparkles } from 'lucide-react';

function KnowledgeBase() {
  const [documents, setDocuments] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchKnowledgeBase = async () => {
    try {
      setLoading(true);
      const [docsRes, statsRes] = await Promise.all([
        getDocuments(),
        getDocumentStats(),
      ]);
      setDocuments(docsRes.documents || []);
      setStats(statsRes);
    } catch (err) {
      console.error('Failed to load knowledge base:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchKnowledgeBase();
  }, []);

  return (
    <div className="page-column">
      {/* Knowledge Base Header */}
      <section className="hero-card">
        <div>
          <div className="hero-badge-row">
            <span className="badge badge-primary">RAG Vector Storage</span>
            <span className="badge badge-neutral">PyMuPDF • python-docx • FAISS</span>
          </div>
          <h3>Knowledge Base Management</h3>
          <p className="muted-text">
            Ingest user-specific documents, extract text per page, partition into overlapping chunks, and build local FAISS vector embeddings for semantic retrieval.
          </p>
        </div>
      </section>

      {/* Upload Component */}
      <section className="panel-card">
        <div className="panel-header-row">
          <div>
            <h3>Upload New Documents</h3>
            <p className="muted-text">Documents are stored locally in <code>backend/storage</code> and never sent to external cloud APIs.</p>
          </div>
        </div>

        <DocumentUploader onUploadSuccess={fetchKnowledgeBase} />
      </section>

      {/* Document List and Vector Stats */}
      <section className="panel-card">
        <DocumentList 
          documents={documents} 
          stats={stats} 
          onRefresh={fetchKnowledgeBase} 
        />
      </section>

      {/* Ingestion & Scanned PDF Notice */}
      <section className="info-callout-card">
        <div className="callout-header">
          <Info size={18} className="text-primary" />
          <h4>Document Processing & Scanned PDF Guidelines</h4>
        </div>
        <div className="callout-grid">
          <div>
            <strong>1. PDF Extraction (PyMuPDF):</strong>
            <p>PDFs are loaded page-by-page. Page numbers are preserved throughout chunking to ensure source citations reflect exact original document pages.</p>
          </div>
          <div>
            <strong>2. Scanned PDFs & OCR:</strong>
            <p>If a PDF is composed entirely of scanned photos or flattened images, no selectable text stream exists. The system flags these with a warning indicating that optical character recognition (OCR) is required.</p>
          </div>
          <div>
            <strong>3. Sliding Window Chunking:</strong>
            <p>Text is split into 500-word segments with 100-word overlap. Overlap preserves contextual continuity between adjacent sentences.</p>
          </div>
          <div>
            <strong>4. FAISS Vector Persistence:</strong>
            <p>Normalized embeddings are generated using <code>sentence-transformers/all-MiniLM-L6-v2</code> and written to disk (<code>vector_index.faiss</code>) for zero-latency restarts.</p>
          </div>
        </div>
      </section>
    </div>
  );
}

export default KnowledgeBase;
