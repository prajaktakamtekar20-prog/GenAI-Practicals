import React, { useState } from 'react';
import { 
  FileText, 
  Trash2, 
  Eye, 
  RefreshCw, 
  Layers, 
  CheckCircle, 
  AlertTriangle, 
  Clock, 
  X,
  FileSpreadsheet
} from 'lucide-react';
import { deleteDocument, indexDocuments } from '../api/client';

function DocumentList({ documents, stats, onRefresh }) {
  const [indexing, setIndexing] = useState(false);
  const [indexResult, setIndexResult] = useState(null);
  const [previewDoc, setPreviewDoc] = useState(null);
  const [deletingId, setDeletingId] = useState(null);

  const handleIndex = async () => {
    try {
      setIndexing(true);
      setIndexResult(null);
      const res = await indexDocuments();
      setIndexResult(res);
      if (onRefresh) onRefresh();
    } catch (err) {
      console.error('Indexing failed:', err);
      alert('Indexing failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setIndexing(false);
    }
  };

  const handleDelete = async (docId, filename) => {
    if (!window.confirm(`Are you sure you want to remove "${filename}" from your knowledge base?`)) {
      return;
    }
    try {
      setDeletingId(docId);
      await deleteDocument(docId);
      if (onRefresh) onRefresh();
    } catch (err) {
      console.error('Delete failed:', err);
      alert('Delete failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setDeletingId(null);
    }
  };

  const formatSize = (bytes) => {
    if (!bytes) return '0 KB';
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const getStatusBadge = (doc) => {
    if (doc.processing_status === 'scanned_pdf_warning') {
      return <span className="badge badge-warning" title={doc.notes}><AlertTriangle size={12} /> Scanned (OCR Needed)</span>;
    }
    if (doc.processing_status === 'indexed') {
      return <span className="badge badge-success"><CheckCircle size={12} /> Indexed</span>;
    }
    return <span className="badge badge-info"><Clock size={12} /> Ready to Index</span>;
  };

  return (
    <div className="document-list-panel">
      <div className="panel-header-row">
        <div>
          <h3>Uploaded Documents ({documents.length})</h3>
          <p className="muted-text">Managed files stored locally and available for RAG vectorization.</p>
        </div>

        <div className="header-actions-group">
          <button 
            type="button" 
            className="primary-button" 
            onClick={handleIndex}
            disabled={indexing || documents.length === 0}
          >
            <RefreshCw size={15} className={indexing ? 'spin' : ''} />
            {indexing ? 'Indexing Chunks...' : 'Re-Index Knowledge Base'}
          </button>
        </div>
      </div>

      {indexResult && (
        <div className="alert-box alert-success">
          <CheckCircle size={16} className="alert-icon" />
          <span>
            Knowledge base re-indexed successfully: <strong>{indexResult.chunk_count}</strong> chunks generated across <strong>{indexResult.document_count}</strong> files.
          </span>
        </div>
      )}

      {/* Vector Stats Summary Bar */}
      <div className="stats-summary-bar">
        <div className="summary-item">
          <Layers size={16} className="text-primary" />
          <span className="summary-label">FAISS Chunks:</span>
          <strong>{stats?.vector_stats?.total_chunks ?? 0}</strong>
        </div>
        <div className="summary-item">
          <FileText size={16} className="text-secondary" />
          <span className="summary-label">Indexed Documents:</span>
          <strong>{stats?.vector_stats?.unique_documents ?? 0}</strong>
        </div>
        <div className="summary-item">
          <FileSpreadsheet size={16} className="text-muted" />
          <span className="summary-label">Vector Dimension:</span>
          <strong>{stats?.vector_stats?.dimension || 384}d (Cosine IP)</strong>
        </div>
      </div>

      {documents.length === 0 ? (
        <div className="empty-state-card">
          <FileText size={36} className="empty-icon" />
          <h4>No documents in knowledge base yet</h4>
          <p>Upload your research papers, notes, or project documentation above to get started.</p>
        </div>
      ) : (
        <div className="table-responsive">
          <table className="doc-table">
            <thead>
              <tr>
                <th>Document Name</th>
                <th>Format</th>
                <th>File Size</th>
                <th>Pages</th>
                <th>Indexed Chunks</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr key={doc.document_id}>
                  <td className="doc-name-cell">
                    <FileText size={16} className="doc-icon" />
                    <span className="doc-title" title={doc.filename}>{doc.filename}</span>
                  </td>
                  <td>
                    <span className="filetype-badge">{doc.file_type?.toUpperCase().replace('.', '')}</span>
                  </td>
                  <td>{formatSize(doc.size_bytes)}</td>
                  <td>{doc.page_count ?? 1}</td>
                  <td>
                    <span className="chunk-count-badge">
                      {doc.chunk_count > 0 ? doc.chunk_count : 'Not indexed'}
                    </span>
                  </td>
                  <td>{getStatusBadge(doc)}</td>
                  <td className="action-buttons-cell">
                    <button 
                      className="icon-btn" 
                      onClick={() => setPreviewDoc(doc)}
                      title="View extracted text preview"
                    >
                      <Eye size={15} />
                    </button>
                    <button 
                      className="icon-btn text-danger" 
                      onClick={() => handleDelete(doc.document_id, doc.filename)}
                      disabled={deletingId === doc.document_id}
                      title="Remove document"
                    >
                      <Trash2 size={15} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Extracted Text Preview Modal */}
      {previewDoc && (
        <div className="modal-overlay" onClick={() => setPreviewDoc(null)}>
          <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h4>Extracted Text Preview: {previewDoc.filename}</h4>
                <p className="modal-sub">
                  Page count: {previewDoc.page_count} • Size: {formatSize(previewDoc.size_bytes)}
                </p>
              </div>
              <button className="icon-btn" onClick={() => setPreviewDoc(null)}>
                <X size={18} />
              </button>
            </div>

            <div className="modal-body">
              {previewDoc.notes && (
                <div className="alert-box alert-warning">
                  <AlertTriangle size={16} className="alert-icon" />
                  <span>{previewDoc.notes}</span>
                </div>
              )}
              <pre className="text-preview-box">
                {previewDoc.extracted_text_preview || 'No text could be extracted from this document.'}
              </pre>
            </div>

            <div className="modal-footer">
              <button className="ghost-button" onClick={() => setPreviewDoc(null)}>
                Close Preview
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default DocumentList;
