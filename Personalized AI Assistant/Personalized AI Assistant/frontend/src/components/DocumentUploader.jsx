import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, CheckCircle2, AlertTriangle, AlertCircle, Loader2 } from 'lucide-react';
import { uploadDocument } from '../api/client';

const ALLOWED_EXTS = ['.pdf', '.docx', '.txt'];
const MAX_SIZE_MB = 10;

function DocumentUploader({ onUploadSuccess }) {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUploaded, setLastUploaded] = useState(null);
  const fileInputRef = useRef(null);

  const validateFile = (file) => {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTS.includes(ext)) {
      return `Unsupported file format (${ext}). Supported formats: PDF, DOCX, TXT.`;
    }
    if (file.size === 0) {
      return 'File is empty (0 bytes).';
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      return `File exceeds maximum allowed size of ${MAX_SIZE_MB} MB.`;
    }
    return null;
  };

  const handleProcessFile = async (file) => {
    setError(null);
    setLastUploaded(null);
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    try {
      setUploading(true);
      const result = await uploadDocument(file);
      setLastUploaded(result);
      if (onUploadSuccess) onUploadSuccess(result);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Upload failed';
      setError(`Upload error: ${msg}`);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleProcessFile(e.target.files[0]);
    }
  };

  return (
    <div className="uploader-card">
      <div
        className={`dropzone ${dragActive ? 'drag-active' : ''} ${uploading ? 'uploading-state' : ''}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => !uploading && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          onChange={handleChange}
          style={{ display: 'none' }}
        />

        <div className="dropzone-inner">
          <div className="dropzone-icon-circle">
            {uploading ? (
              <Loader2 size={28} className="spin text-primary" />
            ) : (
              <UploadCloud size={28} className="text-primary" />
            )}
          </div>

          <h4>{uploading ? 'Ingesting & Extracting Text...' : 'Drag & Drop your research documents here'}</h4>
          <p className="dropzone-sub">
            Supported formats: <strong>PDF</strong> (page-indexed via PyMuPDF), <strong>DOCX</strong> (python-docx), and <strong>TXT</strong>
          </p>
          <span className="file-limit-pill">Up to 10 MB per document • Pure Local Storage</span>

          <button
            type="button"
            className="primary-button dropzone-button"
            disabled={uploading}
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
          >
            {uploading ? 'Processing File...' : 'Browse Computer Files'}
          </button>
        </div>
      </div>

      {error && (
        <div className="alert-box alert-error">
          <AlertCircle size={18} className="alert-icon" />
          <span>{error}</span>
        </div>
      )}

      {lastUploaded && (
        <div className={`alert-box ${lastUploaded.processing_status === 'scanned_pdf_warning' ? 'alert-warning' : 'alert-success'}`}>
          {lastUploaded.processing_status === 'scanned_pdf_warning' ? (
            <>
              <AlertTriangle size={18} className="alert-icon" />
              <div>
                <strong>Uploaded with OCR Warning:</strong> {lastUploaded.filename}
                <p className="alert-subtext">{lastUploaded.notes}</p>
              </div>
            </>
          ) : (
            <>
              <CheckCircle2 size={18} className="alert-icon" />
              <div>
                <strong>Successfully Ingested:</strong> {lastUploaded.filename} ({Math.round(lastUploaded.size_bytes / 1024)} KB, {lastUploaded.page_count} pages)
                <p className="alert-subtext">Click <em>Index Documents</em> to generate vector embeddings for RAG retrieval.</p>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default DocumentUploader;
