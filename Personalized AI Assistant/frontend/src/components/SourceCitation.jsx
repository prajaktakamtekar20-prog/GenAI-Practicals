import React, { useState } from 'react';
import { FileText, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';

function SourceCitation({ source, index }) {
  const [expanded, setExpanded] = useState(false);

  const filename = source.filename || 'Document';
  const page = source.page_number;
  const score = source.score != null ? Math.round(source.score * 100) : null;
  const excerpt = source.excerpt || '';

  return (
    <div className={`citation-card ${expanded ? 'expanded' : ''}`}>
      <button 
        type="button" 
        className="citation-trigger" 
        onClick={() => setExpanded(!expanded)}
      >
        <div className="citation-header-left">
          <FileText size={14} className="citation-icon" />
          <span className="citation-filename">{filename}</span>
          {page != null && (
            <span className="citation-page">Page {page}</span>
          )}
        </div>

        <div className="citation-header-right">
          {score != null && (
            <span className="citation-score">
              {score}% match
            </span>
          )}
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </div>
      </button>

      {expanded && (
        <div className="citation-body">
          <div className="citation-meta-row">
            <span className="badge badge-soft">Chunk #{source.chunk_index ?? index}</span>
            <span className="citation-hint">Exact excerpt passed to LLM context:</span>
          </div>
          <div className="citation-text">
            {excerpt}
          </div>
        </div>
      )}
    </div>
  );
}

export default SourceCitation;
