import React, { useState } from 'react';
import { User, Bot, Copy, Check, AlertTriangle, Layers } from 'lucide-react';
import SourceCitation from './SourceCitation';

function ChatMessage({ message }) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';

  const handleCopy = () => {
    if (message.content) {
      navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const getModeBadge = (mode) => {
    switch (mode) {
      case 'rag':
        return <span className="mode-badge mode-rag"><Layers size={12} /> RAG Grounded</span>;
      case 'base':
        return <span className="mode-badge mode-base">Base LLM</span>;
      case 'fine_tuned':
        return <span className="mode-badge mode-lora">LoRA Fine-Tuned</span>;
      default:
        return null;
    }
  };

  return (
    <div className={`message-container ${isUser ? 'user-side' : 'assistant-side'}`}>
      <div className="message-avatar">
        {isUser ? <User size={18} /> : <Bot size={18} />}
      </div>

      <div className="message-bubble">
        <div className="message-header-bar">
          <span className="sender-name">{isUser ? 'You' : 'Personalized AI'}</span>
          <div className="header-actions">
            {!isUser && message.mode && getModeBadge(message.mode)}
            {!isUser && (
              <button className="icon-subtle-button" onClick={handleCopy} title="Copy answer">
                {copied ? <Check size={14} className="text-success" /> : <Copy size={14} />}
              </button>
            )}
          </div>
        </div>

        {/* Warning if answer is unsupported by document context */}
        {!isUser && message.is_unsupported && (
          <div className="unsupported-callout">
            <AlertTriangle size={16} className="callout-icon" />
            <div>
              <strong>Context Insufficient:</strong>
              <p>The answer could not be grounded in uploaded documents. The assistant refused speculation.</p>
            </div>
          </div>
        )}

        <div className="message-text">
          {message.content}
        </div>

        {/* Citations section */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="message-sources">
            <div className="sources-heading">
              <span>Supporting Evidence ({message.sources.length} sources)</span>
              {message.retrieval_score != null && (
                <span className="top-similarity">Top similarity: {Math.round(message.retrieval_score * 100)}%</span>
              )}
            </div>
            <div className="citations-stack">
              {message.sources.map((source, idx) => (
                <SourceCitation key={idx} source={source} index={idx} />
              ))}
            </div>
          </div>
        )}

        <div className="message-footer">
          {message.timestamp && (
            <span className="message-time">
              {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default ChatMessage;
