import React, { useState, useEffect, useRef } from 'react';
import { 
  Send, 
  Trash2, 
  Sparkles, 
  Layers, 
  Cpu, 
  HelpCircle, 
  Loader2, 
  RefreshCw,
  Info
} from 'lucide-react';
import ChatMessage from './ChatMessage';
import { sendChat, getChatHistory, clearChatHistory } from '../api/client';

const SAMPLE_PROMPTS = [
  "What is the primary aim of this academic project?",
  "How does RAG differ from fine-tuning?",
  "When should the assistant admit uncertainty?",
  "Which vector store and similarity metric are used?"
];

function ChatWindow() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [mode, setMode] = useState('rag');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadHistory = async () => {
    try {
      const data = await getChatHistory();
      if (data && data.history) {
        setMessages(data.history);
      }
    } catch (err) {
      console.error('Failed to load chat history:', err);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async (customPrompt = null) => {
    const textToSend = (customPrompt || input).trim();
    if (!textToSend || loading) return;

    setError(null);
    setInput('');

    // Append optimistic user message
    const userMsg = {
      id: `user_${Date.now()}`,
      role: 'user',
      content: textToSend,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await sendChat(textToSend, mode);
      const assistantMsg = {
        id: `assistant_${Date.now()}`,
        role: 'assistant',
        content: res.answer,
        mode: res.mode,
        sources: res.sources,
        retrieval_score: res.retrieval_score,
        is_unsupported: res.is_unsupported,
        timestamp: res.timestamp || new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Generation failed';
      setError(`Inference error: ${msg}`);
      const errorMsg = {
        id: `err_${Date.now()}`,
        role: 'assistant',
        content: `Error: ${msg}`,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    if (messages.length === 0) return;
    if (window.confirm('Clear conversation history?')) {
      try {
        await clearChatHistory();
        setMessages([]);
      } catch (err) {
        console.error('Failed to clear history:', err);
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const getModeDescription = () => {
    switch (mode) {
      case 'rag':
        return 'RAG Mode: Retrieves top relevant document chunks via FAISS vector search and grounds the answer strictly in your uploaded files with citations.';
      case 'base':
        return 'Base LLM Mode: Queries the open-source foundational model directly from pre-trained weights without accessing any uploaded documents.';
      case 'fine_tuned':
        return 'Fine-Tuned Mode: Evaluates the model adapted via LoRA on domain instruction data for specialized academic answering style.';
      default:
        return '';
    }
  };

  return (
    <div className="chat-window-card">
      {/* Chat Control Toolbar */}
      <div className="chat-toolbar">
        <div className="mode-selector-group">
          <span className="mode-label">Operating Mode:</span>
          <button
            type="button"
            className={`mode-btn ${mode === 'rag' ? 'active rag-active' : ''}`}
            onClick={() => setMode('rag')}
          >
            <Layers size={14} />
            <span>RAG (Document-Grounded)</span>
          </button>
          <button
            type="button"
            className={`mode-btn ${mode === 'base' ? 'active base-active' : ''}`}
            onClick={() => setMode('base')}
          >
            <Cpu size={14} />
            <span>Base LLM (Direct)</span>
          </button>
          <button
            type="button"
            className={`mode-btn ${mode === 'fine_tuned' ? 'active lora-active' : ''}`}
            onClick={() => setMode('fine_tuned')}
          >
            <Sparkles size={14} />
            <span>LoRA Fine-Tuned</span>
          </button>
        </div>

        <button 
          type="button" 
          className="ghost-button clear-btn" 
          onClick={handleClear} 
          disabled={messages.length === 0}
          title="Clear all conversation messages"
        >
          <Trash2 size={14} />
          <span>Clear Chat</span>
        </button>
      </div>

      {/* Mode Explanation Banner */}
      <div className="mode-explanation-strip">
        <Info size={14} className="text-primary flex-shrink-0" />
        <span>{getModeDescription()}</span>
      </div>

      {/* Messages Scroll Area */}
      <div className="chat-messages-scroll">
        {messages.length === 0 && (
          <div className="chat-placeholder">
            <div className="placeholder-icon-circle">
              <Sparkles size={28} className="text-primary" />
            </div>
            <h4>Personalized AI Research Assistant</h4>
            <p className="placeholder-sub">
              Ask questions about your uploaded documents. The assistant will retrieve relevant excerpts, cite source filenames and pages, and refuse to invent unsupported facts.
            </p>

            <div className="sample-prompts-container">
              <span className="prompts-heading">Suggested research queries:</span>
              <div className="prompts-grid">
                {SAMPLE_PROMPTS.map((promptText, idx) => (
                  <button
                    key={idx}
                    type="button"
                    className="prompt-pill"
                    onClick={() => handleSend(promptText)}
                  >
                    {promptText}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}

        {loading && (
          <div className="message-container assistant-side">
            <div className="message-avatar">
              <Loader2 size={18} className="spin text-primary" />
            </div>
            <div className="message-bubble thinking-bubble">
              <div className="thinking-dots">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
              </div>
              <span className="thinking-text">
                {mode === 'rag' 
                  ? 'Retrieving vector context & synthesizing grounded answer...' 
                  : 'Generating response from model weights...'}
              </span>
            </div>
          </div>
        )}

        {error && (
          <div className="alert-box alert-error chat-error-alert">
            <span>{error}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Chat Composer Bar */}
      <div className="chat-composer-wrap">
        <textarea
          rows="2"
          className="chat-textarea"
          placeholder="Ask a question about your uploaded research documents (Press Enter to send, Shift+Enter for new line)..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />

        <button
          type="button"
          className="primary-button send-button"
          onClick={() => handleSend()}
          disabled={loading || !input.trim()}
          title="Send message"
        >
          {loading ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
          <span>Send</span>
        </button>
      </div>
    </div>
  );
}

export default ChatWindow;
