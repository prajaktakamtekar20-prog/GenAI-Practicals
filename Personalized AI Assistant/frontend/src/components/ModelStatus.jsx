import React, { useState, useEffect } from 'react';
import { Cpu, Sparkles, CheckCircle2, AlertCircle, HardDrive, RefreshCw } from 'lucide-react';
import { getModelStatus, loadModel } from '../api/client';

function ModelStatus({ onRefresh }) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);

  const fetchStatus = async () => {
    try {
      setLoading(true);
      const data = await getModelStatus();
      setStatus(data);
    } catch (err) {
      console.error('Failed to fetch model status:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleWarmup = async () => {
    try {
      setActionLoading(true);
      await loadModel(status?.model_name);
      await fetchStatus();
      if (onRefresh) onRefresh();
    } catch (err) {
      console.error('Failed to preload model:', err);
    } finally {
      setActionLoading(false);
    }
  };

  if (!status) return null;

  return (
    <div className="model-status-card">
      <div className="status-item">
        <Sparkles size={16} className="text-primary" />
        <span className="label">Active Model:</span>
        <strong className="value">{status.model_name?.split('/')[1] || status.model_name}</strong>
      </div>

      <div className="status-item">
        <Cpu size={16} className="text-secondary" />
        <span className="label">Compute:</span>
        <span className="badge badge-neutral">{status.device?.toUpperCase() || 'CPU'}</span>
      </div>

      <div className="status-item">
        <HardDrive size={16} className="text-muted" />
        <span className="label">RAM Available:</span>
        <span className="value">{status.system_ram_available_gb} GB</span>
      </div>

      <div className="status-item status-badge-item">
        {status.loaded ? (
          <span className="pill pill-success">
            <CheckCircle2 size={14} /> Model Loaded
          </span>
        ) : (
          <button 
            className="pill pill-action" 
            onClick={handleWarmup}
            disabled={actionLoading}
            title="Preload model weights into memory for instant inference"
          >
            <RefreshCw size={14} className={actionLoading ? 'spin' : ''} />
            {actionLoading ? 'Loading Weights...' : 'Preload Model'}
          </button>
        )}
      </div>

      {status.loaded_adapter && (
        <div className="status-item">
          <span className="pill pill-purple">LoRA Adapter Active</span>
        </div>
      )}
    </div>
  );
}

export default ModelStatus;
