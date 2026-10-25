import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  Download, 
  Play, 
  Sparkles, 
  CheckCircle2, 
  Clock, 
  ShieldCheck, 
  AlertCircle, 
  Cpu, 
  Layers, 
  ChevronDown, 
  ChevronUp,
  FileSpreadsheet,
  Activity,
  Award
} from 'lucide-react';
import { 
  getEvaluationResults, 
  runEvaluation, 
  getEvaluationExportUrl, 
  getTrainingStatus 
} from '../api/client';

function ModelComparison() {
  const [results, setResults] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState(null);
  const [selectedQuestionIndex, setSelectedQuestionIndex] = useState(0);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [evalRes, trainRes] = await Promise.all([
        getEvaluationResults(),
        getTrainingStatus()
      ]);
      setResults(evalRes);
      setTrainingStatus(trainRes);
    } catch (err) {
      console.error('Failed to load comparison data:', err);
      setError('Could not load evaluation data. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleRunEvaluation = async () => {
    try {
      setEvaluating(true);
      setError(null);
      const res = await runEvaluation();
      setResults(res);
      await fetchData();
    } catch (err) {
      console.error('Evaluation run failed:', err);
      setError('Evaluation benchmark failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setEvaluating(false);
    }
  };

  const getMetricColor = (val, max = 1.0) => {
    const ratio = val / max;
    if (ratio >= 0.7) return 'progress-green';
    if (ratio >= 0.35) return 'progress-blue';
    return 'progress-orange';
  };

  const summaries = results?.mode_summaries || [];
  const questionsList = results?.mode_summaries?.[0]?.details || [];

  return (
    <div className="page-column">
      {/* Hero Card */}
      <section className="hero-card">
        <div>
          <div className="hero-badge-row">
            <span className="badge badge-primary">Empirical Evaluation Suite</span>
            <span className="badge badge-neutral">Token F1 • Groundedness • Latency</span>
          </div>
          <h3>Model Comparison & Experimental Evaluation</h3>
          <p className="muted-text">
            Compare Base LLM, RAG-Enhanced LLM, and LoRA Fine-Tuned variants on identical benchmark questions to quantitatively assess factual accuracy, citation fidelity, and response grounding.
          </p>
        </div>

        <div className="hero-actions">
          <button 
            type="button" 
            className="primary-button" 
            onClick={handleRunEvaluation}
            disabled={evaluating}
          >
            <Play size={16} className={evaluating ? 'spin' : ''} />
            {evaluating ? 'Computing Benchmark...' : 'Run Evaluation Benchmark'}
          </button>

          {results?.status === 'completed' && (
            <a 
              href={getEvaluationExportUrl()} 
              download="model_evaluation_results.csv"
              className="ghost-button"
            >
              <Download size={16} /> Export CSV
            </a>
          )}
        </div>
      </section>

      {error && (
        <div className="alert-box alert-error">
          <AlertCircle size={18} className="alert-icon" />
          <span>{error}</span>
        </div>
      )}

      {/* LoRA Training Status Card */}
      <section className="panel-card">
        <div className="panel-header-row">
          <div>
            <h3>LoRA / QLoRA Fine-Tuning Status</h3>
            <p className="muted-text">Parameter-efficient adaptation configuration and training history.</p>
          </div>
          {trainingStatus?.status === 'completed' ? (
            <span className="pill pill-success">
              <CheckCircle2 size={14} /> Adapter Trained & Saved
            </span>
          ) : (
            <span className="pill pill-neutral">
              <Clock size={14} /> Training Pending
            </span>
          )}
        </div>

        {trainingStatus?.status === 'completed' ? (
          <div className="training-report-grid">
            <div className="training-metric-box">
              <span className="label">LoRA Rank (r)</span>
              <strong>{trainingStatus.lora_rank}</strong>
              <span className="sub">Alpha: {trainingStatus.lora_alpha}</span>
            </div>
            <div className="training-metric-box">
              <span className="label">Trainable Parameters</span>
              <strong>{trainingStatus.trainable_parameters?.toLocaleString()}</strong>
              <span className="sub">{trainingStatus.trainable_percentage}% of total weights</span>
            </div>
            <div className="training-metric-box">
              <span className="label">Final Training Loss</span>
              <strong>{trainingStatus.final_loss ?? 'N/A'}</strong>
              <span className="sub">Epochs: {trainingStatus.epochs}</span>
            </div>
            <div className="training-metric-box">
              <span className="label">Training Duration</span>
              <strong>{trainingStatus.duration_seconds}s</strong>
              <span className="sub">Device: {trainingStatus.device?.toUpperCase()}</span>
            </div>
          </div>
        ) : (
          <div className="pending-training-box">
            <Clock size={24} className="text-muted" />
            <p>
              No custom LoRA adapter trained yet. Run <code>python -m backend.training.train_lora --epochs 2</code> in your terminal or trigger training to adapt the model.
            </p>
          </div>
        )}
      </section>

      {/* Benchmark Summary Table */}
      <section className="panel-card">
        <div className="panel-header-row">
          <div>
            <h3>Comparative Performance Matrix</h3>
            <p className="muted-text">
              Real quantitative scores measured on identical test questions.
            </p>
          </div>
          {results?.evaluated_at && (
            <span className="timestamp-badge">
              Benchmark evaluated at: {new Date(results.evaluated_at).toLocaleString()}
            </span>
          )}
        </div>

        {summaries.length === 0 ? (
          <div className="empty-state-card">
            <BarChart3 size={36} className="empty-icon" />
            <h4>Evaluation results pending</h4>
            <p>Click "Run Evaluation Benchmark" above to generate live empirical scores across the modes.</p>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th>Operating Mode</th>
                  <th>Questions</th>
                  <th>Correctness (Token F1)</th>
                  <th>Context Groundedness</th>
                  <th>Citation Accuracy</th>
                  <th>Average Latency</th>
                  <th>Refusal Rate</th>
                </tr>
              </thead>
              <tbody>
                {summaries.map((summary) => (
                  <tr key={summary.mode} className={summary.mode === 'fine_tuned_rag' ? 'highlight-row' : ''}>
                    <td className="mode-name-cell">
                      <strong>{summary.label}</strong>
                      <span className="cell-sub">{summary.mode}</span>
                    </td>
                    <td>{summary.questions_evaluated}</td>
                    <td>
                      <div className="metric-bar-cell">
                        <span className="metric-val">{summary.average_correctness_f1.toFixed(3)}</span>
                        <div className="bar-track">
                          <div 
                            className={`bar-fill ${getMetricColor(summary.average_correctness_f1, 0.4)}`} 
                            style={{ width: `${Math.min(100, (summary.average_correctness_f1 / 0.4) * 100)}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td>
                      <div className="metric-bar-cell">
                        <span className="metric-val">{summary.average_groundedness.toFixed(3)}</span>
                        <div className="bar-track">
                          <div 
                            className={`bar-fill ${getMetricColor(summary.average_groundedness, 1.0)}`} 
                            style={{ width: `${summary.average_groundedness * 100}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td>
                      <div className="metric-bar-cell">
                        <span className="metric-val">{(summary.average_citation_accuracy * 100).toFixed(0)}%</span>
                        <div className="bar-track">
                          <div 
                            className={`bar-fill ${getMetricColor(summary.average_citation_accuracy, 1.0)}`} 
                            style={{ width: `${summary.average_citation_accuracy * 100}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td>
                      <strong>{(summary.average_latency_ms / 1000).toFixed(2)}s</strong>
                      <span className="cell-sub">{Math.round(summary.average_latency_ms)} ms</span>
                    </td>
                    <td>
                      <span>{(summary.unsupported_refusal_rate * 100).toFixed(0)}%</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Visual Comparative Charts */}
      {summaries.length > 0 && (
        <section className="panel-card">
          <h3>Visual Metric Comparison</h3>
          <p className="muted-text">Side-by-side performance comparison across key evaluation criteria.</p>

          <div className="charts-grid">
            <div className="chart-card">
              <h4>Factual Correctness (Token F1 Score)</h4>
              <p className="chart-sub">Higher score indicates higher lexical and semantic match to ground-truth answers.</p>
              <div className="chart-bars-wrap">
                {summaries.map((s) => (
                  <div key={s.mode} className="chart-bar-item">
                    <div className="bar-item-header">
                      <span className="bar-label">{s.label}</span>
                      <strong className="bar-value">{s.average_correctness_f1.toFixed(3)}</strong>
                    </div>
                    <div className="bar-outer">
                      <div 
                        className={`bar-inner ${getMetricColor(s.average_correctness_f1, 0.4)}`} 
                        style={{ width: `${Math.min(100, (s.average_correctness_f1 / 0.4) * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="chart-card">
              <h4>Context Groundedness Score</h4>
              <p className="chart-sub">Measures the percentage of key terms directly supported by retrieved context.</p>
              <div className="chart-bars-wrap">
                {summaries.map((s) => (
                  <div key={s.mode} className="chart-bar-item">
                    <div className="bar-item-header">
                      <span className="bar-label">{s.label}</span>
                      <strong className="bar-value">{s.average_groundedness.toFixed(3)}</strong>
                    </div>
                    <div className="bar-outer">
                      <div 
                        className={`bar-inner ${getMetricColor(s.average_groundedness, 1.0)}`} 
                        style={{ width: `${s.average_groundedness * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Question-by-Question Comparative Inspector */}
      {questionsList.length > 0 && (
        <section className="panel-card">
          <div className="panel-header-row">
            <div>
              <h3>Question-by-Question Comparative Inspector</h3>
              <p className="muted-text">
                Select a benchmark question to inspect side-by-side responses across all tested modes.
              </p>
            </div>
          </div>

          <div className="question-selector-row">
            {questionsList.map((q, idx) => (
              <button
                key={idx}
                type="button"
                className={`q-select-btn ${selectedQuestionIndex === idx ? 'active' : ''}`}
                onClick={() => setSelectedQuestionIndex(idx)}
              >
                Q{idx + 1}
              </button>
            ))}
          </div>

          {questionsList[selectedQuestionIndex] && (
            <div className="question-inspector-box">
              <div className="question-meta-bar">
                <span className="badge badge-primary">Category: {questionsList[selectedQuestionIndex].category}</span>
                <span className="question-text-title">
                  <strong>Question:</strong> {questionsList[selectedQuestionIndex].question}
                </span>
              </div>

              <div className="reference-answer-box">
                <span className="ref-label">Reference Ground-Truth Answer:</span>
                <p>{questionsList[selectedQuestionIndex].reference_answer}</p>
              </div>

              <div className="answers-comparison-grid">
                {summaries.map((summary) => {
                  const detail = summary.details?.[selectedQuestionIndex];
                  if (!detail) return null;
                  return (
                    <div key={summary.mode} className="answer-mode-col">
                      <div className="col-header">
                        <strong>{summary.label}</strong>
                        <span className="col-score">F1: {detail.correctness_f1.toFixed(3)}</span>
                      </div>
                      <div className="col-body">
                        <p>{detail.generated_answer}</p>
                      </div>
                      <div className="col-footer">
                        <span>Latency: {detail.latency_ms} ms</span>
                        {detail.groundedness > 0 && (
                          <span>Groundedness: {detail.groundedness.toFixed(2)}</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}

export default ModelComparison;
