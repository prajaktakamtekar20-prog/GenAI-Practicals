import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api',
  timeout: 180000, // 3 minutes timeout for CPU LLM operations
});

export const getHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const getModelStatus = async () => {
  const response = await api.get('/model/status');
  return response.data;
};

export const loadModel = async (modelName) => {
  const response = await api.post('/model/load', { model_name: modelName });
  return response.data;
};

export const getDocuments = async () => {
  const response = await api.get('/documents');
  return response.data;
};

export const getDocumentStats = async () => {
  const response = await api.get('/documents/stats');
  return response.data;
};

export const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const deleteDocument = async (documentId) => {
  const response = await api.delete(`/documents/${documentId}`);
  return response.data;
};

export const indexDocuments = async () => {
  const response = await api.post('/documents/index');
  return response.data;
};

export const sendChat = async (question, mode = 'rag', modelName = null) => {
  const response = await api.post('/chat', {
    question,
    mode,
    model_name: modelName,
  });
  return response.data;
};

export const getChatHistory = async () => {
  const response = await api.get('/chat/history');
  return response.data;
};

export const clearChatHistory = async () => {
  const response = await api.delete('/chat/history');
  return response.data;
};

export const getEvaluationResults = async () => {
  const response = await api.get('/evaluation/results');
  return response.data;
};

export const runEvaluation = async (modes = null) => {
  const response = await api.post('/evaluation/run', { modes });
  return response.data;
};

export const getEvaluationExportUrl = () => {
  return `${api.defaults.baseURL}/evaluation/export`;
};

export const getTrainingStatus = async () => {
  const response = await api.get('/training/status');
  return response.data;
};

export const runTraining = async (config = {}) => {
  const response = await api.post('/training/run', config);
  return response.data;
};

export default api;
