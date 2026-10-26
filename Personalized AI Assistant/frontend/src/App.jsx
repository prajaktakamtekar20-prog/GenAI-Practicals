import { Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import KnowledgeBase from './pages/KnowledgeBase';
import Assistant from './pages/Assistant';
import ModelComparison from './pages/ModelComparison';
import AboutExperiment from './pages/AboutExperiment';

function App() {
  return (
    <div className="app-shell">
      <Sidebar />
      <main className="main-panel">
        <Header />
        <div className="content-wrap">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/knowledge-base" element={<KnowledgeBase />} />
            <Route path="/assistant" element={<Assistant />} />
            <Route path="/model-comparison" element={<ModelComparison />} />
            <Route path="/about" element={<AboutExperiment />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

export default App;
