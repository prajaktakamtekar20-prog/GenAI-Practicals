import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Database, 
  Bot, 
  BarChart3, 
  BookOpen, 
  ShieldCheck,
  Cpu
} from 'lucide-react';

const navItems = [
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/knowledge-base', label: 'Knowledge Base', icon: Database },
  { to: '/assistant', label: 'AI Assistant', icon: Bot },
  { to: '/model-comparison', label: 'Model Comparison', icon: BarChart3 },
  { to: '/about', label: 'About Experiment', icon: BookOpen },
];

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand-box">
        <div className="brand-mark">
          <Bot size={22} />
        </div>
        <div>
          <p className="eyebrow">Academic GenAI</p>
          <h1>Personalized AI</h1>
        </div>
      </div>

      <nav className="nav-list">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            <Icon size={18} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="architecture-badge">
          <ShieldCheck size={14} className="text-success" />
          <span>RAG • FAISS • LoRA</span>
        </div>
        <p className="sidebar-note">
          100% Local Inference<br />No Paid APIs Required
        </p>
      </div>
    </aside>
  );
}

export default Sidebar;
