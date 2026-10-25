import React, { useState, useEffect } from 'react';
import { ShieldCheck, Server, AlertCircle } from 'lucide-react';
import { getHealth } from '../api/client';
import ModelStatus from './ModelStatus';

function Header() {
  const [online, setOnline] = useState(false);

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const res = await getHealth();
        setOnline(res.status === 'ok');
      } catch (err) {
        setOnline(false);
      }
    };
    checkConnection();
    const interval = setInterval(checkConnection, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="topbar">
      <div className="topbar-title-section">
        <p className="eyebrow">Academic GenAI Project • Localhost In-Memory Inference</p>
        <h2>Development of a Personalized AI Assistant</h2>
      </div>

      <div className="topbar-controls">
        <div className={`connection-pill ${online ? 'pill-online' : 'pill-offline'}`}>
          <Server size={14} />
          <span>{online ? 'Backend Online (Port 8000)' : 'Backend Disconnected'}</span>
        </div>

        <ModelStatus />
      </div>
    </header>
  );
}

export default Header;
