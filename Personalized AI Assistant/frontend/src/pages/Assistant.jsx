import React from 'react';
import ChatWindow from '../components/ChatWindow';
import { Bot, ShieldCheck, Sparkles, Layers, AlertCircle } from 'lucide-react';

function Assistant() {
  return (
    <div className="page-column assistant-page">
      <ChatWindow />
    </div>
  );
}

export default Assistant;
