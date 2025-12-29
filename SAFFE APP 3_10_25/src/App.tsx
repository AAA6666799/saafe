import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import SaafeLovable from './components/SaafeLovable';
import FireDataSender from './components/FireDataSender';
import EmailRecipientManager from './components/EmailRecipientManager';

// Navigation component
function Navigation() {
  const location = useLocation();
  
  const navStyle = {
    display: 'flex',
    gap: '12px',
    padding: '16px',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    borderRadius: '12px',
    marginBottom: '16px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
  };

  const linkStyle = (isActive: boolean) => ({
    padding: '12px 24px',
    borderRadius: '8px',
    textDecoration: 'none',
    fontWeight: 'bold',
    fontSize: '16px',
    transition: 'all 0.3s ease',
    background: isActive ? 'white' : 'rgba(255, 255, 255, 0.2)',
    color: isActive ? '#667eea' : 'white',
    border: 'none',
    cursor: 'pointer',
    boxShadow: isActive ? '0 2px 4px rgba(0, 0, 0, 0.1)' : 'none'
  });

  return (
    <nav style={navStyle}>
      <Link
        to="/"
        style={linkStyle(location.pathname === '/')}
      >
        🏠 Dashboard
      </Link>
      <Link
        to="/data-sender"
        style={linkStyle(location.pathname === '/data-sender')}
      >
        📡 Fire Data Sender
      </Link>
      <Link
        to="/email-recipients"
        style={linkStyle(location.pathname === '/email-recipients')}
      >
        📧 Email Recipients
      </Link>
    </nav>
  );
}

// Dashboard page wrapper
function DashboardPage() {
  return (
    <div>
      <SaafeLovable />
    </div>
  );
}

// Data Sender page wrapper
function DataSenderPage() {
  return (
    <div style={{ minHeight: '100vh', padding: '16px' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        <FireDataSender />
      </div>
    </div>
  );
}

// Email Recipients page wrapper
function EmailRecipientsPage() {
  return (
    <div style={{ minHeight: '100vh', padding: '16px' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        <EmailRecipientManager />
      </div>
    </div>
  );
}

// Main App component
export default function App() {
  return (
    <Router>
      <div style={{ minHeight: '100vh', background: '#f8fafc' }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '16px' }}>
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/data-sender" element={<DataSenderPage />} />
            <Route path="/email-recipients" element={<EmailRecipientsPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}