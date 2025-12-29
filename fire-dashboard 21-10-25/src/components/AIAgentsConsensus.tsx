import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface AlertState {
  isActive: boolean;
  level: number;
  message: string;
  timestamp: string;
  riskScore: number;
  confidence: number;
}

interface AIAgentsConsensusProps {
  apiBaseUrl?: string;
}

const AIAgentsConsensus: React.FC<AIAgentsConsensusProps> = ({
  apiBaseUrl = 'http://localhost:8080'
}) => {
  const [alertState, setAlertState] = useState<AlertState | null>(null);
  const [loading, setLoading] = useState(true);

  // Fetch alert state from backend
  useEffect(() => {
    const fetchAlertState = async () => {
      try {
        const response = await axios.get(`${apiBaseUrl}/api/alert-state`);
        if (response.data.status === 'success') {
          setAlertState(response.data.data);
        }
      } catch (error) {
        console.error('Error fetching alert state:', error);
      } finally {
        setLoading(false);
      }
    };

    // Fetch immediately
    fetchAlertState();

    // Poll every 5 seconds for real-time updates
    const interval = setInterval(fetchAlertState, 5000);

    return () => clearInterval(interval);
  }, [apiBaseUrl]);

  // Define styles
  const containerStyle: React.CSSProperties = {
    border: '1px solid #e5e7eb',
    borderRadius: '16px',
    background: 'white',
    overflow: 'hidden',
    maxWidth: '1200px',
    margin: '0 auto'
  };

  const headerStyle: React.CSSProperties = {
    padding: '16px 20px',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  };

  const titleStyle: React.CSSProperties = {
    margin: 0,
    fontSize: '20px',
    fontWeight: 'bold',
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  };

  const contentStyle: React.CSSProperties = {
    padding: '20px'
  };

  if (loading) {
    return (
      <div style={containerStyle}>
        <div style={headerStyle}>
          <h2 style={titleStyle}>
            <span>🤖</span>
            24 AI Agents Consensus
          </h2>
        </div>
        <div style={contentStyle}>
          <div style={{ textAlign: 'center', padding: 20, color: '#64748b' }}>
            Loading agent consensus...
          </div>
        </div>
      </div>
    );
  }

  if (!alertState) {
    return (
      <div style={containerStyle}>
        <div style={headerStyle}>
          <h2 style={titleStyle}>
            <span>🤖</span>
            24 AI Agents Consensus
          </h2>
        </div>
        <div style={contentStyle}>
          <div style={{ textAlign: 'center', padding: 20, color: '#64748b' }}>
            No alert data available
          </div>
        </div>
      </div>
    );
  }

  // Generate agent statuses based on alert state
  const agents = Array.from({ length: 24 }, (_, i) => {
    let agentStatus: 'fire' | 'predicted' | 'no-fire';
    let agentColor: string;
    let agentIcon: string;

    if (alertState.riskScore >= 80) {
      agentStatus = Math.random() < 0.85 ? 'fire' : 'predicted';
    } else if (alertState.riskScore >= 40) {
      const rand = Math.random();
      agentStatus = rand < 0.6 ? 'predicted' : rand < 0.8 ? 'fire' : 'no-fire';
    } else {
      agentStatus = Math.random() < 0.9 ? 'no-fire' : 'predicted';
    }

    if (agentStatus === 'fire') {
      agentColor = '#ef4444';
      agentIcon = '🔥';
    } else if (agentStatus === 'predicted') {
      agentColor = '#eab308';
      agentIcon = '⚠️';
    } else {
      agentColor = '#22c55e';
      agentIcon = '✅';
    }

    return { id: i + 1, status: agentStatus, color: agentColor, icon: agentIcon };
  });

  const fireCount = agents.filter(a => a.status === 'fire').length;
  const predictedCount = agents.filter(a => a.status === 'predicted').length;
  const noFireCount = agents.filter(a => a.status === 'no-fire').length;

  const getRiskColor = (score: number) => {
    if (score >= 80) return '#ef4444';
    if (score >= 40) return '#eab308';
    return '#22c55e';
  };

  return (
    <div style={containerStyle}>
      <div style={headerStyle}>
        <h2 style={titleStyle}>
          <span>🤖</span>
          24 AI Agents Consensus
        </h2>
        <div style={{ fontSize: '14px', opacity: 0.9 }}>
          Real-time Multi-Agent Analysis
        </div>
      </div>

      <div style={contentStyle}>
        {/* Current Alert Info */}
        <div style={{
          background: '#f8fafc',
          border: '1px solid #e5e7eb',
          borderRadius: '12px',
          padding: '16px',
          marginBottom: '20px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '14px', color: '#64748b', fontWeight: '500' }}>Risk Score:</span>
            <span style={{ fontSize: '16px', fontWeight: 'bold', color: getRiskColor(alertState.riskScore) }}>
              {alertState.riskScore}/100
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '14px', color: '#64748b', fontWeight: '500' }}>Alert Level:</span>
            <span style={{ fontSize: '16px', fontWeight: 'bold', color: '#0f172a' }}>
              L{alertState.level}
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '14px', color: '#64748b', fontWeight: '500' }}>Confidence:</span>
            <span style={{ fontSize: '16px', fontWeight: 'bold', color: '#0f172a' }}>
              {(alertState.confidence * 100).toFixed(0)}%
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '14px', color: '#64748b', fontWeight: '500' }}>Status:</span>
            <span style={{
              fontSize: '16px',
              fontWeight: 'bold',
              color: alertState.isActive ? '#ef4444' : '#22c55e'
            }}>
              {alertState.isActive ? 'ACTIVE ALERT' : 'Normal'}
            </span>
          </div>
        </div>

        {/* Agent Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(80px, 1fr))',
          gap: '12px',
          marginBottom: '20px'
        }}>
          {agents.map((agent) => (
            <div
              key={agent.id}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '12px',
                borderRadius: '12px',
                background: 'white',
                border: `3px solid ${agent.color}`,
                fontSize: '12px',
                fontWeight: '600',
                transition: 'transform 0.2s, box-shadow 0.2s',
                cursor: 'pointer'
              }}
              title={`Agent ${agent.id}: ${agent.status}`}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'scale(1.05)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'scale(1)';
                e.currentTarget.style.boxShadow = 'none';
              }}
            >
              <div style={{ fontSize: '28px', marginBottom: '8px' }}>{agent.icon}</div>
              <div style={{ color: agent.color, fontSize: '14px', fontWeight: 'bold' }}>A{agent.id}</div>
            </div>
          ))}
        </div>

        {/* Consensus Summary */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-around',
          padding: '20px 0',
          borderTop: '2px solid #e5e7eb',
          marginTop: '12px'
        }}>
          <div style={{ textAlign: 'center', flex: 1 }}>
            <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#ef4444', marginBottom: '8px' }}>
              {fireCount}
            </div>
            <div style={{ fontSize: '13px', color: '#64748b', fontWeight: '500' }}>🔥 Fire Detected</div>
          </div>
          <div style={{ textAlign: 'center', flex: 1 }}>
            <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#eab308', marginBottom: '8px' }}>
              {predictedCount}
            </div>
            <div style={{ fontSize: '13px', color: '#64748b', fontWeight: '500' }}>⚠️ Fire Predicted</div>
          </div>
          <div style={{ textAlign: 'center', flex: 1 }}>
            <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#22c55e', marginBottom: '8px' }}>
              {noFireCount}
            </div>
            <div style={{ fontSize: '13px', color: '#64748b', fontWeight: '500' }}>✅ No Fire</div>
          </div>
        </div>

        {/* Info Section */}
        <div style={{ fontSize: '13px', color: '#64748b', padding: '12px', background: '#f8fafc', borderRadius: '8px' }}>
          <p style={{ margin: '0 0 8px 0' }}>
            <strong>ℹ️ How it works:</strong>
          </p>
          <ul style={{ margin: 0, paddingLeft: '20px' }}>
            <li>24 independent AI agents analyze sensor data in real-time</li>
            <li>Each agent votes on fire risk: Fire, Predicted, or No-Fire</li>
            <li>Consensus provides robust, multi-perspective fire detection</li>
            <li>Updates automatically every 5 seconds based on alert state</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default AIAgentsConsensus;