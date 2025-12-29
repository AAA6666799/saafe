import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '@/config/api';

interface Recipient {
  email: string;
  name: string;
  alertLevels: string[];
  enabled: boolean;
}

interface EmailRecipientManagerProps {
  apiBaseUrl?: string;
}

const EmailRecipientManager: React.FC<EmailRecipientManagerProps> = ({
  apiBaseUrl = API_BASE_URL
}) => {
  const [recipients, setRecipients] = useState<Recipient[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Form state for adding new recipient
  const [newRecipient, setNewRecipient] = useState({
    email: '',
    name: '',
    alertLevels: ['all'] as string[],
    enabled: true
  });

  // Edit mode state
  const [editingEmail, setEditingEmail] = useState<string | null>(null);
  const [editForm, setEditForm] = useState<Recipient | null>(null);

  // Test email state
  const [testEmail, setTestEmail] = useState('');
  const [testRiskScore, setTestRiskScore] = useState(85);
  const [sendingTest, setSendingTest] = useState(false);

  // Fetch recipients from backend
  const fetchRecipients = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get(`${apiBaseUrl}/api/email-recipients`);
      if (response.data.status === 'success') {
        setRecipients(response.data.data);
      }
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to fetch recipients');
      console.error('Error fetching recipients:', err);
    } finally {
      setLoading(false);
    }
  };

  // Add new recipient
  const handleAddRecipient = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setError(null);
      const response = await axios.post(`${apiBaseUrl}/api/email-recipients`, newRecipient);
      if (response.data.status === 'success') {
        setSuccessMessage(`Successfully added ${newRecipient.name}`);
        setNewRecipient({ email: '', name: '', alertLevels: ['all'], enabled: true });
        fetchRecipients();
        setTimeout(() => setSuccessMessage(null), 3000);
      }
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to add recipient');
      console.error('Error adding recipient:', err);
    }
  };

  // Update recipient
  const handleUpdateRecipient = async (email: string) => {
    if (!editForm) return;
    try {
      setError(null);
      const response = await axios.put(`${apiBaseUrl}/api/email-recipients/${encodeURIComponent(email)}`, {
        name: editForm.name,
        alertLevels: editForm.alertLevels,
        enabled: editForm.enabled
      });
      if (response.data.status === 'success') {
        setSuccessMessage(`Successfully updated ${editForm.name}`);
        setEditingEmail(null);
        setEditForm(null);
        fetchRecipients();
        setTimeout(() => setSuccessMessage(null), 3000);
      }
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to update recipient');
      console.error('Error updating recipient:', err);
    }
  };

  // Delete recipient
  const handleDeleteRecipient = async (email: string, name: string) => {
    if (!confirm(`Are you sure you want to remove ${name} from the recipient list?`)) return;
    try {
      setError(null);
      const response = await axios.delete(`${apiBaseUrl}/api/email-recipients/${encodeURIComponent(email)}`);
      if (response.data.status === 'success') {
        setSuccessMessage(`Successfully removed ${name}`);
        fetchRecipients();
        setTimeout(() => setSuccessMessage(null), 3000);
      }
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to delete recipient');
      console.error('Error deleting recipient:', err);
    }
  };

  // Send test email
  const handleSendTestEmail = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setSendingTest(true);
      setError(null);
      const response = await axios.post(`${apiBaseUrl}/api/test-alert-email`, {
        email: testEmail,
        riskScore: testRiskScore
      });
      if (response.data.status === 'success') {
        setSuccessMessage(`Test email sent to ${testEmail}`);
        setTestEmail('');
        setTimeout(() => setSuccessMessage(null), 3000);
      }
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to send test email');
      console.error('Error sending test email:', err);
    } finally {
      setSendingTest(false);
    }
  };

  // Toggle alert level
  const toggleAlertLevel = (level: string, isEdit: boolean = false) => {
    if (isEdit && editForm) {
      const levels = editForm.alertLevels.includes(level)
        ? editForm.alertLevels.filter(l => l !== level)
        : [...editForm.alertLevels.filter(l => l !== 'all'), level];
      setEditForm({ ...editForm, alertLevels: levels.length === 0 ? ['all'] : levels });
    } else {
      const levels = newRecipient.alertLevels.includes(level)
        ? newRecipient.alertLevels.filter(l => l !== level)
        : [...newRecipient.alertLevels.filter(l => l !== 'all'), level];
      setNewRecipient({ ...newRecipient, alertLevels: levels.length === 0 ? ['all'] : levels });
    }
  };

  useEffect(() => {
    fetchRecipients();
  }, []);

  const styles = {
    container: {
      border: '1px solid #e5e7eb',
      borderRadius: '16px',
      background: 'white',
      overflow: 'hidden',
      maxWidth: '1200px',
      margin: '0 auto'
    },
    header: {
      padding: '16px 20px',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      color: 'white',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center'
    },
    title: {
      margin: 0,
      fontSize: '20px',
      fontWeight: 'bold',
      display: 'flex',
      alignItems: 'center',
      gap: '8px'
    },
    content: {
      padding: '20px'
    },
    section: {
      marginBottom: '24px',
      padding: '16px',
      background: '#f8fafc',
      borderRadius: '12px',
      border: '1px solid #e5e7eb'
    },
    sectionTitle: {
      fontSize: '16px',
      fontWeight: '600',
      color: '#0f172a',
      marginBottom: '12px',
      display: 'flex',
      alignItems: 'center',
      gap: '8px'
    },
    form: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '12px',
      marginBottom: '12px'
    },
    formGroup: {
      display: 'flex',
      flexDirection: 'column' as const,
      gap: '4px'
    },
    label: {
      fontSize: '13px',
      fontWeight: '500',
      color: '#475569'
    },
    input: {
      padding: '8px 12px',
      border: '1px solid #e5e7eb',
      borderRadius: '8px',
      fontSize: '14px',
      outline: 'none',
      transition: 'border-color 0.2s'
    },
    button: {
      padding: '8px 16px',
      borderRadius: '8px',
      border: 'none',
      fontSize: '14px',
      fontWeight: '500',
      cursor: 'pointer',
      transition: 'all 0.2s',
      display: 'inline-flex',
      alignItems: 'center',
      gap: '6px'
    },
    buttonPrimary: {
      background: '#059669',
      color: 'white'
    },
    buttonSecondary: {
      background: '#e5e7eb',
      color: '#0f172a'
    },
    buttonDanger: {
      background: '#ef4444',
      color: 'white'
    },
    buttonWarning: {
      background: '#f59e0b',
      color: 'white'
    },
    table: {
      width: '100%',
      borderCollapse: 'collapse' as const,
      fontSize: '14px'
    },
    th: {
      textAlign: 'left' as const,
      padding: '12px',
      borderBottom: '2px solid #e5e7eb',
      fontWeight: '600',
      color: '#475569',
      background: '#f8fafc'
    },
    td: {
      padding: '12px',
      borderBottom: '1px solid #e5e7eb',
      color: '#0f172a'
    },
    badge: {
      display: 'inline-block',
      padding: '4px 8px',
      borderRadius: '6px',
      fontSize: '12px',
      fontWeight: '500',
      marginRight: '4px'
    },
    badgeAll: {
      background: '#dbeafe',
      color: '#1e40af'
    },
    badgeUrgent: {
      background: '#fee2e2',
      color: '#991b1b'
    },
    badgeWarning: {
      background: '#fef3c7',
      color: '#92400e'
    },
    badgeCaution: {
      background: '#fef9c3',
      color: '#713f12'
    },
    statusEnabled: {
      color: '#059669',
      fontWeight: '500'
    },
    statusDisabled: {
      color: '#94a3b8',
      fontWeight: '500'
    },
    alert: {
      padding: '12px 16px',
      borderRadius: '8px',
      marginBottom: '16px',
      display: 'flex',
      alignItems: 'center',
      gap: '8px'
    },
    alertSuccess: {
      background: '#dcfce7',
      color: '#166534',
      border: '1px solid #22c55e'
    },
    alertError: {
      background: '#fee2e2',
      color: '#991b1b',
      border: '1px solid #ef4444'
    },
    checkboxGroup: {
      display: 'flex',
      gap: '12px',
      flexWrap: 'wrap' as const,
      marginTop: '8px'
    },
    checkbox: {
      display: 'flex',
      alignItems: 'center',
      gap: '6px',
      cursor: 'pointer'
    }
  };

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <h2 style={styles.title}>
          <span>📧</span>
          Email Recipient Management
        </h2>
        <button
          onClick={fetchRecipients}
          style={{ ...styles.button, ...styles.buttonSecondary }}
        >
          🔄 Refresh
        </button>
      </div>

      <div style={styles.content}>
        {/* Success/Error Messages */}
        {successMessage && (
          <div style={{ ...styles.alert, ...styles.alertSuccess }}>
            <span>✅</span>
            {successMessage}
          </div>
        )}
        {error && (
          <div style={{ ...styles.alert, ...styles.alertError }}>
            <span>❌</span>
            {error}
          </div>
        )}

        {/* Add New Recipient Section */}
        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>
            <span>➕</span>
            Add New Recipient
          </h3>
          <form onSubmit={handleAddRecipient}>
            <div style={styles.form}>
              <div style={styles.formGroup}>
                <label style={styles.label}>Name *</label>
                <input
                  type="text"
                  value={newRecipient.name}
                  onChange={(e) => setNewRecipient({ ...newRecipient, name: e.target.value })}
                  placeholder="John Doe"
                  required
                  style={styles.input}
                />
              </div>
              <div style={styles.formGroup}>
                <label style={styles.label}>Email Address *</label>
                <input
                  type="email"
                  value={newRecipient.email}
                  onChange={(e) => setNewRecipient({ ...newRecipient, email: e.target.value })}
                  placeholder="john@example.com"
                  required
                  style={styles.input}
                />
              </div>
            </div>
            <div style={styles.formGroup}>
              <label style={styles.label}>Alert Levels</label>
              <div style={styles.checkboxGroup}>
                <label style={styles.checkbox}>
                  <input
                    type="checkbox"
                    checked={newRecipient.alertLevels.includes('all')}
                    onChange={() => setNewRecipient({ ...newRecipient, alertLevels: ['all'] })}
                  />
                  <span>All Alerts</span>
                </label>
                <label style={styles.checkbox}>
                  <input
                    type="checkbox"
                    checked={newRecipient.alertLevels.includes('urgent')}
                    onChange={() => toggleAlertLevel('urgent')}
                    disabled={newRecipient.alertLevels.includes('all')}
                  />
                  <span>Urgent (≥80)</span>
                </label>
                <label style={styles.checkbox}>
                  <input
                    type="checkbox"
                    checked={newRecipient.alertLevels.includes('warning')}
                    onChange={() => toggleAlertLevel('warning')}
                    disabled={newRecipient.alertLevels.includes('all')}
                  />
                  <span>Warning (≥40)</span>
                </label>
                <label style={styles.checkbox}>
                  <input
                    type="checkbox"
                    checked={newRecipient.alertLevels.includes('caution')}
                    onChange={() => toggleAlertLevel('caution')}
                    disabled={newRecipient.alertLevels.includes('all')}
                  />
                  <span>Caution (≥20)</span>
                </label>
              </div>
            </div>
            <button
              type="submit"
              style={{ ...styles.button, ...styles.buttonPrimary, marginTop: '12px' }}
            >
              ➕ Add Recipient
            </button>
          </form>
        </div>

        {/* Test Email Section */}
        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>
            <span>🧪</span>
            Send Test Email
          </h3>
          <form onSubmit={handleSendTestEmail}>
            <div style={styles.form}>
              <div style={styles.formGroup}>
                <label style={styles.label}>Test Email Address *</label>
                <input
                  type="email"
                  value={testEmail}
                  onChange={(e) => setTestEmail(e.target.value)}
                  placeholder="test@example.com"
                  required
                  style={styles.input}
                />
              </div>
              <div style={styles.formGroup}>
                <label style={styles.label}>Risk Score (20-100)</label>
                <input
                  type="number"
                  value={testRiskScore}
                  onChange={(e) => setTestRiskScore(Number(e.target.value))}
                  min="20"
                  max="100"
                  style={styles.input}
                />
              </div>
            </div>
            <button
              type="submit"
              disabled={sendingTest}
              style={{ ...styles.button, ...styles.buttonWarning, marginTop: '12px' }}
            >
              {sendingTest ? '📤 Sending...' : '📧 Send Test Email'}
            </button>
          </form>
        </div>

        {/* Recipients List Section */}
        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>
            <span>👥</span>
            Current Recipients ({recipients.length})
          </h3>
          {loading ? (
            <div style={{ textAlign: 'center', padding: '20px', color: '#64748b' }}>
              Loading recipients...
            </div>
          ) : recipients.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '20px', color: '#64748b' }}>
              No recipients configured. Add your first recipient above.
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>Name</th>
                    <th style={styles.th}>Email</th>
                    <th style={styles.th}>Alert Levels</th>
                    <th style={styles.th}>Status</th>
                    <th style={styles.th}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {recipients.map((recipient) => (
                    <tr key={recipient.email}>
                      <td style={styles.td}>
                        {editingEmail === recipient.email ? (
                          <input
                            type="text"
                            value={editForm?.name || ''}
                            onChange={(e) => setEditForm({ ...editForm!, name: e.target.value })}
                            style={{ ...styles.input, width: '100%' }}
                          />
                        ) : (
                          recipient.name
                        )}
                      </td>
                      <td style={styles.td}>{recipient.email}</td>
                      <td style={styles.td}>
                        {editingEmail === recipient.email ? (
                          <div style={styles.checkboxGroup}>
                            <label style={styles.checkbox}>
                              <input
                                type="checkbox"
                                checked={editForm?.alertLevels.includes('all')}
                                onChange={() => setEditForm({ ...editForm!, alertLevels: ['all'] })}
                              />
                              <span style={{ fontSize: '12px' }}>All</span>
                            </label>
                            <label style={styles.checkbox}>
                              <input
                                type="checkbox"
                                checked={editForm?.alertLevels.includes('urgent')}
                                onChange={() => toggleAlertLevel('urgent', true)}
                                disabled={editForm?.alertLevels.includes('all')}
                              />
                              <span style={{ fontSize: '12px' }}>Urgent</span>
                            </label>
                            <label style={styles.checkbox}>
                              <input
                                type="checkbox"
                                checked={editForm?.alertLevels.includes('warning')}
                                onChange={() => toggleAlertLevel('warning', true)}
                                disabled={editForm?.alertLevels.includes('all')}
                              />
                              <span style={{ fontSize: '12px' }}>Warning</span>
                            </label>
                            <label style={styles.checkbox}>
                              <input
                                type="checkbox"
                                checked={editForm?.alertLevels.includes('caution')}
                                onChange={() => toggleAlertLevel('caution', true)}
                                disabled={editForm?.alertLevels.includes('all')}
                              />
                              <span style={{ fontSize: '12px' }}>Caution</span>
                            </label>
                          </div>
                        ) : (
                          <>
                            {recipient.alertLevels.map((level) => (
                              <span
                                key={level}
                                style={{
                                  ...styles.badge,
                                  ...(level === 'all' ? styles.badgeAll :
                                      level === 'urgent' ? styles.badgeUrgent :
                                      level === 'warning' ? styles.badgeWarning :
                                      styles.badgeCaution)
                                }}
                              >
                                {level === 'all' ? '🌐 All' :
                                 level === 'urgent' ? '🔥 Urgent' :
                                 level === 'warning' ? '⚠️ Warning' :
                                 '⚡ Caution'}
                              </span>
                            ))}
                          </>
                        )}
                      </td>
                      <td style={styles.td}>
                        {editingEmail === recipient.email ? (
                          <label style={styles.checkbox}>
                            <input
                              type="checkbox"
                              checked={editForm?.enabled}
                              onChange={(e) => setEditForm({ ...editForm!, enabled: e.target.checked })}
                            />
                            <span>Enabled</span>
                          </label>
                        ) : (
                          <span style={recipient.enabled ? styles.statusEnabled : styles.statusDisabled}>
                            {recipient.enabled ? '✅ Enabled' : '⏸️ Disabled'}
                          </span>
                        )}
                      </td>
                      <td style={styles.td}>
                        <div style={{ display: 'flex', gap: '8px' }}>
                          {editingEmail === recipient.email ? (
                            <>
                              <button
                                onClick={() => handleUpdateRecipient(recipient.email)}
                                style={{ ...styles.button, ...styles.buttonPrimary, padding: '6px 12px' }}
                              >
                                💾 Save
                              </button>
                              <button
                                onClick={() => {
                                  setEditingEmail(null);
                                  setEditForm(null);
                                }}
                                style={{ ...styles.button, ...styles.buttonSecondary, padding: '6px 12px' }}
                              >
                                ❌ Cancel
                              </button>
                            </>
                          ) : (
                            <>
                              <button
                                onClick={() => {
                                  setEditingEmail(recipient.email);
                                  setEditForm(recipient);
                                }}
                                style={{ ...styles.button, ...styles.buttonSecondary, padding: '6px 12px' }}
                              >
                                ✏️ Edit
                              </button>
                              <button
                                onClick={() => handleDeleteRecipient(recipient.email, recipient.name)}
                                style={{ ...styles.button, ...styles.buttonDanger, padding: '6px 12px' }}
                              >
                                🗑️ Delete
                              </button>
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Info Section */}
        <div style={{ fontSize: '13px', color: '#64748b', padding: '12px', background: '#f8fafc', borderRadius: '8px' }}>
          <p style={{ margin: '0 0 8px 0' }}>
            <strong>ℹ️ How it works:</strong>
          </p>
          <ul style={{ margin: 0, paddingLeft: '20px' }}>
            <li>Recipients will receive email alerts based on their configured alert levels</li>
            <li>Emails are automatically sent when fire risks are detected by the system</li>
            <li>A 5-minute cooldown prevents email spam (bypassed if risk increases by 20+ points)</li>
            <li>Disabled recipients remain in the list but won't receive alerts</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default EmailRecipientManager;