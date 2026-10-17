import { useEffect, useMemo, useState } from 'react';

const API_BASE = 'http://localhost:8000';

function App() {
  const [sessionId, setSessionId] = useState(() => localStorage.getItem('datarefine_session') || '');
  const [file, setFile] = useState(null);
  const [summary, setSummary] = useState(null);
  const [preview, setPreview] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [history, setHistory] = useState([]);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);

  const ensureSession = async () => {
    if (sessionId) return sessionId;
    const response = await fetch(`${API_BASE}/api/sessions`, { method: 'POST' });
    const data = await response.json();
    setSessionId(data.session_id);
    localStorage.setItem('datarefine_session', data.session_id);
    return data.session_id;
  };

  useEffect(() => {
    if (!sessionId) {
      ensureSession();
    }
  }, [sessionId]);

  const refreshSummary = async () => {
    const active = sessionId || (await ensureSession());
    const response = await fetch(`${API_BASE}/api/datasets/${active}/summary`);
    if (response.ok) {
      setSummary(await response.json());
    }
  };

  const refreshPreview = async () => {
    const active = sessionId || (await ensureSession());
    const response = await fetch(`${API_BASE}/api/datasets/${active}/preview?limit=10`);
    if (response.ok) {
      setPreview(await response.json());
    }
  };

  const refreshRecommendations = async () => {
    const active = sessionId || (await ensureSession());
    const response = await fetch(`${API_BASE}/api/datasets/${active}/recommendations`, { method: 'POST' });
    if (response.ok) {
      const data = await response.json();
      setRecommendations(data.items || []);
    }
  };

  const refreshHistory = async () => {
    const active = sessionId || (await ensureSession());
    const response = await fetch(`${API_BASE}/api/datasets/${active}/history`);
    if (response.ok) {
      setHistory(await response.json());
    }
  };

  const refreshAll = async () => {
    await refreshSummary();
    await refreshPreview();
    await refreshRecommendations();
    await refreshHistory();
  };

  const uploadDataset = async () => {
    if (!file) {
      setMessage('Please choose a file first.');
      return;
    }

    const active = sessionId || (await ensureSession());
    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);

    const response = await fetch(`${API_BASE}/api/datasets/${active}/upload`, {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    setLoading(false);

    if (!response.ok) {
      setMessage(data.detail || 'Upload failed.');
      return;
    }

    setMessage('Dataset uploaded successfully.');
    await refreshAll();
  };

  const applyOperation = async (operation, params = {}) => {
    if (!sessionId) {
      setMessage('No active session.');
      return;
    }

    const response = await fetch(`${API_BASE}/api/datasets/${sessionId}/clean`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ operation, ...params }),
    });
    const data = await response.json();

    if (!response.ok) {
      setMessage(data.detail || 'Cleaning failed.');
      return;
    }

    setMessage(`Applied ${operation}.`);
    await refreshAll();
  };

  const exportFile = async (kind) => {
    if (!sessionId) return;
    window.open(`${API_BASE}/api/datasets/${sessionId}/export/${kind}`, '_blank');
  };

  const tableColumns = useMemo(() => (preview.length ? Object.keys(preview[0]) : []), [preview]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-title">DataRefine</div>
          <div className="brand-subtitle">AI Data Workspace</div>
        </div>

        <nav className="sidebar-nav">
          {['Overview', 'Data Workspace', 'Profiling', 'AI Cleaning', 'Manual Cleaning', 'Reports'].map((item) => (
            <div key={item} className="nav-item">{item}</div>
          ))}
        </nav>
      </aside>

      <main className="content">
        <header className="topbar">
          <div>
            <p className="eyebrow">Workspace</p>
            <h1>Modern Data Cleaning Studio</h1>
          </div>
          <div className="header-actions">
            <button onClick={() => applyOperation('fill_missing', { strategy: 'mean' })}>Fill Missing</button>
            <button onClick={() => applyOperation('drop_duplicates')}>Drop Duplicates</button>
            <button className="primary" onClick={() => exportFile('csv')}>Export CSV</button>
          </div>
        </header>

        <div className="grid two-up">
          <section className="card upload-card">
            <div className="card-header">
              <h2>Upload dataset</h2>
              <span className="badge">CSV / XLSX / XLS</span>
            </div>

            <div className="upload-box">
              <input type="file" accept=".csv,.xlsx,.xls" onChange={(event) => setFile(event.target.files?.[0] || null)} />
              <button onClick={uploadDataset} disabled={loading || !file}>
                {loading ? 'Uploading…' : 'Upload dataset'}
              </button>
            </div>

            {message && <p className="status-text">{message}</p>}
          </section>

          <section className="card actions-card">
            <h2>Quick actions</h2>
            <div className="action-list">
              <button onClick={() => applyOperation('replace_invalid', { treat_negative_as_invalid: true })}>Replace invalid values</button>
              <button onClick={() => applyOperation('remove_outliers', { method: 'iqr', threshold: 1.5 })}>Remove outliers</button>
              <button onClick={() => exportFile('xlsx')}>Download Excel</button>
              <button onClick={() => exportFile('pdf')}>Generate PDF report</button>
            </div>
          </section>
        </div>

        {summary && (
          <section className="stats-grid">
            <Stat label="Rows" value={summary.rows} />
            <Stat label="Columns" value={summary.cols} />
            <Stat label="Missing" value={summary.missing} />
            <Stat label="Duplicates" value={summary.duplicates} />
            <Stat label="Quality" value={`${summary.quality_score}/100`} />
          </section>
        )}

        {preview.length > 0 && (
          <section className="card table-card">
            <div className="card-header table-header">
              <h2>Dataset preview</h2>
              <span>first 10 rows</span>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {tableColumns.map((key) => (
                      <th key={key}>{key}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.map((row, index) => (
                    <tr key={index}>
                      {tableColumns.map((key) => (
                        <td key={`${index}-${key}`}>{String(row[key] ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {recommendations.length > 0 && (
          <section className="card">
            <h2>AI recommendations</h2>
            <div className="recommendation-list">
              {recommendations.map((item, index) => (
                <div key={index} className="recommendation-item">
                  <div className="recommendation-head">
                    <strong>{item.title}</strong>
                    <span>{item.severity}</span>
                  </div>
                  <p>{item.issue}</p>
                  <p>Recommended action: {item.recommendation}</p>
                  <small>Evidence: {item.evidence}</small>
                </div>
              ))}
            </div>
          </section>
        )}

        {history.length > 0 && (
          <section className="card">
            <h2>Cleaning history</h2>
            <div className="history-list">
              {history.map((entry, index) => (
                <div key={index} className="history-item">
                  {entry.label || 'Operation'} — {entry.quality_delta !== undefined ? `Δ quality ${entry.quality_delta}` : 'applied'}
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

export default App;
