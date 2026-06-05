import { useMemo, useState } from 'react';
import { testLoginRoute, testQuery } from '../lib/api';

export default function QueryScreen() {
  const [query, setQuery] = useState('SELECT * FROM users WHERE id = 1 OR 1=1 --');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [loginResult, setLoginResult] = useState(null);
  const [error, setError] = useState('');
  const [loginError, setLoginError] = useState('');
  const [loginLoading, setLoginLoading] = useState(false);

  const statusTone = useMemo(() => {
    if (!result) return 'neutral';
    return result.is_malicious ? 'danger' : 'success';
  }, [result]);

  const submit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const payload = await testQuery(query);
      setResult(payload);
    } catch (err) {
      setError(err.message || 'Failed to test query');
    } finally {
      setLoading(false);
    }
  };

  const testLogin = async () => {
    setLoginLoading(true);
    setLoginError('');
    setLoginResult(null);
    try {
      const payload = await testLoginRoute();
      setLoginResult(payload);
    } catch (err) {
      setLoginError(err.message || 'Failed to test login route');
    } finally {
      setLoginLoading(false);
    }
  };

  return (
    <div className="page-grid">
      <section className="panel hero-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Query testing</p>
            <h1>SQL injection classifier</h1>
            <p className="muted">
              Paste a SQL query, then send it to the Flask API for malicious/benign inspection.
            </p>
          </div>
          <div className="status-chip">Live API: /api/query-test</div>
        </div>

        <form onSubmit={submit} className="query-form">
          <label className="field-label" htmlFor="query">
            SQL query
          </label>
          <textarea
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="textarea"
            placeholder="Type or paste SQL here..."
            rows={10}
          />
          <div className="actions-row">
            <button className="primary-btn" type="submit" disabled={loading || !query.trim()}>
              {loading ? 'Inspecting...' : 'Inject Query'}
            </button>
            <button
              className="ghost-btn"
              type="button"
              onClick={() => setQuery('SELECT name FROM users WHERE id = 42')}
            >
              Load benign sample
            </button>
            <button
              className="ghost-btn"
              type="button"
              onClick={() => setQuery("1 UNION SELECT username,password FROM users")}
            >
              Load malicious sample
            </button>
            <button className="ghost-btn" type="button" onClick={testLogin} disabled={loginLoading}>
              {loginLoading ? 'Testing login...' : 'Test /api/login'}
            </button>
          </div>
        </form>

        <div className="report-note">
          <strong>Note:</strong> Standalone SQL statements can still be flagged as suspicious because the classifier
          is tuned to recognise injection-style structure, not database semantics. Route context and request shape
          matter.
        </div>
      </section>

      <section className="panel result-panel">
        <p className="eyebrow">Prediction result</p>
        {!result && !error && <p className="muted">Run a query to view the response from the model.</p>}

        {error && <div className="alert alert-danger">{error}</div>}

        {result && (
          <div className={`result-card ${statusTone}`}>
            <div className="result-badge">
              {result.is_malicious ? 'Threat detected' : 'Benign query'}
            </div>
            <h2>{result.message}</h2>
            <div className="metric-grid">
              <div>
                <span>Confidence</span>
                <strong>{(Number(result.confidence) * 100).toFixed(2)}%</strong>
              </div>
              <div>
                <span>Attack type</span>
                <strong>{result.attack_type || 'Unknown'}</strong>
              </div>
              <div>
                <span>Latency</span>
                <strong>{Number(result.inference_ms || 0).toFixed(2)} ms</strong>
              </div>
              <div>
                <span>Score</span>
                <strong>{Number(result.prediction_score || 0).toFixed(4)}</strong>
              </div>
            </div>
            <pre className="result-json">{JSON.stringify(result.details, null, 2)}</pre>
          </div>
        )}

        {(loginResult || loginError) && (
          <div className="endpoint-card">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Route sanity check</p>
                <h3>/api/login</h3>
              </div>
              <div className="status-chip">{loginResult?.success ? 'Allowed route' : 'Check failed'}</div>
            </div>
            {loginError && <div className="alert alert-danger">{loginError}</div>}
            {loginResult && (
              <>
                <p className="muted">
                  This route is useful for showing that a benign application request can pass through the middleware
                  without being treated as a SQL injection payload.
                </p>
                <div className="metric-grid">
                  <div>
                    <span>Success</span>
                    <strong>{String(loginResult.success)}</strong>
                  </div>
                  <div>
                    <span>Message</span>
                    <strong>{loginResult.message}</strong>
                  </div>
                </div>
                <pre className="result-json">{JSON.stringify(loginResult, null, 2)}</pre>
              </>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
