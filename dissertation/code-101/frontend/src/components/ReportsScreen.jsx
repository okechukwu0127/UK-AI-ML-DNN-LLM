import { useEffect, useMemo, useState } from 'react';
import { listReports, loadReport } from '../lib/api';
import ReportCharts from './ReportCharts';

const REPORT_LABELS = {
  'runtime_dashboard_summary.csv': 'Runtime dashboard summary',
  'runtime_dashboard_summary_dataset_batch.csv': 'Dataset-batch runtime dashboard',
  'dataset_batch_class_distribution.png': 'Batch class distribution (image artifact)',
  'dataset_batch_injection_type_distribution.png': 'Batch injection-type distribution (image artifact)',
};

const REPORT_NOTES = {
  'attack_family_precision.csv': 'Shows how accurately the middleware attributes a detected payload to a SQL injection family. Useful for discussing interpretability after binary classification.',
  'attack_type_attribution_summary.csv': 'Summarises the attack families inferred by the middleware during runtime logging. This helps explain what the model sees beyond simple malicious/benign output.',
  'dataset_batch_logs.csv': 'Row-level log export for the automated batch endpoint. It is the raw source for class distribution, injection-type, and batch deployment analysis.',
  'dataset_batch_sample_size_summary.csv': 'Summarises how batch behaviour changes with the sample size parameter. This links deployment behaviour to the dissertation’s scalability discussion.',
  'false_negative_analysis.csv': 'Highlights malicious rows that were not flagged. Use this to discuss missed detections and security risk.',
  'false_positive_analysis.csv': 'Highlights benign rows that were incorrectly treated as suspicious. Use this to discuss user-impact and over-blocking risk.',
  'figure_3_4_model_f1_by_sample_size.csv': 'Compares the test F1-score of Random Forest, DNN, LSTM, and GRU across the training sample sizes used in the study.',
  'figure_3_4_sample_size_summary.csv': 'Captures the best-model metrics and feature-space growth at each sample size. This supports the sample-size sensitivity discussion.',
  'inference_latency_summary.csv': 'Measures average and median inference time per route. This is used to argue deployment feasibility and runtime overhead.',
  'random_forest_feature_importance.csv': 'Shows which TF-IDF and engineered SQL features contributed most to the selected best model.',
  'route_block_allow_frequency.csv': 'Shows how often each route was blocked or allowed. Useful for route-level deployment behaviour analysis.',
  'route_level_testing_summary.csv': 'A compact route-wise summary of total requests, block counts, allow counts, and top attack types.',
  'runtime_dashboard_summary.csv': 'Overall middleware activity across all logged traffic. This is the main operational summary of the API layer.',
  'runtime_dashboard_summary_dataset_batch.csv': 'Batch-only runtime behaviour for the dataset-driven endpoint. Useful for controlled benchmark evaluation.',
  'threshold_sensitivity_analysis.csv': 'Shows how the number of blocked and allowed requests changes as the confidence threshold changes.',
  'unknown_attack_error_analysis.csv': 'Highlights where the middleware could not confidently map a payload to a family. Useful for discussing interpretive limits.',
};

function formatTitle(fileName) {
  return (REPORT_LABELS[fileName] || fileName.replace(/[_-]/g, ' ').replace(/\.(csv|json)$/i, ''))
    .replace(/^Figure\s+\d+(?:\s+\d+)?\s*/i, '');
}

function formatHeading(fileName) {
  return formatTitle(fileName)
    .replace(/^Figure\s+\d+(?:\s+\d+)?\s*/i, '')
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function formatNote(fileName) {
  return REPORT_NOTES[fileName] || 'This report is part of the SQL injection detection evaluation and supports the runtime analytics discussion in the dissertation.';
}

function MetricCard({ label, value, subtext }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      {subtext ? <small>{subtext}</small> : null}
    </div>
  );
}

function DataTable({ report }) {
  const records = report?.records || [];
  const columns = report?.columns || [];

  if (!records.length) return <div className="empty-state">No tabular rows available.</div>;

  const visibleRows = records.slice(0, 25);

  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {visibleRows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column}>{String(row[column] ?? '')}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ReportsScreen() {
  const [catalog, setCatalog] = useState({ csv_files: [], json_files: [], all_files: [] });
  const [selected, setSelected] = useState('');
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    (async () => {
      try {
        const data = await listReports();
        setCatalog(data);
        const defaultFile = data.csv_files?.[0] || data.json_files?.[0] || '';
        setSelected(defaultFile);
      } catch (err) {
        setError(err.message || 'Unable to load reports');
      }
    })();
  }, []);

  useEffect(() => {
    if (!selected) return;
    (async () => {
      setLoading(true);
      setError('');
      try {
        const data = await loadReport(selected);
        setReport(data);
      } catch (err) {
        setError(err.message || 'Unable to load report');
      } finally {
        setLoading(false);
      }
    })();
  }, [selected]);

  const summary = useMemo(() => {
    if (!report) return {};
    const rowCount = report.row_count ?? report.records?.length ?? (report.data ? Object.keys(report.data).length : 0);
    return {
      rowCount,
      columnsCount: report.columns?.length || 0,
      fileType: report.type || 'unknown',
    };
  }, [report]);

  return (
    <div className="reports-grid">
      <aside className="panel reports-sidebar">
        <p className="eyebrow">Analytics catalog</p>
        <h1>Report explorer</h1>
        <p className="muted">
          Browse the generated CSV and JSON outputs, then inspect them with charts and tables.
        </p>

        <div className="file-list">
          {(catalog.csv_files || []).map((file) => (
            <button key={file} className={`file-pill ${selected === file ? 'active' : ''}`} onClick={() => setSelected(file)}>
              {formatTitle(file)}
            </button>
          ))}
          {(catalog.json_files || []).map((file) => (
            <button key={file} className={`file-pill ${selected === file ? 'active' : ''}`} onClick={() => setSelected(file)}>
              {formatTitle(file)}
            </button>
          ))}
        </div>
      </aside>

      <main className="report-main">
        <section className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Selected output</p>
              <h2>{selected ? formatHeading(selected) : 'No file selected'}</h2>
            </div>
            <div className="status-chip">{summary.fileType?.toUpperCase() || 'CSV/JSON'}</div>
          </div>

          {selected ? <div className="report-note warning">{formatNote(selected)}</div> : null}

          {error ? <div className="alert alert-danger">{error}</div> : null}

          <div className="stats-row">
            <MetricCard label="Rows" value={summary.rowCount || 0} subtext="Loaded from the selected report" />
            <MetricCard label="Columns" value={summary.columnsCount || 0} subtext="Tabular data fields" />
            <MetricCard label="Source type" value={summary.fileType || 'n/a'} subtext="CSV or JSON report" />
          </div>

          {loading ? (
            <div className="empty-state">Loading report...</div>
          ) : report?.type === 'json' ? (
            <pre className="result-json">{JSON.stringify(report.data, null, 2)}</pre>
          ) : (
            <>
              <div className="chart-shell">
                <ReportCharts report={report} fileName={selected} />
              </div>
              <DataTable report={report} />
            </>
          )}
        </section>
      </main>
    </div>
  );
}
