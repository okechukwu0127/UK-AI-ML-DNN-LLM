import {
  BarChart,
  Bar,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  AreaChart,
  Area,
} from 'recharts';

const COLORS = ['#7c3aed', '#0ea5e9', '#14b8a6', '#f59e0b', '#ef4444', '#22c55e', '#8b5cf6', '#ec4899'];

const toNumber = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

const safeValue = (value, fallback = 'N/A') => (value === null || value === undefined || value === '' ? fallback : value);

function firstObject(records) {
  return Array.isArray(records) && records.length > 0 ? records[0] : {};
}

function fileHint(fileName = '') {
  return fileName.toLowerCase();
}

function lineSeriesKeys(columns) {
  return columns.filter((column) => column !== 'sample_size' && column !== 'threshold' && column !== 'route' && column !== 'file');
}

function numericSeriesKeys(records, columns, xKey) {
  return lineSeriesKeys(columns).filter((column) =>
    records.some((row) => row[column] !== undefined && row[column] !== null && row[column] !== '' && toNumber(row[column]) !== null && column !== xKey),
  );
}

function buildSimplePieData(records, labelKey, valueKey) {
  return records.map((row) => ({
    name: String(row[labelKey]),
    value: toNumber(row[valueKey]) ?? 0,
  }));
}

function buildFeatureImportance(records) {
  return records
    .map((row) => ({
      feature: String(row.feature),
      importance: toNumber(row.importance) ?? 0,
    }))
    .slice(0, 20)
    .reverse();
}

export default function ReportCharts({ report, fileName }) {
  const records = report?.records || [];
  const columns = report?.columns || Object.keys(firstObject(records));
  const hint = fileHint(fileName);

  if (!records.length) {
    return <div className="empty-state">No rows to chart for this file.</div>;
  }

  if (hint.includes('feature_importance')) {
    const data = buildFeatureImportance(records);
    return (
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={data} layout="vertical" margin={{ top: 10, right: 20, left: 20, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" stroke="#94a3b8" />
          <YAxis type="category" dataKey="feature" width={160} stroke="#94a3b8" />
          <Tooltip />
          <Bar dataKey="importance" fill="#8b5cf6" radius={[0, 8, 8, 0]} />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (hint.includes('sample_size') || hint.includes('threshold')) {
    const xKey = columns.includes('sample_size') ? 'sample_size' : 'threshold';
    const seriesKeys = numericSeriesKeys(records, columns, xKey);
    const data = records.map((row) => {
      const next = { [xKey]: row[xKey] };
      seriesKeys.forEach((key) => {
        const value = toNumber(row[key]);
        if (value !== null) next[key] = value;
      });
      return next;
    });

    return (
      <ResponsiveContainer width="100%" height={360}>
        <LineChart data={data} margin={{ top: 10, right: 20, left: 8, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey={xKey} stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" />
          <Tooltip />
          <Legend />
          {seriesKeys.slice(0, 5).map((key, index) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={COLORS[index % COLORS.length]}
              strokeWidth={2}
              dot={{ r: 3 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    );
  }

  if (hint.includes('latency')) {
    const data = records.map((row) => ({
      route: safeValue(row.route),
      avg_inference_ms: toNumber(row.avg_inference_ms) ?? 0,
      median_inference_ms: toNumber(row.median_inference_ms) ?? 0,
      request_count: toNumber(row.request_count) ?? 0,
    }));

    return (
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={data} margin={{ top: 10, right: 20, left: 8, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="route" stroke="#94a3b8" angle={-20} textAnchor="end" height={80} interval={0} />
          <YAxis yAxisId="left" stroke="#94a3b8" />
          <YAxis yAxisId="right" orientation="right" stroke="#94a3b8" />
          <Tooltip />
          <Legend />
          <Bar yAxisId="left" dataKey="avg_inference_ms" fill="#7c3aed" radius={[8, 8, 0, 0]} />
          <Bar yAxisId="left" dataKey="median_inference_ms" fill="#0ea5e9" radius={[8, 8, 0, 0]} />
          <Bar yAxisId="right" dataKey="request_count" fill="#14b8a6" radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (hint.includes('feature_importance') || (columns.includes('feature') && columns.includes('importance'))) {
    const data = buildFeatureImportance(records);
    return (
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={data} layout="vertical" margin={{ top: 10, right: 20, left: 20, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" stroke="#94a3b8" />
          <YAxis type="category" dataKey="feature" width={160} stroke="#94a3b8" />
          <Tooltip />
          <Bar dataKey="importance" fill="#8b5cf6" radius={[0, 8, 8, 0]} />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (columns.includes('attack_family') && columns.includes('precision')) {
    const data = records.map((row) => ({
      attack_family: row.attack_family,
      precision: toNumber(row.precision) ?? 0,
    }));
    return (
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={data} layout="vertical" margin={{ top: 10, right: 20, left: 20, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" domain={[0, 1]} stroke="#94a3b8" />
          <YAxis type="category" dataKey="attack_family" width={160} stroke="#94a3b8" />
          <Tooltip />
          <Bar dataKey="precision" fill="#0ea5e9" radius={[0, 8, 8, 0]} />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (columns.includes('route') && (columns.includes('blocked_requests') || columns.includes('allowed_requests'))) {
    const data = records.map((row) => ({
      route: row.route,
      blocked_requests: toNumber(row.blocked_requests) ?? 0,
      allowed_requests: toNumber(row.allowed_requests) ?? 0,
    }));
    return (
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={data} margin={{ top: 10, right: 20, left: 8, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="route" stroke="#94a3b8" angle={-18} textAnchor="end" height={80} interval={0} />
          <YAxis stroke="#94a3b8" />
          <Tooltip />
          <Legend />
          <Bar dataKey="blocked_requests" stackId="a" fill="#ef4444" radius={[8, 8, 0, 0]} />
          <Bar dataKey="allowed_requests" stackId="a" fill="#22c55e" radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (columns.includes('attack_type') && columns.includes('count')) {
    const pieData = buildSimplePieData(records, 'attack_type', 'count');
    return (
      <ResponsiveContainer width="100%" height={360}>
        <PieChart>
          <Tooltip />
          <Legend />
          <Pie data={pieData} dataKey="value" nameKey="name" outerRadius={120} innerRadius={55}>
            {pieData.map((_, index) => (
              <Cell key={index} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    );
  }

  if (columns.includes('count') && (columns.includes('route') || columns.includes('injection_type'))) {
    const data = records.map((row) => ({
      label: safeValue(row.route || row.injection_type || row.attack_type),
      count: toNumber(row.count) ?? 0,
    }));
    return (
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={data} margin={{ top: 10, right: 20, left: 8, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="label" stroke="#94a3b8" angle={-20} textAnchor="end" height={80} interval={0} />
          <YAxis stroke="#94a3b8" />
          <Tooltip />
          <Bar dataKey="count" fill="#7c3aed" radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (columns.includes('blocked_requests') && columns.includes('allowed_requests') && !columns.includes('route')) {
    const data = records.map((row) => ({
      label: row.threshold ?? row.sample_size ?? row.file ?? 'overall',
      blocked_requests: toNumber(row.blocked_requests) ?? 0,
      allowed_requests: toNumber(row.allowed_requests) ?? 0,
    }));
    return (
      <ResponsiveContainer width="100%" height={360}>
        <AreaChart data={data} margin={{ top: 10, right: 20, left: 8, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="label" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" />
          <Tooltip />
          <Legend />
          <Area type="monotone" dataKey="blocked_requests" stackId="1" stroke="#ef4444" fill="#ef4444" fillOpacity={0.35} />
          <Area type="monotone" dataKey="allowed_requests" stackId="1" stroke="#22c55e" fill="#22c55e" fillOpacity={0.35} />
        </AreaChart>
      </ResponsiveContainer>
    );
  }

  if (columns.includes('total_requests') && columns.includes('blocked_requests')) {
    const data = records.map((row) => ({
      label: row.route || row.threshold || row.sample_size || row.file || 'overall',
      total_requests: toNumber(row.total_requests) ?? 0,
      blocked_requests: toNumber(row.blocked_requests) ?? 0,
      allowed_requests: toNumber(row.allowed_requests) ?? 0,
    }));
    return (
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={data} margin={{ top: 10, right: 20, left: 8, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="label" stroke="#94a3b8" angle={-15} textAnchor="end" height={70} interval={0} />
          <YAxis stroke="#94a3b8" />
          <Tooltip />
          <Legend />
          <Bar dataKey="blocked_requests" fill="#ef4444" />
          <Bar dataKey="allowed_requests" fill="#22c55e" />
          <Bar dataKey="total_requests" fill="#0ea5e9" />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (columns.includes('blocked_requests') || columns.includes('allowed_requests')) {
    const data = records.map((row) => ({
      label: row.route || row.sample_size || row.threshold || row.file || 'overall',
      blocked_requests: toNumber(row.blocked_requests) ?? 0,
      allowed_requests: toNumber(row.allowed_requests) ?? 0,
    }));
    return (
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={data} margin={{ top: 10, right: 20, left: 8, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="label" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" />
          <Tooltip />
          <Legend />
          <Bar dataKey="blocked_requests" fill="#ef4444" />
          <Bar dataKey="allowed_requests" fill="#22c55e" />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (columns.includes('count') && columns.includes('percentage')) {
    const data = records.map((row) => ({
      label: row.attack_type || row.route || row.injection_type,
      count: toNumber(row.count) ?? 0,
      percentage: toNumber(row.percentage) ?? 0,
    }));
    return (
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={data} margin={{ top: 10, right: 20, left: 8, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="label" stroke="#94a3b8" angle={-20} textAnchor="end" height={70} interval={0} />
          <YAxis stroke="#94a3b8" />
          <Tooltip />
          <Legend />
          <Bar dataKey="count" fill="#7c3aed" />
          <Bar dataKey="percentage" fill="#0ea5e9" />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  return <div className="empty-state">This report is better viewed as a table. Use the data grid below.</div>;
}
