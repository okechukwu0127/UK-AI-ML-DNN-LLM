async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  });

  const contentType = response.headers.get('content-type') || '';
  const payload = contentType.includes('application/json')
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message = typeof payload === 'object' && payload?.error ? payload.error : response.statusText;
    throw new Error(message || 'Request failed');
  }

  return payload;
}

export async function testQuery(query) {
  return fetchJson('/api/query-test', {
    method: 'POST',
    body: JSON.stringify({ query }),
  });
}

export async function testLoginRoute(payload = { username: 'admin', password: 'secret123' }) {
  return fetchJson('/api/login', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function listReports() {
  return fetchJson('/api/reports');
}

export async function loadReport(fileName) {
  return fetchJson(`/api/reports/${encodeURIComponent(fileName)}`);
}

export async function getHealth() {
  return fetchJson('/health');
}
