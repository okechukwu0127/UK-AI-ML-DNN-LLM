import { NavLink, Routes, Route, Navigate } from 'react-router-dom';
import QueryScreen from './components/QueryScreen';
import ReportsScreen from './components/ReportsScreen';

function Shell({ children }) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">SQL Injection Security Platform</p>
          <h1>Middleware Query Tester and Report Dashboard</h1>
        </div>
        <nav className="nav-tabs">
          <NavLink to="/" end className={({ isActive }) => `nav-tab ${isActive ? 'active' : ''}`}>
            Query Test
          </NavLink>
          <NavLink to="/reports" className={({ isActive }) => `nav-tab ${isActive ? 'active' : ''}`}>
            Reports
          </NavLink>
        </nav>
      </header>
      <main className="content-wrap">{children}</main>
    </div>
  );
}

export default function App() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<QueryScreen />} />
        <Route path="/reports" element={<ReportsScreen />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Shell>
  );
}
