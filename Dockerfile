export const API_BASE = 'http://localhost:8000';

export const createSession = async () => {
  const response = await fetch(`${API_BASE}/api/sessions`, { method: 'POST' });
  return response.json();
};
