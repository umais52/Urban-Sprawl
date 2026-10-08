/**
 * Axios API Service
 * =================
 * Centralised HTTP client pointing at the FastAPI backend.
 * Every component imports from here so the base URL is defined once.
 */
import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000/api/v1",
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 30_000, // 30 s – model inference can be slow
});

// ── Endpoint wrappers ─────────────────────────────────────────────────────

/** CDX Point 8 – Run the fusion forecast */
export const postForecast = (payload) => api.post("/forecast", payload);

/** CDX Point 8 – Fetch GeoJSON forecast growth polygon layer */
export const getForecastLayer = () => api.get("/forecast_layer");

/** CDX Point 3 – Fetch Markov demand matrix */
export const getDemand = () => api.get("/demand");

/** What-If Sandbox – Simulate a new road */
export const postSimulateRoad = (payload) =>
  api.post("/simulate_road", payload);

/** Health check */
export const getHealth = () =>
  axios.get("http://localhost:8000/health"); // root-level, not /api/v1

export default api;
