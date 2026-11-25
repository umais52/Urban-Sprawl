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

/** CDX Point 8 – full-grid 2030 raster overlay. Cold inference covers every 64px tile. */
export const getForecastLayer = () =>
  api.get("/forecast_layer", { timeout: 300_000 });

/** Temporal History – full-resolution urban raster overlay for one year.
 *  Cold reprojection of a city-wide GeoTIFF can exceed the shared 30s client timeout. */
export const getHistory = (year) =>
  api.get(`/history/${year}`, { timeout: 120_000 });

/** CDX Point 3 – Fetch Markov demand matrix */
export const getDemand = () => api.get("/demand");

/** What-If Sandbox – full-grid road uplift overlay. The first call may build the 2030 baseline. */
export const postSimulateRoad = (payload) =>
  api.post("/simulate_road", payload, { timeout: 300_000 });

/** Health check */
export const getHealth = () =>
  axios.get("http://localhost:8000/health"); // root-level, not /api/v1

export default api;
