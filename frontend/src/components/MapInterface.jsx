/**
 * MapInterface – Core map + floating CDX control panel
 * =====================================================
 * CDX Point 2/8 : Forecast trigger
 * Visual        : Dark frosted-glass sidebar with neon-green (#39ff14) accents
 *
 * Map is rendered full-viewport via react-map-gl (Mapbox GL JS wrapper).
 * Sidebar floats on the left with a glassmorphism aesthetic.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import Map, { NavigationControl, ScaleControl } from "react-map-gl/mapbox";
import "mapbox-gl/dist/mapbox-gl.css";

import {
  Activity,
  AlertTriangle,
  ChevronRight,
  Cpu,
  Key,
  Layers,
  MapPin,
  Radar,
  Satellite,
  Zap,
} from "lucide-react";

import { getHealth, postForecast } from "../services/api";

// ---------------------------------------------------------------------------
// Mapbox token from .env  (VITE_ prefix required by Vite)
// ---------------------------------------------------------------------------
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;
const IS_TOKEN_PLACEHOLDER =
  !MAPBOX_TOKEN || MAPBOX_TOKEN.includes("YOUR_MAPBOX_TOKEN");


// ---------------------------------------------------------------------------
// Karachi centre coordinates
// ---------------------------------------------------------------------------
const KARACHI = {
  longitude: 67.0011,
  latitude: 24.8607,
  zoom: 10.5,
  pitch: 45,
  bearing: -12,
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function MapInterface() {
  // Map viewport state
  const [viewState, setViewState] = useState(KARACHI);
  const mapRef = useRef(null);

  // UI state
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [backendStatus, setBackendStatus] = useState(null); // null | "online" | "offline"
  const [forecasting, setForecasting] = useState(false);
  const [forecastResult, setForecastResult] = useState(null);

  // ── Health ping on mount ────────────────────────────────────────────────
  useEffect(() => {
    getHealth()
      .then(() => setBackendStatus("online"))
      .catch(() => setBackendStatus("offline"));
  }, []);

  // ── Forecast handler ────────────────────────────────────────────────────
  const handleForecast = useCallback(async () => {
    setForecasting(true);
    setForecastResult(null);
    try {
      const { data } = await postForecast({
        s_dl: 0.72,
        s_xgb: 0.68,
        s_ca: 0.75,
        w1: 0.4,
        w2: 0.35,
        w3: 0.25,
      });
      setForecastResult(data);
    } catch (err) {
      console.error("[Forecast]", err);
      setForecastResult({ error: err.message });
    } finally {
      setForecasting(false);
    }
  }, []);

  // ── Render ──────────────────────────────────────────────────────────────
  return (
    <div className="relative h-screen w-screen overflow-hidden bg-black font-sans">
      {/* ─── MAP ──────────────────────────────────────────────────────── */}
      <Map
        ref={mapRef}
        {...viewState}
        onMove={(evt) => setViewState(evt.viewState)}
        mapboxAccessToken={MAPBOX_TOKEN}
        mapStyle="mapbox://styles/mapbox/dark-v11"
        style={{ width: "100%", height: "100%" }}
        antialias
        attributionControl={false}
      >
        <NavigationControl position="bottom-right" showCompass showZoom />
        <ScaleControl position="bottom-left" maxWidth={200} unit="metric" />
      </Map>

      {/* ─── TOKEN NOTICE (if token is placeholder) ──────────────────── */}
      {IS_TOKEN_PLACEHOLDER && (
        <div className="absolute top-4 right-4 z-40 max-w-sm rounded-xl border border-yellow-500/30 bg-black/80 p-3.5 shadow-2xl backdrop-blur-xl">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-yellow-400" />
            <div className="text-xs">
              <p className="font-semibold text-yellow-300">Mapbox Token Required</p>
              <p className="mt-1 text-gray-300">
                To load live dark vector & satellite tiles, paste your public token in{" "}
                <code className="rounded bg-white/10 px-1 py-0.5 text-green-300 font-mono text-[11px]">frontend/.env</code>:
              </p>
              <p className="mt-1 font-mono text-[10px] text-gray-400">
                VITE_MAPBOX_TOKEN=pk.eyJ1...
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ─── SIDEBAR TOGGLE (collapsed state) ─────────────────────────── */}
      {!sidebarOpen && (
        <button
          onClick={() => setSidebarOpen(true)}
          className="absolute left-4 top-4 z-50 flex items-center gap-2
                     rounded-lg border border-green-500/30
                     bg-black/60 px-3 py-2 text-sm text-green-400
                     shadow-lg shadow-green-500/10 backdrop-blur-xl
                     transition-all hover:border-green-400 hover:bg-black/80
                     hover:shadow-green-500/20"
          aria-label="Open control panel"
        >
          <Radar size={16} />
          <ChevronRight size={14} />
        </button>
      )}

      {/* ─── FLOATING SIDEBAR ─────────────────────────────────────────── */}
      <aside
        className={`
          absolute left-4 top-4 bottom-4 z-40 flex w-80 flex-col
          overflow-hidden rounded-2xl
          border border-green-500/20
          bg-gradient-to-b from-black/70 via-gray-950/60 to-black/70
          shadow-2xl shadow-green-500/5 backdrop-blur-2xl
          transition-all duration-500 ease-out
          ${sidebarOpen ? "translate-x-0 opacity-100" : "-translate-x-full opacity-0 pointer-events-none"}
        `}
      >
        {/* ── Header ────────────────────────────────────────────────── */}
        <div className="relative border-b border-green-500/20 px-5 py-5">
          {/* Decorative glow line */}
          <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-green-400/60 to-transparent" />

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg
                              border border-green-500/30 bg-green-500/10">
                <Satellite size={18} className="text-green-400" />
              </div>
              <div>
                <h1 className="text-base font-bold tracking-tight text-green-400">
                  Aks-e-Shajr CDX
                </h1>
                <p className="text-[11px] font-medium uppercase tracking-widest text-gray-500">
                  Urban Intelligence System
                </p>
              </div>
            </div>

            <button
              onClick={() => setSidebarOpen(false)}
              className="rounded-md p-1 text-gray-500 transition-colors
                         hover:bg-white/5 hover:text-green-400"
              aria-label="Close panel"
            >
              ✕
            </button>
          </div>
        </div>

        {/* ── Status row ────────────────────────────────────────────── */}
        <div className="flex items-center gap-2 border-b border-white/5 px-5 py-3">
          <span
            className={`h-2 w-2 rounded-full ${
              backendStatus === "online"
                ? "bg-green-400 shadow-sm shadow-green-400/50"
                : backendStatus === "offline"
                  ? "bg-red-500"
                  : "animate-pulse bg-yellow-400"
            }`}
          />
          <span className="text-xs text-gray-400">
            {backendStatus === "online"
              ? "Backend connected"
              : backendStatus === "offline"
                ? "Backend offline"
                : "Connecting…"}
          </span>
          <span className="ml-auto flex items-center gap-1 text-[10px] text-gray-600">
            <MapPin size={10} /> Karachi, PK
          </span>
        </div>

        {/* ── Coordinate readout ────────────────────────────────────── */}
        <div className="grid grid-cols-3 gap-px border-b border-white/5 bg-white/[0.02]">
          {[
            { label: "LAT", value: viewState.latitude.toFixed(4) },
            { label: "LNG", value: viewState.longitude.toFixed(4) },
            { label: "ZOOM", value: viewState.zoom.toFixed(1) },
          ].map((item) => (
            <div key={item.label} className="px-3 py-2.5 text-center">
              <p className="text-[9px] font-semibold uppercase tracking-widest text-gray-600">
                {item.label}
              </p>
              <p className="mt-0.5 font-mono text-xs text-green-400/80">
                {item.value}
              </p>
            </div>
          ))}
        </div>

        {/* ── CDX Metrics (scrollable body) ─────────────────────────── */}
        <div className="flex-1 overflow-y-auto px-5 py-4">
          <p className="mb-3 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-widest text-gray-500">
            <Layers size={12} /> CDX Modules
          </p>

          <div className="space-y-2">
            {[
              {
                icon: <Cpu size={14} />,
                title: "XGBoost Classifier",
                subtitle: "15 features · 300 trees",
                status: "ready",
              },
              {
                icon: <Activity size={14} />,
                title: "CNN-LSTM Predictor",
                subtitle: "4-step temporal · 64×64 tiles",
                status: "ready",
              },
              {
                icon: <Zap size={14} />,
                title: "CA Module",
                subtitle: "Contextual attention branch",
                status: "ready",
              },
            ].map((mod) => (
              <div
                key={mod.title}
                className="flex items-center gap-3 rounded-lg border border-white/5
                           bg-white/[0.02] px-3 py-2.5 transition-colors
                           hover:border-green-500/20 hover:bg-green-500/[0.03]"
              >
                <div className="text-green-400/60">{mod.icon}</div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-gray-300">{mod.title}</p>
                  <p className="text-[10px] text-gray-600">{mod.subtitle}</p>
                </div>
                <span className="h-1.5 w-1.5 rounded-full bg-green-400/60" />
              </div>
            ))}
          </div>

          {/* ── Forecast Result ──────────────────────────────────────── */}
          {forecastResult && !forecastResult.error && (
            <div className="mt-4 rounded-lg border border-green-500/20
                            bg-green-500/[0.04] p-3">
              <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-green-400/70">
                Forecast Result
              </p>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <p className="text-[9px] text-gray-500">P(Urban)</p>
                  <p className="font-mono text-lg font-bold text-green-400">
                    {(forecastResult.p_urban * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-[9px] text-gray-500">Class</p>
                  <p className={`font-mono text-lg font-bold ${
                    forecastResult.urban_class === "Urban"
                      ? "text-red-400"
                      : "text-green-400"
                  }`}>
                    {forecastResult.urban_class}
                  </p>
                </div>
                <div>
                  <p className="text-[9px] text-gray-500">L(fused)</p>
                  <p className="font-mono text-xs text-gray-400">
                    {forecastResult.l_fused}
                  </p>
                </div>
                <div>
                  <p className="text-[9px] text-gray-500">z(XGB)</p>
                  <p className="font-mono text-xs text-gray-400">
                    {forecastResult.z_xgb}
                  </p>
                </div>
              </div>
            </div>
          )}

          {forecastResult?.error && (
            <div className="mt-4 rounded-lg border border-red-500/20 bg-red-500/[0.04] p-3">
              <p className="text-xs text-red-400">{forecastResult.error}</p>
            </div>
          )}
        </div>

        {/* ── Footer / Forecast Button ──────────────────────────────── */}
        <div className="border-t border-green-500/20 px-5 py-4">
          <button
            onClick={handleForecast}
            disabled={forecasting || backendStatus !== "online"}
            className="group relative w-full overflow-hidden rounded-xl
                       border border-green-500/40 bg-green-500/10
                       px-4 py-3 text-sm font-bold uppercase tracking-wider
                       text-green-400 shadow-lg shadow-green-500/10
                       transition-all duration-300
                       hover:border-green-400 hover:bg-green-500/20
                       hover:shadow-green-500/25
                       disabled:cursor-not-allowed disabled:opacity-40"
          >
            {/* Animated shimmer */}
            <div className="absolute inset-0 -translate-x-full bg-gradient-to-r
                            from-transparent via-green-400/10 to-transparent
                            transition-transform duration-700
                            group-hover:translate-x-full" />
            <span className="relative flex items-center justify-center gap-2">
              <Radar
                size={16}
                className={forecasting ? "animate-spin" : ""}
              />
              {forecasting ? "Processing CDX…" : "Forecast 2025"}
            </span>
          </button>

          <p className="mt-2 text-center text-[10px] text-gray-600">
            CDX Fusion · Logit Transform · Sigmoid Activation
          </p>
        </div>
      </aside>

      {/* ─── MAP WATERMARK / BRANDING ─────────────────────────────────── */}
      <div className="absolute bottom-5 right-14 z-30 flex items-center gap-2
                      rounded-lg bg-black/50 px-3 py-1.5 backdrop-blur-md">
        <div className="h-1.5 w-1.5 rounded-full bg-green-400 shadow-sm shadow-green-400" />
        <span className="text-[10px] font-medium tracking-wider text-gray-400">
          AKS-E-SHAJR
        </span>
      </div>
    </div>
  );
}
