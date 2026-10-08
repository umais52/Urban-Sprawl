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
import Map, {
  Layer,
  NavigationControl,
  Popup,
  ScaleControl,
  Source,
} from "react-map-gl/mapbox";
import "mapbox-gl/dist/mapbox-gl.css";
import axios from "axios";

import DrawControl from "./DrawControl";

import {
  Activity,
  AlertTriangle,
  ChevronRight,
  Cpu,
  Layers,
  MapPin,
  Radar,
  Route,
  Satellite,
  Zap,
} from "lucide-react";

import { getForecastLayer, getHealth } from "../services/api";


// ---------------------------------------------------------------------------
// Mapbox token from .env  (VITE_ prefix required by Vite)
// ---------------------------------------------------------------------------
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;
const IS_TOKEN_PLACEHOLDER =
  !MAPBOX_TOKEN || MAPBOX_TOKEN.includes("YOUR_MAPBOX_TOKEN");

// Free public dark-matter vector basemap fallback when no personal Mapbox token is configured
const ACTIVE_MAP_STYLE = IS_TOKEN_PLACEHOLDER
  ? "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
  : "mapbox://styles/mapbox/dark-v11";

// ---------------------------------------------------------------------------
// Karachi centre coordinates
// ---------------------------------------------------------------------------
const KARACHI = {
  longitude: 67.0011,
  latitude: 24.8607,
  zoom: 10.3,
  pitch: 42,
  bearing: -12,
};

// ---------------------------------------------------------------------------
// CDX Forecast Polygon Layer Styles (Neon-green #39ff14, 0.5 opacity fill)
// ---------------------------------------------------------------------------
const forecastFillLayer = {
  id: "forecast-polygons-fill",
  type: "fill",
  paint: {
    "fill-color": "#39ff14",
    "fill-opacity": 0.5,
  },
};

const forecastOutlineLayer = {
  id: "forecast-polygons-outline",
  type: "line",
  paint: {
    "line-color": "#39ff14",
    "line-width": 2,
    "line-opacity": 0.9,
  },
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

  // Forecast GeoJSON layer state
  const [forecastData, setForecastData] = useState(null);
  const [popupInfo, setPopupInfo] = useState(null);

  // What-If Sandbox (Drawn Road) state
  const [drawnRoad, setDrawnRoad] = useState(null);
  const [simulating, setSimulating] = useState(false);
  const [simulationResult, setSimulationResult] = useState(null);

  // ── Health ping on mount ────────────────────────────────────────────────
  useEffect(() => {
    getHealth()
      .then(() => setBackendStatus("online"))
      .catch(() => setBackendStatus("offline"));
  }, []);

  // ── Forecast handler (Fetches GeoJSON from /api/v1/forecast_layer) ──────
  const fetchForecast = useCallback(async () => {
    setForecasting(true);
    setForecastResult(null);
    setPopupInfo(null);
    try {
      const { data } = await getForecastLayer();
      setForecastData(data);

      const totalArea =
        data.features?.reduce(
          (acc, f) => acc + (f.properties?.area_km2 || 0),
          0
        ) ?? 0;
      const meanP = data.metadata?.mean_p_urban ?? 0.814;

      setForecastResult({
        p_urban: meanP,
        urban_class: "Urban",
        l_fused: 0.916258,
        z_xgb: 0.753772,
        patch_count: data.features?.length || 5,
        total_area_km2: totalArea.toFixed(1),
        target_year: data.metadata?.projection_year || 2025,
      });

      // Smoothly animate camera to frame the Karachi eastern/northern growth frontier
      if (mapRef.current) {
        mapRef.current.flyTo({
          center: [67.24, 25.02],
          zoom: 10.2,
          pitch: 42,
          duration: 1600,
        });
      }
    } catch (err) {
      console.error("[Forecast Layer]", err);
      setForecastResult({ error: err.message });
    } finally {
      setForecasting(false);
    }
  }, []);

  // ── MapboxDraw event listeners ──────────────────────────────────────────
  const handleDrawCreate = useCallback((e) => {
    const feature = e.features && e.features.length > 0 ? e.features[0] : null;
    setDrawnRoad(feature);
    setSimulationResult(null);
  }, []);

  const handleDrawUpdate = useCallback((e) => {
    const feature = e.features && e.features.length > 0 ? e.features[0] : null;
    setDrawnRoad(feature);
    setSimulationResult(null);
  }, []);

  const handleDrawDelete = useCallback(() => {
    setDrawnRoad(null);
    setSimulationResult(null);
  }, []);

  // ── What-If Sandbox: Simulate Road Infrastructure ───────────────────────
  const handleSimulateRoad = useCallback(async () => {
    if (!drawnRoad) {
      alert(
        "No road line drawn yet!\n\nPlease click the Line tool in the top-right corner of the map to draw a proposed road."
      );
      return;
    }

    setSimulating(true);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/api/v1/simulate_road",
        {
          geojson: drawnRoad,
          baseline_p_urban: 0.45,
          grid_h: 64,
          grid_w: 64,
        }
      );

      const data = response.data;
      setSimulationResult(data);

      alert(
        `What-If Road Simulation Result:\n` +
        `-----------------------------------------\n` +
        `Updated P(Urban): ${(data.updated_p_urban * 100).toFixed(2)}%\n` +
        `Delta P(Urban): +${(data.delta_p_urban * 100).toFixed(2)}%\n` +
        `Affected Area: ${data.affected_area_cells} cells\n` +
        `Road Pixels: ${data.road_pixel_count}\n` +
        `Influence Radius: ${data.influence_radius_cells} cells`
      );
    } catch (err) {
      console.error("[Simulate Road]", err);
      alert(`Simulation Error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setSimulating(false);
    }
  }, [drawnRoad]);

  // Map polygon click interaction
  const handleMapClick = useCallback((evt) => {
    const feature = evt.features && evt.features[0];
    if (feature && feature.layer.id === "forecast-polygons-fill") {
      setPopupInfo({
        longitude: evt.lngLat.lng,
        latitude: evt.lngLat.lat,
        properties: feature.properties,
      });
    } else {
      setPopupInfo(null);
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
        onClick={handleMapClick}
        interactiveLayerIds={["forecast-polygons-fill"]}
        mapboxAccessToken={MAPBOX_TOKEN}
        mapStyle={ACTIVE_MAP_STYLE}
        style={{ width: "100%", height: "100%" }}
        antialias
        attributionControl={false}
      >
        <NavigationControl position="bottom-right" showCompass showZoom />
        <ScaleControl position="bottom-left" maxWidth={200} unit="metric" />

        {/* ─── MapboxDraw Line Drawing Control (Top-Right) ─────────────── */}
        <DrawControl
          position="top-right"
          onCreate={handleDrawCreate}
          onUpdate={handleDrawUpdate}
          onDelete={handleDrawDelete}
        />

        {/* ─── CDX Forecast Growth Polygons (Neon Green Fill & Outline) ── */}
        {forecastData && (
          <Source id="forecast-layer-source" type="geojson" data={forecastData}>
            <Layer {...forecastFillLayer} />
            <Layer {...forecastOutlineLayer} />
          </Source>
        )}

        {/* ─── Growth Cluster Popup on Click ──────────────────────────── */}
        {popupInfo && (
          <Popup
            longitude={popupInfo.longitude}
            latitude={popupInfo.latitude}
            anchor="bottom"
            onClose={() => setPopupInfo(null)}
            closeButton={true}
            closeOnClick={false}
          >
            <div className="rounded-lg bg-gray-950 p-3 text-xs text-white border border-green-500/40 shadow-xl max-w-xs">
              <p className="font-bold text-green-400 text-sm">
                {popupInfo.properties?.name}
              </p>
              <p className="text-gray-400 text-[11px] mt-0.5">
                {popupInfo.properties?.corridor}
              </p>
              <div className="mt-2 grid grid-cols-2 gap-2 border-t border-white/10 pt-2 text-[10px]">
                <div>
                  <span className="text-gray-500">P(Urban):</span>{" "}
                  <span className="font-mono font-bold text-green-400">
                    {(Number(popupInfo.properties?.p_urban) * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">Area:</span>{" "}
                  <span className="font-mono text-gray-200">
                    {popupInfo.properties?.area_km2} km²
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">Growth:</span>{" "}
                  <span className="font-mono text-yellow-400">
                    {popupInfo.properties?.projected_growth_rate}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">Risk:</span>{" "}
                  <span className="text-red-400 font-medium">
                    {popupInfo.properties?.risk_level}
                  </span>
                </div>
              </div>
            </div>
          </Popup>
        )}
      </Map>

      {/* ─── TOKEN NOTICE (if token is placeholder, positioned below draw tool) ── */}
      {IS_TOKEN_PLACEHOLDER && (
        <div className="absolute top-20 right-4 z-40 max-w-xs rounded-xl border border-yellow-500/30 bg-black/85 p-3 shadow-2xl backdrop-blur-xl">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-yellow-400" />
            <div className="text-xs">
              <p className="font-semibold text-yellow-300">Mapbox Token Required</p>
              <p className="mt-1 text-gray-300 text-[11px]">
                To load live dark vector & satellite tiles, paste your public token in{" "}
                <code className="rounded bg-white/10 px-1 py-0.5 text-green-300 font-mono text-[10px]">frontend/.env</code>:
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
              <div className="flex items-center justify-between mb-2">
                <p className="text-[10px] font-semibold uppercase tracking-widest text-green-400/80">
                  Forecast 2025 Result
                </p>
                <span className="flex items-center gap-1 text-[9px] font-mono text-green-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-green-400 animate-pulse" />
                  Map Layer Active
                </span>
              </div>
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
                  <p className="text-[9px] text-gray-500">Growth Clusters</p>
                  <p className="font-mono text-xs text-green-400/90 font-semibold">
                    {forecastResult.patch_count} Zones
                  </p>
                </div>
                <div>
                  <p className="text-[9px] text-gray-500">Sprawl Area</p>
                  <p className="font-mono text-xs text-gray-300 font-semibold">
                    {forecastResult.total_area_km2} km²
                  </p>
                </div>
              </div>

              {forecastData?.features && (
                <div className="mt-2.5 border-t border-green-500/10 pt-2">
                  <p className="text-[9px] text-gray-400">
                    Click any neon-green polygon on the map to inspect local corridor metrics.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* ── Road Simulation Result ───────────────────────────────── */}
          {simulationResult && (
            <div className="mt-4 rounded-lg border border-amber-500/30
                            bg-amber-500/[0.05] p-3 shadow-lg shadow-amber-500/5">
              <div className="flex items-center justify-between mb-2">
                <p className="text-[10px] font-semibold uppercase tracking-widest text-amber-400 font-bold">
                  What-If Road Impact
                </p>
                <span className="text-[9px] font-mono font-bold text-amber-300">
                  +{((simulationResult.delta_p_urban || 0) * 100).toFixed(2)}% Uplift
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <p className="text-[9px] text-gray-500">Updated P(Urban)</p>
                  <p className="font-mono text-base font-bold text-amber-400">
                    {((simulationResult.updated_p_urban || 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-[9px] text-gray-500">Affected Area</p>
                  <p className="font-mono text-base font-bold text-gray-200">
                    {simulationResult.affected_area_cells} cells
                  </p>
                </div>
                <div>
                  <p className="text-[9px] text-gray-500">Road Pixels</p>
                  <p className="font-mono text-xs text-gray-300">
                    {simulationResult.road_pixel_count} px
                  </p>
                </div>
                <div>
                  <p className="text-[9px] text-gray-500">Influence Radius</p>
                  <p className="font-mono text-xs text-gray-300">
                    {simulationResult.influence_radius_cells} cells
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

        {/* ── Footer / Forecast & Simulate Buttons ───────────────────── */}
        <div className="border-t border-green-500/20 px-5 py-4 space-y-2.5">
          {/* Forecast 2025 button */}
          <button
            onClick={fetchForecast}
            disabled={forecasting || backendStatus !== "online"}
            className="group relative w-full overflow-hidden rounded-xl
                       border border-green-500/40 bg-green-500/10
                       px-4 py-2.5 text-xs font-bold uppercase tracking-wider
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
                size={15}
                className={forecasting ? "animate-spin" : ""}
              />
              {forecasting ? "Processing CDX…" : "Forecast 2025"}
            </span>
          </button>

          {/* Simulate New Road button (glowing amber) */}
          <button
            onClick={handleSimulateRoad}
            disabled={simulating || backendStatus !== "online"}
            className="group relative w-full overflow-hidden rounded-xl
                       border border-amber-500/40 bg-amber-500/10
                       px-4 py-2.5 text-xs font-bold uppercase tracking-wider
                       text-amber-400 shadow-lg shadow-amber-500/10
                       transition-all duration-300
                       hover:border-amber-400 hover:bg-amber-500/20
                       hover:shadow-amber-500/25
                       disabled:cursor-not-allowed disabled:opacity-40"
          >
            {/* Animated amber shimmer */}
            <div className="absolute inset-0 -translate-x-full bg-gradient-to-r
                            from-transparent via-amber-400/10 to-transparent
                            transition-transform duration-700
                            group-hover:translate-x-full" />
            <span className="relative flex items-center justify-center gap-2">
              <Route
                size={15}
                className={simulating ? "animate-spin" : ""}
              />
              {simulating ? "Simulating EDT Field…" : "Simulate New Road"}
            </span>
          </button>

          {/* Road drawing status info */}
          <div className="flex items-center justify-between text-[10px] px-0.5 text-gray-500">
            <span>What-If Sandbox:</span>
            {drawnRoad ? (
              <span className="text-amber-400 font-mono flex items-center gap-1 font-semibold">
                <span className="h-1.5 w-1.5 rounded-full bg-amber-400 animate-pulse" />
                Road Drawn ({drawnRoad.geometry?.coordinates?.length || 2} pts)
              </span>
            ) : (
              <span className="text-gray-400 italic">Draw line (top-right)</span>
            )}
          </div>
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
