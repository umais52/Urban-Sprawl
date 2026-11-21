/**
 * MapInterface — Sprawl Vision analytical workspace
 * ==================================================
 * Refined GIS interface:
 *   - Strict Carbon/Zinc neutral palette (#0C0C0E / #141417 / #26262B)
 *   - Unified analytical Amber (#D97706 / #F59E0B) matching urban raster
 *   - Bottom timeline is single source of truth for scrubbing & playback
 *   - Repurposed sidebar History tab: Temporal Summary Card + opacity/visibility
 *   - Centered floating bottom timeline dock (unobstructed Mapbox controls)
 *   - Context-aware cartographic legend with uniform 12x12 raster swatches
 *   - Drawing tools scoped to Simulate workflow
 *   - Precision GIS coordinate readout footer
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  AlertTriangle,
  Clock,
  Eye,
  EyeOff,
  Layers,
  PanelLeftClose,
  PanelLeftOpen,
  Pause,
  Play,
  RotateCcw,
  Route,
  Sliders,
  TrendingUp,
} from "lucide-react";

import { getForecastLayer, getHealth, getHistory } from "../services/api";

// ---------------------------------------------------------------------------
// Mapbox token from .env (VITE_ prefix required by Vite)
// ---------------------------------------------------------------------------
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;
const IS_TOKEN_PLACEHOLDER =
  !MAPBOX_TOKEN || MAPBOX_TOKEN.includes("YOUR_MAPBOX_TOKEN");

// Free public dark-matter vector basemap fallback when no personal Mapbox token is configured
const ACTIVE_MAP_STYLE = IS_TOKEN_PLACEHOLDER
  ? "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
  : "mapbox://styles/mapbox/dark-v11";

// ---------------------------------------------------------------------------
// Karachi centre — analytical 2D view, 0° pitch
// ---------------------------------------------------------------------------
const KARACHI = {
  longitude: 67.0011,
  latitude: 24.8607,
  zoom: 10.3,
  pitch: 0,
  bearing: 0,
};

// ---------------------------------------------------------------------------
// Helper: Generate polygon buffer ring around road coordinates for delta
// ---------------------------------------------------------------------------
function bufferLineCoordinates(coords, bufferDeg = 0.012) {
  if (!coords || coords.length < 2) return [];
  const left = [];
  const right = [];

  for (let i = 0; i < coords.length; i++) {
    let dx = 0;
    let dy = 0;

    if (i === 0) {
      dx = coords[1][0] - coords[0][0];
      dy = coords[1][1] - coords[0][1];
    } else if (i === coords.length - 1) {
      dx = coords[i][0] - coords[i - 1][0];
      dy = coords[i][1] - coords[i - 1][1];
    } else {
      dx = coords[i + 1][0] - coords[i - 1][0];
      dy = coords[i + 1][1] - coords[i - 1][1];
    }

    const len = Math.sqrt(dx * dx + dy * dy) || 1e-6;
    const nx = -dy / len;
    const ny = dx / len;

    left.push([coords[i][0] + nx * bufferDeg, coords[i][1] + ny * bufferDeg]);
    right.push([coords[i][0] - nx * bufferDeg, coords[i][1] - ny * bufferDeg]);
  }

  const ring = [...left, ...right.reverse()];
  ring.push(ring[0]);
  return ring;
}

// ---------------------------------------------------------------------------
// Available timeline years
// ---------------------------------------------------------------------------
const TIMELINE_YEARS = [1995, 2000, 2005, 2010, 2015, 2020, 2025];

const HISTORY_SWATCH_FALLBACK = "#D97706";

function formatKm2(value) {
  return Number(value).toLocaleString("en-US", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
}

function formatCount(value) {
  return Number(value).toLocaleString("en-US");
}

// ---------------------------------------------------------------------------
// Workflow tabs
// ---------------------------------------------------------------------------
const WORKFLOWS = {
  HISTORY: "history",
  FORECAST: "forecast",
  SIMULATE: "simulate",
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function MapInterface() {
  // Map viewport state
  const [viewState, setViewState] = useState(KARACHI);
  const mapRef = useRef(null);

  // UI state
  const [panelOpen, setPanelOpen] = useState(true);
  const [activeWorkflow, setActiveWorkflow] = useState(WORKFLOWS.HISTORY);
  const [backendStatus, setBackendStatus] = useState(null); // null | "online" | "offline"
  const [forecasting, setForecasting] = useState(false);
  const [forecastResult, setForecastResult] = useState(null);

  // Forecast GeoJSON layer state
  const [forecastData, setForecastData] = useState(null);
  const [popupInfo, setPopupInfo] = useState(null);

  // Temporal Time-Lapse state
  const [currentYear, setCurrentYear] = useState(2020);
  const [historyData, setHistoryData] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [historyOpacity, setHistoryOpacity] = useState(0.5);
  const [historyVisible, setHistoryVisible] = useState(true);
  const [labelLayerId, setLabelLayerId] = useState(null);
  // `Map` is the react-map-gl component, so the year cache uses the global constructor.
  const historyCacheRef = useRef(new globalThis.Map());
  const historyInflightRef = useRef(new globalThis.Map());

  // What-If Sandbox (Drawn Road) state
  const [drawnRoad, setDrawnRoad] = useState(null);
  const [simulating, setSimulating] = useState(false);
  const [simulationResult, setSimulationResult] = useState(null);
  const [simulationData, setSimulationData] = useState(null);

  // ── Health ping on mount ────────────────────────────────────────────────
  useEffect(() => {
    getHealth()
      .then(() => setBackendStatus("online"))
      .catch(() => setBackendStatus("offline"));
  }, []);

  // ── Fetch Temporal History overlay, then prefetch the other years ────────
  const loadHistoryYear = useCallback((year) => {
    const cached = historyCacheRef.current.get(year);
    if (cached) return Promise.resolve(cached);
    const pending = historyInflightRef.current.get(year);
    if (pending) return pending;

    const request = getHistory(year)
      .then(({ data }) => {
        historyCacheRef.current.set(year, data);
        return data;
      })
      .finally(() => {
        historyInflightRef.current.delete(year);
      });
    historyInflightRef.current.set(year, request);
    return request;
  }, []);

  useEffect(() => {
    let isSubscribed = true;
    const cached = historyCacheRef.current.get(currentYear);
    if (cached) setHistoryData(cached);

    loadHistoryYear(currentYear)
      .then((data) => {
        if (isSubscribed) setHistoryData(data);
        for (const year of TIMELINE_YEARS) {
          if (year === currentYear) continue;
          loadHistoryYear(year).catch((err) => {
            console.error(`[History ${year}]`, err);
          });
        }
      })
      .catch((err) => {
        console.error(`[History ${currentYear}]`, err);
      });

    return () => {
      isSubscribed = false;
    };
  }, [currentYear, loadHistoryYear]);

  const handleMapLoad = useCallback((evt) => {
    const map = evt?.target || mapRef.current?.getMap?.();
    const firstSymbol = map
      ?.getStyle()
      ?.layers?.find((layer) => layer.type === "symbol");
    if (firstSymbol) setLabelLayerId(firstSymbol.id);
  }, []);

  // ── Auto-play time-lapse playback loop ───────────────────────────────────
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setCurrentYear((prev) => (prev >= 2025 ? 1995 : prev + 5));
    }, 1500);
    return () => clearInterval(interval);
  }, [isPlaying]);

  // ── Forecast handler ─────────────────────────────────────────────────────
  const fetchForecast = useCallback(async () => {
    setForecasting(true);
    setForecastResult(null);
    setPopupInfo(null);
    try {
      const { data } = await getForecastLayer();
      setForecastData(data);

      const totalArea =
        data.metadata?.total_area_km2 ??
        data.features?.reduce(
          (acc, f) => acc + (f.properties?.area_km2 || 0),
          0
        ) ?? 0;
      const meanP = data.metadata?.mean_p_urban ?? 0.814;

      setForecastResult({
        p_urban: meanP,
        urban_class: "Urban",
        l_fused: data.metadata?.mean_l_fused ?? 0.916258,
        z_xgb: data.metadata?.mean_z_xgb ?? 0.753772,
        patch_count: data.features?.length || 5,
        total_area_km2: Number(totalArea).toFixed(1),
        target_year: data.metadata?.projection_year || 2030,
        inference_time_ms: data.metadata?.inference_time_ms,
        threshold: data.metadata?.threshold,
        fusion_model: data.metadata?.fusion_model,
      });

      // Smoothly animate camera to frame the Karachi eastern/northern growth frontier
      if (mapRef.current) {
        mapRef.current.flyTo({
          center: [67.24, 25.02],
          zoom: 10.2,
          pitch: 0,
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
    setSimulationData(null);
  }, []);

  const handleDrawUpdate = useCallback((e) => {
    const feature = e.features && e.features.length > 0 ? e.features[0] : null;
    setDrawnRoad(feature);
    setSimulationResult(null);
    setSimulationData(null);
  }, []);

  const handleDrawDelete = useCallback(() => {
    setDrawnRoad(null);
    setSimulationResult(null);
    setSimulationData(null);
  }, []);

  // ── What-If Sandbox: Simulate Road Infrastructure ───────────────────────
  const handleSimulateRoad = useCallback(async () => {
    if (!drawnRoad) {
      alert(
        "No road line drawn yet.\n\nUse the line tool in the top-right corner of the map to trace a proposed road corridor."
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

      let geojson = data;
      if (data && (data.type === "FeatureCollection" || data.type === "Feature")) {
        geojson = data;
      } else if (data?.geojson && (data.geojson.type === "FeatureCollection" || data.geojson.type === "Feature")) {
        geojson = data.geojson;
      } else if (drawnRoad?.geometry?.coordinates) {
        const polyRing = bufferLineCoordinates(drawnRoad.geometry.coordinates, 0.012);
        geojson = {
          type: "FeatureCollection",
          metadata: { ...data },
          features: [
            {
              type: "Feature",
              id: "simulated_road_delta_polygon",
              properties: {
                name: "Simulated Urban Growth Delta",
                corridor: "Proposed Road Infrastructure Corridor",
                delta_p_urban: data.delta_p_urban,
                updated_p_urban: data.updated_p_urban,
                affected_area_cells: data.affected_area_cells,
                road_pixel_count: data.road_pixel_count,
                influence_radius_cells: data.influence_radius_cells,
              },
              geometry: {
                type: "Polygon",
                coordinates: [polyRing],
              },
            },
          ],
        };
      }

      setSimulationData(geojson);
    } catch (err) {
      console.error("[Simulate Road]", err);
      setSimulationResult({ error: err.message });
    } finally {
      setSimulating(false);
    }
  }, [drawnRoad]);

  // ── Map polygon click interaction ───────────────────────────────────────
  const handleMapClick = useCallback((evt) => {
    const feature = evt.features && evt.features[0];
    if (
      feature &&
      (feature.layer.id === "forecast-polygons-fill" ||
        feature.layer.id === "simulation-polygons-fill")
    ) {
      setPopupInfo({
        longitude: evt.lngLat.lng,
        latitude: evt.lngLat.lat,
        properties: feature.properties,
        layerId: feature.layer.id,
      });
    } else {
      setPopupInfo(null);
    }
  }, []);

  const getLayerLabel = (layerId) => {
    if (layerId === "forecast-polygons-fill") return "Forecast";
    if (layerId === "simulation-polygons-fill") return "Scenario";
    return "Observed";
  };

  const getLayerColor = (layerId) => {
    if (layerId === "forecast-polygons-fill") return "#E06356";
    if (layerId === "simulation-polygons-fill") return "#5B98D4";
    return "#D97706";
  };

  // ── Forecast and simulation styles stay on GeoJSON fill/line layers ─────
  const forecastFillLayer = {
    id: "forecast-polygons-fill",
    type: "fill",
    paint: {
      "fill-color": "#E06356",
      "fill-opacity": 0.45,
    },
  };

  const forecastOutlineLayer = {
    id: "forecast-polygons-outline",
    type: "line",
    paint: {
      "line-color": "#E06356",
      "line-width": 1.5,
      "line-opacity": 0.85,
    },
  };

  const simulationFillLayer = {
    id: "simulation-polygons-fill",
    type: "fill",
    paint: {
      "fill-color": "#5B98D4",
      "fill-opacity": 0.5,
    },
  };

  const simulationOutlineLayer = {
    id: "simulation-polygons-outline",
    type: "line",
    paint: {
      "line-color": "#5B98D4",
      "line-width": 1.5,
      "line-opacity": 0.85,
    },
  };

  const historyStatsReady = historyData?.year === currentYear;

  const observedAreaKm2 = useMemo(() => {
    if (!historyStatsReady || historyData?.area_km2 == null) return "—";
    return formatKm2(historyData.area_km2);
  }, [historyData, historyStatsReady]);

  const observedPatchesCount = useMemo(() => {
    if (!historyStatsReady || historyData?.patches == null) return "—";
    return formatCount(historyData.patches);
  }, [historyData, historyStatsReady]);

  const spatialResolutionLabel = useMemo(() => {
    if (!historyStatsReady || historyData?.resolution_m == null) return "—";
    return `${historyData.resolution_m} m × ${historyData.resolution_m} m`;
  }, [historyData, historyStatsReady]);

  const showHistoryLegend =
    activeWorkflow === WORKFLOWS.HISTORY && Boolean(historyData?.image_url) && historyVisible;
  const historySwatchColor = historyData?.color || HISTORY_SWATCH_FALLBACK;

  return (
    <div
      className="relative h-screen w-screen overflow-hidden font-sans select-none"
      style={{ background: "#0C0C0E" }}
    >
      {/* ─── MAP CANVAS ─────────────────────────────────────────────────── */}
      <Map
        ref={mapRef}
        {...viewState}
        onMove={(evt) => setViewState(evt.viewState)}
        onClick={handleMapClick}
        onLoad={handleMapLoad}
        interactiveLayerIds={[
          "forecast-polygons-fill",
          "simulation-polygons-fill",
        ]}
        mapboxAccessToken={MAPBOX_TOKEN}
        mapStyle={ACTIVE_MAP_STYLE}
        style={{ width: "100%", height: "100%" }}
        antialias
      >
        <NavigationControl position="bottom-right" showCompass showZoom />
        <ScaleControl position="bottom-left" maxWidth={180} unit="metric" />

        {/* ─── MapboxDraw: Only mounted & visible in Simulate workflow ─── */}
        {activeWorkflow === WORKFLOWS.SIMULATE && (
          <DrawControl
            position="top-right"
            onCreate={handleDrawCreate}
            onUpdate={handleDrawUpdate}
            onDelete={handleDrawDelete}
          />
        )}

        {/* ─── Observed History raster (History tab only) ─── */}
        {activeWorkflow === WORKFLOWS.HISTORY && historyData?.image_url && (
          <Source
            id="history-urban-source"
            type="image"
            url={historyData.image_url}
            coordinates={historyData.coordinates}
          >
            <Layer
              id="history-urban-layer"
              type="raster"
              beforeId={labelLayerId || undefined}
              layout={{ visibility: historyVisible ? "visible" : "none" }}
              paint={{
                "raster-opacity": historyOpacity,
                "raster-resampling": "nearest",
                "raster-fade-duration": 0,
              }}
            />
          </Source>
        )}

        {/* ─── CDX Forecast Growth Layer ─── */}
        {forecastData && (
          <Source id="forecast-layer-source" type="geojson" data={forecastData}>
            <Layer {...forecastFillLayer} />
            <Layer {...forecastOutlineLayer} />
          </Source>
        )}

        {/* ─── What-If Road Simulation Delta Layer ─── */}
        {simulationData && (
          <Source id="simulation-layer-source" type="geojson" data={simulationData}>
            <Layer {...simulationFillLayer} />
            <Layer {...simulationOutlineLayer} />
          </Source>
        )}

        {/* ─── Feature Popup on Click ───────────────────────────────────── */}
        {popupInfo && (
          <Popup
            longitude={popupInfo.longitude}
            latitude={popupInfo.latitude}
            anchor="bottom"
            onClose={() => setPopupInfo(null)}
            closeButton={true}
            closeOnClick={false}
          >
            <div className="p-3 text-[13px] max-w-[270px]">
              <div className="flex items-start justify-between gap-2 mb-2">
                <p
                  className="font-semibold text-[13px] leading-tight"
                  style={{ color: getLayerColor(popupInfo.layerId) }}
                >
                  {popupInfo.properties?.name || "Feature"}
                </p>
                <span
                  className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-mono font-medium"
                  style={{
                    background: `${getLayerColor(popupInfo.layerId)}18`,
                    color: getLayerColor(popupInfo.layerId),
                    border: `1px solid ${getLayerColor(popupInfo.layerId)}33`,
                  }}
                >
                  {popupInfo.properties?.year
                    ? `Observed ${popupInfo.properties.year}`
                    : getLayerLabel(popupInfo.layerId)}
                </span>
              </div>

              {popupInfo.properties?.corridor && (
                <p className="text-[11px] mb-2" style={{ color: "#A1A1AA" }}>
                  {popupInfo.properties.corridor}
                </p>
              )}
              {popupInfo.properties?.stage && (
                <p className="text-[11px] mb-2" style={{ color: "#A1A1AA" }}>
                  {popupInfo.properties.stage}
                </p>
              )}

              <div
                className="grid grid-cols-2 gap-x-4 gap-y-2 pt-2"
                style={{ borderTop: "1px solid #26262B" }}
              >
                {popupInfo.properties?.p_urban !== undefined && (
                  <div>
                    <span className="text-[10px] uppercase tracking-wider block" style={{ color: "#71717A" }}>
                      P(Urban)
                    </span>
                    <p className="font-mono text-[13px] font-semibold text-[#F4F4F6]">
                      {(Number(popupInfo.properties.p_urban) * 100).toFixed(1)}%
                    </p>
                  </div>
                )}
                {popupInfo.properties?.delta_p_urban !== undefined && (
                  <div>
                    <span className="text-[10px] uppercase tracking-wider block" style={{ color: "#71717A" }}>
                      Delta
                    </span>
                    <p className="font-mono text-[13px] font-semibold" style={{ color: "#5B98D4" }}>
                      +{((Number(popupInfo.properties.delta_p_urban)) * 100).toFixed(2)}%
                    </p>
                  </div>
                )}
                {popupInfo.properties?.affected_area_cells !== undefined && (
                  <div>
                    <span className="text-[10px] uppercase tracking-wider block" style={{ color: "#71717A" }}>
                      Cells
                    </span>
                    <p className="font-mono text-[12px] text-[#F4F4F6]">
                      {popupInfo.properties.affected_area_cells}
                    </p>
                  </div>
                )}
                {popupInfo.properties?.area_km2 !== undefined && (
                  <div>
                    <span className="text-[10px] uppercase tracking-wider block" style={{ color: "#71717A" }}>
                      Area
                    </span>
                    <p className="font-mono text-[12px] text-[#F4F4F6]">
                      {popupInfo.properties?.area_km2} km²
                    </p>
                  </div>
                )}
                {popupInfo.properties?.projected_growth_rate && (
                  <div>
                    <span className="text-[10px] uppercase tracking-wider block" style={{ color: "#71717A" }}>
                      Growth Rate
                    </span>
                    <p className="font-mono text-[12px] text-[#F4F4F6]">
                      {popupInfo.properties?.projected_growth_rate}
                    </p>
                  </div>
                )}
                {popupInfo.properties?.risk_level && (
                  <div>
                    <span className="text-[10px] uppercase tracking-wider block" style={{ color: "#71717A" }}>
                      Sprawl
                    </span>
                    <p className="text-[12px] font-medium text-[#F4F4F6]">
                      {popupInfo.properties?.risk_level}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </Popup>
        )}
      </Map>

      {/* ─── MAPBOX TOKEN WARNING ───────────────────────────────────────── */}
      {IS_TOKEN_PLACEHOLDER && (
        <div
          className="absolute top-4 right-4 z-40 max-w-[270px] rounded-md p-3"
          style={{
            background: "#141417",
            border: "1px solid #D97706",
            boxShadow: "0 4px 16px rgba(0,0,0,0.4)",
          }}
        >
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-[#F59E0B]" />
            <div className="text-[12px]">
              <p className="font-medium text-[#F59E0B]">Mapbox token required</p>
              <p className="mt-1 text-[#A1A1AA]">
                Set your personal token in{" "}
                <code className="rounded px-1 py-0.5 text-[11px] font-mono bg-[#1F1F23] text-[#F4F4F6]">
                  frontend/.env
                </code>
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ─── PANEL TOGGLE (collapsed state) ─────────────────────────────── */}
      {!panelOpen && (
        <button
          onClick={() => setPanelOpen(true)}
          className="absolute left-4 top-4 z-50 flex items-center justify-center
                     rounded-md p-2 transition-colors hover:bg-[#1F1F23]"
          style={{
            background: "rgba(12, 12, 14, 0.94)",
            border: "1px solid #26262B",
            color: "#A1A1AA",
            backdropFilter: "blur(8px)",
          }}
          aria-label="Open analysis panel"
          title="Open analysis panel"
        >
          <PanelLeftOpen size={16} />
        </button>
      )}

      {/* ─── ANALYSIS PANEL ────────────────────────────────────────────── */}
      <aside
        className={`
          absolute left-0 top-0 bottom-0 z-40 flex flex-col
          overflow-hidden panel-transition
          ${panelOpen ? "w-[280px] opacity-100" : "w-0 opacity-0 pointer-events-none"}
        `}
        style={{
          background: "#0C0C0E",
          borderRight: "1px solid #26262B",
        }}
      >
        {/* ── Header ────────────────────────────────────────────────── */}
        <div
          className="flex items-center justify-between px-4 py-3"
          style={{ borderBottom: "1px solid #26262B" }}
        >
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <h1 className="text-[14px] font-semibold tracking-tight text-[#F4F4F6]">
                Sprawl Vision
              </h1>
              {/* Status pill badge */}
              <div
                className="flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-mono"
                style={{
                  background: "#141417",
                  border: "1px solid #26262B",
                  color: "#A1A1AA",
                }}
                title={backendStatus === "online" ? "FastAPI backend active" : "Backend disconnected"}
              >
                <span
                  className="h-1.5 w-1.5 rounded-full"
                  style={{
                    background:
                      backendStatus === "online"
                        ? "#10B981"
                        : backendStatus === "offline"
                          ? "#EF4444"
                          : "#F59E0B",
                  }}
                />
                <span>{backendStatus === "online" ? "Online" : "Offline"}</span>
              </div>
            </div>
            <p className="text-[10px] font-medium text-[#71717A] tracking-wide mt-0.5">
              Urban Sprawl Intelligence · Karachi
            </p>
          </div>
          <button
            onClick={() => setPanelOpen(false)}
            className="rounded p-1 transition-colors hover:bg-[#1F1F23] text-[#71717A] hover:text-[#F4F4F6]"
            aria-label="Close panel"
          >
            <PanelLeftClose size={16} />
          </button>
        </div>

        {/* ── Workflow Tabs ──────────────────────────────────────────── */}
        <div className="flex" style={{ borderBottom: "1px solid #26262B" }}>
          {[
            { id: WORKFLOWS.HISTORY, label: "History", icon: <Clock size={13} /> },
            { id: WORKFLOWS.FORECAST, label: "Forecast", icon: <TrendingUp size={13} /> },
            { id: WORKFLOWS.SIMULATE, label: "Simulate", icon: <Route size={13} /> },
          ].map((tab) => {
            const isActive = activeWorkflow === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveWorkflow(tab.id)}
                className="flex-1 flex items-center justify-center gap-1.5 py-2.5 text-[11px] font-medium transition-colors relative"
                style={{
                  color: isActive ? "#F4F4F6" : "#71717A",
                  background: isActive ? "#141417" : "transparent",
                }}
              >
                {tab.icon}
                {tab.label}
                {isActive && (
                  <span
                    className="absolute bottom-0 left-2 right-2 h-[2px] rounded-full"
                    style={{ background: "#F59E0B" }}
                  />
                )}
              </button>
            );
          })}
        </div>

        {/* ── Workflow Content (scrollable) ──────────────────────────── */}
        <div className="flex-1 overflow-y-auto">
          {/* === HISTORY WORKFLOW (Repurposed: No duplicate timeline controls) === */}
          {activeWorkflow === WORKFLOWS.HISTORY && (
            <div className="px-4 py-3 space-y-4">
              <div>
                <p className="text-[11px] leading-relaxed text-[#A1A1AA]">
                  Landsat/Sentinel derived 30m binary urban extent across the Karachi metropolitan region.
                </p>
              </div>

              {/* Temporal Summary Card */}
              <div
                className="rounded-md p-3 space-y-3"
                style={{
                  background: "#141417",
                  border: "1px solid #26262B",
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-[#71717A]">
                    Observed Extent
                  </span>
                  <span
                    className="font-mono text-[12px] font-semibold px-2 py-0.5 rounded"
                    style={{
                      background: "rgba(217, 119, 6, 0.15)",
                      color: "#F59E0B",
                      border: "1px solid rgba(217, 119, 6, 0.3)",
                    }}
                  >
                    {currentYear}
                  </span>
                </div>

                {/* Primary Metric: Observed Area */}
                <div>
                  <span className="text-[10px] text-[#71717A] block mb-0.5">
                    Observed Urban Area
                  </span>
                  <div className="flex items-baseline gap-1.5">
                    <span className="text-[28px] font-mono font-semibold tracking-tight text-[#F4F4F6] leading-none">
                      {observedAreaKm2}
                    </span>
                    <span className="text-[12px] font-mono text-[#A1A1AA]">km²</span>
                  </div>
                </div>

                {/* Secondary Metrics Grid */}
                <div
                  className="grid grid-cols-2 gap-2 pt-2.5"
                  style={{ borderTop: "1px solid #26262B" }}
                >
                  <div>
                    <span className="text-[10px] text-[#71717A] block">Detected Patches</span>
                    <span className="font-mono text-[13px] font-medium text-[#F4F4F6]">
                      {observedPatchesCount}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-[#71717A] block">Spatial Resolution</span>
                    <span className="font-mono text-[13px] font-medium text-[#F4F4F6]">
                      {spatialResolutionLabel}
                    </span>
                  </div>
                </div>
              </div>

              {/* Layer Controls: Opacity & Visibility */}
              <div
                className="rounded-md p-3 space-y-3"
                style={{
                  background: "#141417",
                  border: "1px solid #26262B",
                }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <Sliders size={12} className="text-[#A1A1AA]" />
                    <span className="text-[11px] font-medium text-[#F4F4F6]">
                      Layer Controls
                    </span>
                  </div>
                  <button
                    onClick={() => setHistoryVisible(!historyVisible)}
                    className="p-1 rounded text-[#A1A1AA] hover:text-[#F4F4F6] hover:bg-[#1F1F23] transition-colors"
                    title={historyVisible ? "Hide history layer" : "Show history layer"}
                  >
                    {historyVisible ? <Eye size={14} /> : <EyeOff size={14} className="text-[#71717A]" />}
                  </button>
                </div>

                <div>
                  <div className="flex justify-between text-[10px] mb-1.5">
                    <span className="text-[#71717A]">Layer Opacity</span>
                    <span className="font-mono text-[#A1A1AA]">
                      {Math.round(historyOpacity * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={historyOpacity}
                    disabled={!historyVisible}
                    onChange={(e) => setHistoryOpacity(Number(e.target.value))}
                    className="w-full h-1"
                    aria-label="Layer opacity slider"
                  />
                </div>
              </div>

              <div className="text-[10px] text-[#71717A] leading-normal pt-1">
                Tip: Use the bottom dock to scrub through 1995–2025 or toggle time-lapse playback.
              </div>
            </div>
          )}

          {/* === FORECAST WORKFLOW === */}
          {activeWorkflow === WORKFLOWS.FORECAST && (
            <div className="px-4 py-3 space-y-3">
              <p className="text-[11px] text-[#A1A1AA] leading-relaxed">
                Generate 2030 sprawl prediction using the CDX Logit Fusion model (CNN-LSTM + XGBoost + Contextual Attention).
              </p>

              <button
                onClick={fetchForecast}
                disabled={forecasting || backendStatus !== "online"}
                className="w-full rounded-md py-2.5 text-[12px] font-semibold transition-all
                           disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2
                           shadow-sm"
                style={{
                  background: "#D97706",
                  color: "#F4F4F6",
                }}
                onMouseOver={(e) => {
                  if (!forecasting && backendStatus === "online") {
                    e.currentTarget.style.background = "#B45309";
                  }
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = "#D97706";
                }}
              >
                {forecasting ? (
                  <>
                    <span className="inline-block w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Running inference…
                  </>
                ) : (
                  <>
                    <TrendingUp size={14} />
                    Run 2030 forecast
                  </>
                )}
              </button>

              {backendStatus !== "online" && backendStatus !== null && (
                <p className="text-[11px] text-[#EF4444]">
                  Backend server is offline. Run uvicorn backend.main:app to enable forecasts.
                </p>
              )}

              {/* Forecast results */}
              {forecastResult && !forecastResult.error && (
                <div
                  className="rounded-md p-3 space-y-3 mt-3"
                  style={{
                    background: "#141417",
                    border: "1px solid #26262B",
                  }}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-semibold text-[#E06356]">
                      Forecast {forecastResult.target_year}
                    </span>
                    <span className="text-[10px] font-mono text-[#71717A]">
                      {forecastResult.patch_count} patches
                    </span>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between items-baseline text-[11px]">
                      <span className="text-[#71717A]">Mean P(Urban)</span>
                      <span className="font-mono text-[14px] font-semibold text-[#F4F4F6]">
                        {(forecastResult.p_urban * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-[11px]">
                      <span className="text-[#71717A]">Predicted Area</span>
                      <span className="font-mono text-[#F4F4F6]">
                        {forecastResult.total_area_km2} km²
                      </span>
                    </div>
                    <div className="flex justify-between text-[11px]">
                      <span className="text-[#71717A]">Classification</span>
                      <span className="font-mono text-[#F4F4F6]">
                        {forecastResult.urban_class}
                      </span>
                    </div>
                  </div>

                  {/* Expandable Model Details */}
                  <details className="pt-2 border-t border-[#26262B]">
                    <summary className="text-[10px] text-[#71717A] cursor-pointer hover:text-[#A1A1AA] transition-colors">
                      Technical Model Logs
                    </summary>
                    <div className="mt-2 space-y-1.5 pl-2 border-l border-[#26262B] text-[10px]">
                      <div className="flex justify-between">
                        <span className="text-[#71717A]">L_fused</span>
                        <span className="font-mono text-[#A1A1AA]">
                          {forecastResult.l_fused?.toFixed(4) ?? "—"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[#71717A]">z_XGB</span>
                        <span className="font-mono text-[#A1A1AA]">
                          {forecastResult.z_xgb?.toFixed(4) ?? "—"}
                        </span>
                      </div>
                      {forecastResult.inference_time_ms && (
                        <div className="flex justify-between">
                          <span className="text-[#71717A]">Latency</span>
                          <span className="font-mono text-[#A1A1AA]">
                            {forecastResult.inference_time_ms} ms
                          </span>
                        </div>
                      )}
                      {forecastResult.fusion_model && (
                        <div className="text-[#71717A] pt-1 text-[9px] leading-tight">
                          {forecastResult.fusion_model}
                        </div>
                      )}
                    </div>
                  </details>
                </div>
              )}

              {forecastResult?.error && (
                <div
                  className="rounded-md p-2.5 text-[11px]"
                  style={{
                    background: "rgba(239, 68, 68, 0.1)",
                    border: "1px solid #EF4444",
                    color: "#EF4444",
                  }}
                >
                  {forecastResult.error}
                </div>
              )}
            </div>
          )}

          {/* === SIMULATE WORKFLOW === */}
          {activeWorkflow === WORKFLOWS.SIMULATE && (
            <div className="px-4 py-3 space-y-3">
              <p className="text-[11px] text-[#A1A1AA] leading-relaxed">
                Draw a hypothetical road corridor on the map, then compute the resulting urbanization probability uplift.
              </p>

              {/* Drawing Toolbar Card */}
              <div
                className="rounded-md p-3"
                style={{
                  background: "#141417",
                  border: "1px solid #26262B",
                }}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-[#71717A]">
                    Corridor Status
                  </span>
                  {drawnRoad && (
                    <button
                      onClick={handleDrawDelete}
                      className="flex items-center gap-1 text-[10px] text-[#EF4444] hover:underline"
                    >
                      <RotateCcw size={10} />
                      Reset
                    </button>
                  )}
                </div>

                {drawnRoad ? (
                  <div className="flex items-center gap-2 text-[12px] text-[#5B98D4] font-medium">
                    <Route size={14} />
                    <span>
                      Corridor drawn ({drawnRoad.geometry?.coordinates?.length || 2} vertices)
                    </span>
                  </div>
                ) : (
                  <div>
                    <p className="text-[11px] text-[#F4F4F6] font-medium">No road active</p>
                    <p className="text-[10px] text-[#71717A] mt-1">
                      Use the line tool in the top-right toolbar of the map canvas to trace a proposed road.
                    </p>
                  </div>
                )}
              </div>

              <button
                onClick={handleSimulateRoad}
                disabled={simulating || backendStatus !== "online" || !drawnRoad}
                className="w-full rounded-md py-2.5 text-[12px] font-semibold transition-all
                           disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                style={{
                  background: drawnRoad ? "#5B98D4" : "rgba(91, 152, 212, 0.25)",
                  color: "#F4F4F6",
                }}
              >
                {simulating ? (
                  <>
                    <span className="inline-block w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Simulating impact…
                  </>
                ) : (
                  <>
                    <Route size={14} />
                    Simulate growth delta
                  </>
                )}
              </button>

              {/* Simulation Result */}
              {simulationResult && !simulationResult.error && (
                <div
                  className="rounded-md p-3 space-y-2 mt-3"
                  style={{
                    background: "#141417",
                    border: "1px solid #26262B",
                  }}
                >
                  <span className="text-[11px] font-semibold text-[#5B98D4] block">
                    Scenario Impact Analysis
                  </span>
                  <div className="space-y-1.5 pt-1">
                    <div className="flex justify-between text-[11px]">
                      <span className="text-[#71717A]">Updated P(Urban)</span>
                      <span className="font-mono text-[#F4F4F6] font-semibold">
                        {((simulationResult.updated_p_urban || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-[11px]">
                      <span className="text-[#71717A]">Delta Uplift</span>
                      <span className="font-mono text-[#5B98D4] font-semibold">
                        +{((simulationResult.delta_p_urban || 0) * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-[11px]">
                      <span className="text-[#71717A]">Affected Cells</span>
                      <span className="font-mono text-[#F4F4F6]">
                        {simulationResult.affected_area_cells}
                      </span>
                    </div>
                    <div className="flex justify-between text-[11px]">
                      <span className="text-[#71717A]">Influence Radius</span>
                      <span className="font-mono text-[#F4F4F6]">
                        {simulationResult.influence_radius_cells} cells
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Precision Coordinate Readout Footer ─────────────────────── */}
        <div
          className="flex items-center justify-between px-4 py-2.5 font-mono text-[11px]"
          style={{
            borderTop: "1px solid #26262B",
            background: "#0C0C0E",
          }}
        >
          <span className="text-[#A1A1AA] tracking-tight">
            {viewState.latitude.toFixed(4)}°N &nbsp; {viewState.longitude.toFixed(4)}°E
          </span>
          <span className="text-[#A1A1AA] font-semibold">
            ZOOM {viewState.zoom.toFixed(1)}
          </span>
        </div>
      </aside>

      {/* ─── MAP LEGEND (Bottom-Left, Above Scale) ───────────────────────── */}
      <div
        className="absolute z-30 rounded-md shadow-lg"
        style={{
          bottom: "76px",
          left: panelOpen ? "296px" : "16px",
          background: "rgba(12, 12, 14, 0.94)",
          backdropFilter: "blur(8px)",
          border: "1px solid #26262B",
          padding: "8px 12px",
          transition: "left 0.25s cubic-bezier(0.4, 0, 0.2, 1)",
        }}
      >
        <p className="text-[10px] font-mono uppercase tracking-wider text-[#71717A] mb-2 flex items-center gap-1.5">
          <Layers size={11} />
          <span>Active Layers</span>
        </p>

        <div className="space-y-1.5">
          {/* Historical raster swatch */}
          {showHistoryLegend && (
            <div className="flex items-center gap-2">
              <span
                className="w-3 h-3 shrink-0 rounded-[2px]"
                style={{
                  background: historySwatchColor,
                  border: `1px solid ${historySwatchColor}`,
                  opacity: historyOpacity,
                }}
              />
              <span className="text-[11px] text-[#A1A1AA]">
                Observed Urban Extent ({historyData.year})
              </span>
            </div>
          )}

          {/* Forecast raster swatch */}
          {forecastData && (activeWorkflow === WORKFLOWS.FORECAST || !historyData) && (
            <div className="flex items-center gap-2">
              <span
                className="w-3 h-3 shrink-0 rounded-[2px]"
                style={{
                  background: "#E06356",
                  border: "1px solid #EF4444",
                  opacity: 0.8,
                }}
              />
              <span className="text-[11px] text-[#A1A1AA]">
                Predicted Expansion (2030)
              </span>
            </div>
          )}

          {/* Scenario delta swatch */}
          {simulationData && activeWorkflow === WORKFLOWS.SIMULATE && (
            <div className="flex items-center gap-2">
              <span
                className="w-3 h-3 shrink-0 rounded-[2px]"
                style={{
                  background: "#5B98D4",
                  border: "1px solid #60A5FA",
                  opacity: 0.85,
                }}
              />
              <span className="text-[11px] text-[#A1A1AA]">
                Simulated Growth Delta
              </span>
            </div>
          )}

          {!showHistoryLegend &&
            !(forecastData && (activeWorkflow === WORKFLOWS.FORECAST || !historyData)) &&
            !(simulationData && activeWorkflow === WORKFLOWS.SIMULATE) && (
              <span className="text-[11px] text-[#71717A]">
                No layers visible
              </span>
            )}
        </div>
      </div>

      {/* ─── BOTTOM FLOATING TIMELINE DOCK (Single Source of Truth) ────── */}
      <div
        className="absolute z-40 flex justify-center pointer-events-none"
        style={{
          bottom: "24px",
          left: panelOpen ? "280px" : "0",
          right: "0",
          transition: "left 0.25s cubic-bezier(0.4, 0, 0.2, 1)",
        }}
      >
        <div
          className="pointer-events-auto flex items-center gap-3 w-full"
          style={{
            maxWidth: "680px",
            margin: "0 24px",
            background: "rgba(12, 12, 14, 0.94)",
            backdropFilter: "blur(12px)",
            borderRadius: "8px",
            border: "1px solid #26262B",
            padding: "8px 16px",
            boxShadow: "0 6px 24px rgba(0, 0, 0, 0.5)",
          }}
        >
          {/* Play/Pause Button */}
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center justify-center rounded p-1.5 transition-colors hover:bg-[#1F1F23]"
            style={{ color: isPlaying ? "#F59E0B" : "#A1A1AA" }}
            title={isPlaying ? "Pause time-lapse" : "Play 1995–2025 time-lapse"}
            aria-label={isPlaying ? "Pause time-lapse" : "Play time-lapse"}
          >
            {isPlaying ? <Pause size={15} /> : <Play size={15} />}
          </button>

          {/* Current Year Badge */}
          <span
            className="font-mono text-[14px] font-semibold min-w-[44px] text-center"
            style={{ color: "#F4F4F6" }}
          >
            {currentYear}
          </span>

          {/* Year Range Slider */}
          <input
            type="range"
            min="1995"
            max="2025"
            step="5"
            value={currentYear}
            onChange={(e) => {
              setCurrentYear(Number(e.target.value));
              if (isPlaying) setIsPlaying(false);
            }}
            className="flex-1"
            aria-label="Temporal scrubber"
          />

          {/* Quick-jump Year Ticks */}
          <div className="hidden sm:flex items-center gap-1">
            {TIMELINE_YEARS.map((yr) => {
              const isSelected = currentYear === yr;
              return (
                <button
                  key={yr}
                  onClick={() => {
                    setCurrentYear(yr);
                    if (isPlaying) setIsPlaying(false);
                  }}
                  className="px-1.5 py-0.5 rounded text-[11px] font-mono transition-colors"
                  style={{
                    color: isSelected ? "#F59E0B" : "#71717A",
                    fontWeight: isSelected ? 600 : 400,
                    background: isSelected ? "rgba(217, 119, 6, 0.15)" : "transparent",
                  }}
                >
                  {yr}
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
