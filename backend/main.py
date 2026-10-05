"""
Urban Sprawl Detection API -- Backend
=====================================
CDX Point 2  : Model loading via FastAPI lifespan (XGBoost + CNN-LSTM stubs).
CDX Point 3  : Markov Demand Estimation -- 2x2 Transition Matrix for urban
               expansion baseline demand.
CDX Point 8  : CDX Fusion inference endpoint with strict logit transform and
               sigmoid activation over the fused latent score.
What-If      : Interactive Road Sandbox -- simulate Dist_Roads recalculation
               via SciPy distance_transform_edt and return updated P_urban.

Fusion equation
---------------
  z_XGB    = logit(S_XGB)  = log( S_XGB / (1 - S_XGB) )
  z_DL     = logit(S_DL)   = log( S_DL  / (1 - S_DL)  )   [CNN-LSTM branch]
  z_CA     = logit(S_CA)   = log( S_CA  / (1 - S_CA)  )   [contextual-attention branch]

  L_fused  = w1*z_DL + w2*z_XGB + w3*z_CA
  P_urban  = sigmoid(L_fused) = 1 / (1 + exp(-L_fused))

Markov demand
-------------
  P(t+1) = T @ P(t)
  where T is a 2x2 row-stochastic transition matrix:
      T = [[P(non-urban -> non-urban),  P(non-urban -> urban)],
           [P(urban    -> non-urban),   P(urban    -> urban) ]]"""

from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import xgboost as xgb
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------------
# TensorFlow: configure memory growth BEFORE any other TF import
# ---------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress TF info/warnings

import tensorflow as tf  # noqa: E402 (must follow env-var setup)

_tf_gpus = tf.config.list_physical_devices("GPU")
for _gpu in _tf_gpus:
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
        print(f"[TF] Memory growth enabled for GPU: {_gpu.name}")
    except RuntimeError as _e:
        print(f"[TF] Could not set memory growth for {_gpu.name}: {_e}")

# Also limit CPU memory allocation (TF can be greedy even without GPU)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# CDX POINT 2 -- Asset paths (relative to project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent   # d:\Urban_Sprawl
_DATA_DIR     = _PROJECT_ROOT / "data"

_XGB_MODEL_PATH     = _DATA_DIR / "Urban_Sprawl_XGB.json"
_CNNLSTM_MODEL_PATH = _DATA_DIR / "Urban_Sprawl_CNNLSTM.keras"
_RASTER_PATH        = _DATA_DIR / "Karachi_Urban_2020.tif"


# ---------------------------------------------------------------------------
# CDX POINT 2 -- Model registry (populated at startup via lifespan)
# ---------------------------------------------------------------------------
class ModelRegistry:
    """Holds lazy-loaded model handles and raster metadata."""
    xgboost_model: xgb.Booster | None = None
    cnn_lstm_model: tf.keras.Model | None = None
    # Raster metadata (populated from Karachi_Urban_2020.tif)
    raster_shape: tuple[int, ...] | None = None
    raster_transform: Any = None
    raster_crs: Any = None
    raster_bounds: Any = None


registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    CDX Point 2: Application lifespan -- load real models on startup and
    release resources on shutdown.  Each load is wrapped in a try/except so
    that a missing file logs a warning instead of crashing the server.
    """
    # -- STARTUP --------------------------------------------------------------

    # 1. XGBoost Booster
    try:
        logger.info("[CDX-2] Loading XGBoost model from %s ...", _XGB_MODEL_PATH)
        booster = xgb.Booster()
        booster.load_model(str(_XGB_MODEL_PATH))
        registry.xgboost_model = booster
        logger.info(
            "[CDX-2] XGBoost loaded  --  %d features, %d trees.",
            int(booster.num_features()),
            int(booster.num_boosted_rounds()),
        )
    except FileNotFoundError:
        logger.warning(
            "[CDX-2] XGBoost model NOT FOUND at %s -- endpoint will return 503.",
            _XGB_MODEL_PATH,
        )
    except Exception as exc:
        logger.error("[CDX-2] XGBoost load failed: %s", exc, exc_info=True)

    # 2. CNN-LSTM (TensorFlow / Keras)
    try:
        logger.info("[CDX-2] Loading CNN-LSTM model from %s ...", _CNNLSTM_MODEL_PATH)
        registry.cnn_lstm_model = tf.keras.models.load_model(
            str(_CNNLSTM_MODEL_PATH),
            compile=False,   # skip optimizer restoration (inference-only)
        )
        logger.info(
            "[CDX-2] CNN-LSTM loaded  --  input shape %s, output shape %s.",
            registry.cnn_lstm_model.input_shape,
            registry.cnn_lstm_model.output_shape,
        )
    except FileNotFoundError:
        logger.warning(
            "[CDX-2] CNN-LSTM model NOT FOUND at %s -- endpoint will return 503.",
            _CNNLSTM_MODEL_PATH,
        )
    except Exception as exc:
        logger.error("[CDX-2] CNN-LSTM load failed: %s", exc, exc_info=True)

    # 3. Raster reference map (Karachi_Urban_2020.tif)
    try:
        logger.info("[CDX-2] Reading raster metadata from %s ...", _RASTER_PATH)
        with rasterio.open(str(_RASTER_PATH)) as src:
            registry.raster_shape     = (src.count, src.height, src.width)
            registry.raster_transform = src.transform
            registry.raster_crs       = src.crs
            registry.raster_bounds    = src.bounds
        logger.info(
            "[CDX-2] Raster metadata  --  shape %s, CRS %s, bounds %s.",
            registry.raster_shape,
            registry.raster_crs,
            registry.raster_bounds,
        )
    except FileNotFoundError:
        logger.warning(
            "[CDX-2] Raster NOT FOUND at %s -- spatial metadata unavailable.",
            _RASTER_PATH,
        )
    except Exception as exc:
        logger.error("[CDX-2] Raster read failed: %s", exc, exc_info=True)

    # Summary
    _status = {
        "xgboost":  "loaded" if registry.xgboost_model  else "MISSING",
        "cnn_lstm": "loaded" if registry.cnn_lstm_model else "MISSING",
        "raster":   "loaded" if registry.raster_shape   else "MISSING",
    }
    logger.info("[CDX-2] Startup complete. Model status: %s", _status)

    yield   # -- application runs here ----------------------------------------

    # -- SHUTDOWN -------------------------------------------------------------
    logger.info("[CDX-2] Releasing model resources ...")
    registry.xgboost_model  = None
    registry.cnn_lstm_model = None
    registry.raster_shape   = None
    registry.raster_transform = None
    registry.raster_crs     = None
    registry.raster_bounds  = None
    logger.info("[CDX-2] Shutdown complete.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Urban Sprawl Detection API",
    description=(
        "Backend API for Urban Sprawl monitoring, temporal satellite analysis, "
        "and CDX-fused change detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend and cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# CDX POINT 8 -- Mathematical utilities
# ---------------------------------------------------------------------------
_PROB_MIN: float = 1e-4        # clip lower bound  (prevents log(0))
_PROB_MAX: float = 1.0 - 1e-4  # clip upper bound  (prevents log(inf))


def logit(probability: float) -> float:
    """
    Strict logit transformation:
        z = log( p / (1 - p) )

    Probabilities are clipped to [0.0001, 0.9999] before the transform
    so that edge-case inputs (0.0 or 1.0) never produce +/-inf or NaN.

    Args:
        probability: Raw probability score in [0, 1].

    Returns:
        Logit-transformed score in (-inf, +inf).
    """
    p = max(_PROB_MIN, min(_PROB_MAX, float(probability)))
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid:
        s(x) = 1 / (1 + exp(-x))
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    # Equivalent form for negative x to avoid exp overflow
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def cdx_fusion(
    s_dl: float,
    s_xgb: float,
    s_ca: float,
    w1: float = 0.40,
    w2: float = 0.35,
    w3: float = 0.25,
) -> tuple[float, float, float, float, float]:
    """
    CDX Point 8 -- Fused latent score computation.

    Implements:
        z_DL    = logit(S_DL)
        z_XGB   = logit(S_XGB)
        z_CA    = logit(S_CA)
        L_fused = w1*z_DL + w2*z_XGB + w3*z_CA
        P_urban = sigmoid(L_fused)

    Args:
        s_dl  : Probability from the CNN-LSTM deep-learning branch.
        s_xgb : Probability from the XGBoost branch.
        s_ca  : Probability from the contextual-attention branch.
        w1    : Weight for the DL  logit (default 0.40).
        w2    : Weight for the XGB logit (default 0.35).
        w3    : Weight for the CA  logit (default 0.25).

    Returns:
        Tuple of (z_dl, z_xgb, z_ca, l_fused, p_urban).
    """
    z_dl  = logit(s_dl)
    z_xgb = logit(s_xgb)
    z_ca  = logit(s_ca)

    l_fused = w1 * z_dl + w2 * z_xgb + w3 * z_ca
    p_urban = sigmoid(l_fused)

    return z_dl, z_xgb, z_ca, l_fused, p_urban


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    """
    Payload for the /api/v1/forecast endpoint.

    All three branch probabilities must lie in [0, 1].
    Optional fusion weights must be positive; the endpoint normalises them
    to sum to 1 automatically if they don't.
    """
    s_dl: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="CNN-LSTM branch probability score S_DL in [0, 1].",
        examples=[0.72],
    )
    s_xgb: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="XGBoost branch probability score S_XGB in [0, 1].",
        examples=[0.68],
    )
    s_ca: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Contextual-attention branch probability score S_CA in [0, 1].",
        examples=[0.75],
    )
    w1: float = Field(
        default=0.40,
        gt=0.0,
        description="Fusion weight for z_DL. Must be strictly positive.",
    )
    w2: float = Field(
        default=0.35,
        gt=0.0,
        description="Fusion weight for z_XGB. Must be strictly positive.",
    )
    w3: float = Field(
        default=0.25,
        gt=0.0,
        description="Fusion weight for z_CA. Must be strictly positive.",
    )

    @field_validator("w1", "w2", "w3")
    @classmethod
    def weights_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Fusion weights must be strictly positive.")
        return v


class ForecastResponse(BaseModel):
    """Structured JSON response from the CDX fusion endpoint."""
    # Intermediate logit scores (for transparency / debugging)
    z_dl: float = Field(..., description="Logit of S_DL.")
    z_xgb: float = Field(..., description="Logit of S_XGB.")
    z_ca: float = Field(..., description="Logit of S_CA.")
    weights: dict[str, float] = Field(..., description="Effective (normalised) fusion weights.")
    l_fused: float = Field(..., description="Fused latent score L_fused.")
    # Final prediction
    p_urban: float = Field(..., description="Final P_urban probability after sigmoid activation.")
    urban_class: str = Field(..., description="Thresholded class label (Urban / Non-Urban).")
    # Metadata
    model_status: dict[str, str] = Field(..., description="Loaded model identifiers.")


# ---------------------------------------------------------------------------
# Utility routes
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "service": "Urban Sprawl Detection API",
        "status": "online",
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models": {
            "xgboost": (
                f"loaded ({registry.xgboost_model.num_features()} features, "
                f"{registry.xgboost_model.num_boosted_rounds()} trees)"
                if registry.xgboost_model else "not loaded"
            ),
            "cnn_lstm": (
                f"loaded (input={registry.cnn_lstm_model.input_shape}, "
                f"output={registry.cnn_lstm_model.output_shape})"
                if registry.cnn_lstm_model else "not loaded"
            ),
        },
        "raster": {
            "shape": list(registry.raster_shape) if registry.raster_shape else None,
            "crs": str(registry.raster_crs) if registry.raster_crs else None,
            "bounds": list(registry.raster_bounds) if registry.raster_bounds else None,
        },
    }


# ---------------------------------------------------------------------------
# CDX POINT 8 -- Forecast endpoint
# ---------------------------------------------------------------------------
@app.post("/api/v1/forecast", response_model=ForecastResponse)
def forecast(payload: ForecastRequest) -> ForecastResponse:
    """
    **CDX Fusion Inference Endpoint** (Points 2 and 8)

    Accepts per-branch probability scores from three model heads:
    - `s_dl`  -- CNN-LSTM deep-learning branch
    - `s_xgb` -- XGBoost gradient-boosted branch
    - `s_ca`  -- Contextual-attention branch

    Applies the strict logit transform to each, computes the weighted
    fused latent score **L_fused**, passes it through sigmoid to produce
    **P_urban**, and returns the full intermediate computation for
    transparency and downstream logging.

    **Fusion equation:**
    ```
    z_i     = log( S_i / (1 - S_i) )           # logit; S_i clipped to [0.0001, 0.9999]
    L_fused = w1*z_DL + w2*z_XGB + w3*z_CA
    P_urban = sigmoid(L_fused)
    ```
    """
    # Guard: models must be loaded (CDX Point 2 dependency check)
    if not registry.xgboost_model or not registry.cnn_lstm_model:
        raise HTTPException(
            status_code=503,
            detail="Models are not yet loaded. Please retry in a moment.",
        )

    # Normalise weights so they always sum to exactly 1.0
    w_sum = payload.w1 + payload.w2 + payload.w3
    w1_n = payload.w1 / w_sum
    w2_n = payload.w2 / w_sum
    w3_n = payload.w3 / w_sum

    # Run CDX fusion (CDX Point 8 core computation)
    z_dl, z_xgb, z_ca, l_fused, p_urban = cdx_fusion(
        s_dl=payload.s_dl,
        s_xgb=payload.s_xgb,
        s_ca=payload.s_ca,
        w1=w1_n,
        w2=w2_n,
        w3=w3_n,
    )

    urban_class = "Urban" if p_urban >= 0.5 else "Non-Urban"

    return ForecastResponse(
        z_dl=round(z_dl, 6),
        z_xgb=round(z_xgb, 6),
        z_ca=round(z_ca, 6),
        weights={"w1": round(w1_n, 4), "w2": round(w2_n, 4), "w3": round(w3_n, 4)},
        l_fused=round(l_fused, 6),
        p_urban=round(p_urban, 6),
        urban_class=urban_class,
        model_status={
            "xgboost": f"Booster({registry.xgboost_model.num_features()}F, {registry.xgboost_model.num_boosted_rounds()}T)" if registry.xgboost_model else "not loaded",
            "cnn_lstm": type(registry.cnn_lstm_model).__name__ if registry.cnn_lstm_model else "not loaded",
        },
    )


# ---------------------------------------------------------------------------
# CDX POINT 3 -- Markov Demand Estimation schemas & helpers
# ---------------------------------------------------------------------------

# Baseline empirical transition probabilities (derived from historic LULC data).
# Rows = current state [Non-Urban, Urban]
# Cols = next state    [Non-Urban, Urban]
_MARKOV_T: np.ndarray = np.array(
    [
        [0.82, 0.18],   # Non-Urban row: 82% stays non-urban, 18% converts to urban
        [0.05, 0.95],   # Urban row    :  5% reverts,          95% stays urban
    ],
    dtype=np.float64,
)

# Default initial state distribution [P(non-urban), P(urban)]
_MARKOV_P0: np.ndarray = np.array([0.70, 0.30], dtype=np.float64)


def markov_project(
    transition: np.ndarray,
    p0: np.ndarray,
    steps: int,
) -> list[dict[str, float]]:
    """
    CDX Point 3: Iteratively apply the Markov transition matrix for `steps`
    time-steps and return the state distribution at each step.

    P(t+1) = T @ P(t)   (matrix-vector product; T is row-stochastic)

    Args:
        transition : 2x2 row-stochastic transition matrix.
        p0         : Initial state distribution [P_non_urban, P_urban].
        steps      : Number of projection steps.

    Returns:
        List of dicts, one per step, with keys:
            step, p_non_urban, p_urban, demand_index.
    """
    trajectory: list[dict[str, float]] = []
    p = p0.copy()
    for t in range(1, steps + 1):
        p = transition.T @ p          # column-vector update
        p = np.clip(p, 0.0, 1.0)
        p /= p.sum()                  # renormalise after clipping
        # Demand index: probability-weighted expected urban expansion
        demand_index = float(p[1] - p0[1])   # net gain in urban probability
        trajectory.append(
            {
                "step": t,
                "p_non_urban": round(float(p[0]), 6),
                "p_urban": round(float(p[1]), 6),
                "demand_index": round(demand_index, 6),
            }
        )
    return trajectory


class DemandResponse(BaseModel):
    """Response from the /api/v1/demand endpoint."""
    transition_matrix: list[list[float]] = Field(
        ...,
        description="2x2 Markov Transition Matrix (row = current, col = next).",
    )
    initial_state: dict[str, float] = Field(
        ...,
        description="Initial state distribution: P_non_urban + P_urban = 1.",
    )
    projection_steps: int = Field(..., description="Number of future steps projected.")
    trajectory: list[dict[str, float]] = Field(
        ...,
        description="Per-step state distribution and demand index.",
    )
    steady_state: dict[str, float] = Field(
        ...,
        description="Analytically derived steady-state distribution.",
    )


# ---------------------------------------------------------------------------
# CDX POINT 3 -- GET /api/v1/demand
# ---------------------------------------------------------------------------
@app.get("/api/v1/demand", response_model=DemandResponse)
def demand(
    steps: int = 10,
    p0_urban: float = 0.30,
) -> DemandResponse:
    """
    **CDX Point 3: Markov Demand Estimation**

    Computes a 2x2 Markov Transition Matrix for urban expansion and projects
    the land-use state distribution forward for `steps` time-steps.

    **Transition Matrix (row = current state, col = next state):**
    ```
    T = [[P(NU->NU), P(NU->U)],
         [P( U->NU), P( U->U)]]
      = [[0.82, 0.18],
         [0.05, 0.95]]
    ```
    **Update rule:** `P(t+1) = T^T @ P(t)`

    Query params:
    - `steps`    : Number of projection steps (default 10).
    - `p0_urban` : Initial urban fraction in [0, 1] (default 0.30).
    """
    if not (0.0 <= p0_urban <= 1.0):
        raise HTTPException(status_code=422, detail="p0_urban must be in [0, 1].")
    if steps < 1 or steps > 100:
        raise HTTPException(status_code=422, detail="steps must be between 1 and 100.")

    p0 = np.array([1.0 - p0_urban, p0_urban], dtype=np.float64)
    trajectory = markov_project(_MARKOV_T, p0, steps)

    # Analytical steady-state: solve (T^T - I) pi = 0 subject to sum = 1
    # For 2-state chain: pi_U = (1 - T[0,0]) / (2 - T[0,0] - T[1,1])
    t_nu_u = float(_MARKOV_T[0, 1])   # P(non-urban -> urban)
    t_u_nu = float(_MARKOV_T[1, 0])   # P(urban    -> non-urban)
    pi_urban = t_nu_u / (t_nu_u + t_u_nu)
    pi_non_urban = 1.0 - pi_urban

    return DemandResponse(
        transition_matrix=_MARKOV_T.tolist(),
        initial_state={
            "p_non_urban": round(float(p0[0]), 6),
            "p_urban": round(float(p0[1]), 6),
        },
        projection_steps=steps,
        trajectory=trajectory,
        steady_state={
            "p_non_urban": round(pi_non_urban, 6),
            "p_urban": round(pi_urban, 6),
        },
    )


# ---------------------------------------------------------------------------
# What-If Sandbox -- Schemas
# ---------------------------------------------------------------------------

# Grid constants for the road simulation sandbox
_GRID_H: int = 64    # rows
_GRID_W: int = 64    # cols
_ROAD_INFLUENCE_SIGMA: float = 5.0   # cells; Gaussian softening radius


class GeoJSONCoordinate(BaseModel):
    """A single [longitude, latitude] or [col, row] coordinate pair."""
    x: float = Field(..., description="Longitude or grid column coordinate.")
    y: float = Field(..., description="Latitude or grid row coordinate.")


class SimulateRoadRequest(BaseModel):
    """
    Payload for the /api/v1/simulate_road (What-If Sandbox) endpoint.

    Accepts a GeoJSON LineString-style coordinate list representing a
    user-drawn road, plus the current baseline P_urban to update.
    Coordinates are interpreted as fractional [0, 1] positions within the
    simulation grid so no CRS transform is needed at this stage.
    """
    coordinates: list[GeoJSONCoordinate] = Field(
        ...,
        min_length=2,
        description=(
            "Ordered list of road vertices as (x, y) pairs. "
            "x and y should be fractional positions in [0, 1] "
            "(0 = left/top edge, 1 = right/bottom edge of the tile)."
        ),
    )
    baseline_p_urban: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Current baseline urban probability before road insertion.",
    )
    grid_h: int = Field(default=_GRID_H, ge=8, le=512, description="Simulation grid height (rows).")
    grid_w: int = Field(default=_GRID_W, ge=8, le=512, description="Simulation grid width (cols).")


class SimulateRoadResponse(BaseModel):
    """Response from the /api/v1/simulate_road endpoint."""
    updated_p_urban: float = Field(..., description="New P_urban after road infrastructure effect.")
    delta_p_urban: float = Field(..., description="Change in P_urban (updated - baseline).")
    affected_area_cells: int = Field(
        ...,
        description="Number of grid cells within the road influence radius.",
    )
    influence_radius_cells: float = Field(
        ...,
        description="Effective influence radius used (in grid cells).",
    )
    road_pixel_count: int = Field(..., description="Number of rasterised road pixels on the grid.")
    grid_shape: list[int] = Field(..., description="[height, width] of the simulation grid.")
    dist_roads_stats: dict[str, float] = Field(
        ...,
        description="Statistics of the Dist_Roads distance field (min/max/mean in cells).",
    )


# ---------------------------------------------------------------------------
# What-If Sandbox -- Helpers
# ---------------------------------------------------------------------------

def _rasterise_road(
    coordinates: list[GeoJSONCoordinate],
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    """
    Rasterise a polyline (list of fractional-coordinate vertices) onto a
    boolean grid using Bresenham-style linear interpolation between vertices.

    Args:
        coordinates : Road vertices with x, y in [0, 1].
        grid_h      : Grid height in cells.
        grid_w      : Grid width in cells.

    Returns:
        Boolean numpy array of shape (grid_h, grid_w); True = road pixel.
    """
    road_mask = np.zeros((grid_h, grid_w), dtype=bool)

    def _clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    for i in range(len(coordinates) - 1):
        x0 = _clamp(round(coordinates[i].x * (grid_w - 1)), 0, grid_w - 1)
        y0 = _clamp(round(coordinates[i].y * (grid_h - 1)), 0, grid_h - 1)
        x1 = _clamp(round(coordinates[i + 1].x * (grid_w - 1)), 0, grid_w - 1)
        y1 = _clamp(round(coordinates[i + 1].y * (grid_h - 1)), 0, grid_h - 1)

        # Bresenham's line algorithm
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx - dy
        cx, cy = x0, y0
        while True:
            road_mask[cy, cx] = True
            if cx == x1 and cy == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

    return road_mask


def _road_p_urban_delta(
    dist_field: np.ndarray,
    influence_sigma: float,
    baseline_p_urban: float,
) -> tuple[float, int]:
    """
    Estimate the change in average P_urban caused by the new road.

    Model: cells within `influence_sigma` cells of the road receive a
    Gaussian accessibility bonus that additively increases P_urban,
    capped at 0.99.

    delta(cell) = bonus_strength * exp( -dist^2 / (2*sigma^2) )
    where bonus_strength = 0.15 (empirical road-development uplift).

    Returns:
        (mean_updated_p_urban, affected_area_cells)
    """
    _BONUS_STRENGTH = 0.15
    _INFLUENCE_CUTOFF = 3.0 * influence_sigma   # cells beyond which effect is negligible

    influence_mask = dist_field <= _INFLUENCE_CUTOFF
    affected_cells = int(influence_mask.sum())

    bonus = _BONUS_STRENGTH * np.exp(
        -(dist_field[influence_mask] ** 2) / (2.0 * influence_sigma ** 2)
    )
    p_cells = np.clip(baseline_p_urban + bonus, 0.0, 0.99)
    mean_updated = float(np.mean(p_cells)) if affected_cells > 0 else baseline_p_urban
    return mean_updated, affected_cells


# ---------------------------------------------------------------------------
# What-If Sandbox -- POST /api/v1/simulate_road
# ---------------------------------------------------------------------------
@app.post("/api/v1/simulate_road", response_model=SimulateRoadResponse)
def simulate_road(payload: SimulateRoadRequest) -> SimulateRoadResponse:
    """
    **What-If Interactive Sandbox: Road Infrastructure Simulation**

    Accepts a user-drawn road as a GeoJSON LineString (fractional coordinates)
    and simulates the effect on urban expansion probability by:

    1. Rasterising the road onto a `grid_h x grid_w` boolean mask.
    2. Computing `Dist_Roads` -- the Euclidean distance from every cell to the
       nearest road pixel -- using `scipy.ndimage.distance_transform_edt`.
    3. Applying a Gaussian proximity bonus to estimate `updated_p_urban`.

    Returns the updated P_urban, the number of affected cells, and Dist_Roads
    field statistics for downstream rendering.
    """
    H, W = payload.grid_h, payload.grid_w

    # Step 1: Rasterise the user-drawn road onto the grid
    road_mask = _rasterise_road(payload.coordinates, H, W)
    road_pixel_count = int(road_mask.sum())

    if road_pixel_count == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "No road pixels were rasterised onto the grid. "
                "Check that coordinates are fractional values in [0, 1]."
            ),
        )

    # Step 2: Compute Dist_Roads via SciPy distance_transform_edt
    # distance_transform_edt treats True as background; we invert the mask
    # so road pixels (True) have distance 0, all others get their EDT value.
    non_road = ~road_mask
    dist_field: np.ndarray = distance_transform_edt(non_road)  # shape (H, W), float64

    # Step 3: Estimate P_urban uplift from road proximity
    updated_p_urban, affected_cells = _road_p_urban_delta(
        dist_field,
        influence_sigma=_ROAD_INFLUENCE_SIGMA,
        baseline_p_urban=payload.baseline_p_urban,
    )

    delta = round(updated_p_urban - payload.baseline_p_urban, 6)

    return SimulateRoadResponse(
        updated_p_urban=round(updated_p_urban, 6),
        delta_p_urban=delta,
        affected_area_cells=affected_cells,
        influence_radius_cells=round(3.0 * _ROAD_INFLUENCE_SIGMA, 2),
        road_pixel_count=road_pixel_count,
        grid_shape=[H, W],
        dist_roads_stats={
            "min_cells": round(float(dist_field.min()), 4),
            "max_cells": round(float(dist_field.max()), 4),
            "mean_cells": round(float(dist_field.mean()), 4),
        },
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
