import { useEffect, useRef } from "react";
import { useControl } from "react-map-gl";
import MapboxDraw from "@mapbox/mapbox-gl-draw";
import "@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css";

/**
 * DrawControl – react-map-gl v8 bridge for @mapbox/mapbox-gl-draw
 * =================================================================
 * Wraps vanilla MapboxDraw using react-map-gl's useControl hook.
 * Binds draw.create, draw.update, and draw.delete map events to React callback props.
 *
 * @param {Object} props
 * @param {Function} [props.onCreate] - Callback fired on "draw.create" event
 * @param {Function} [props.onUpdate] - Callback fired on "draw.update" event
 * @param {Function} [props.onDelete] - Callback fired on "draw.delete" event
 * @param {string} [props.position="top-right"] - Control position on map canvas
 * @param {Object} [props.controls] - Enabled controls ({ line_string: true, trash: true, ... })
 * @param {boolean} [props.displayControlsDefault=false]
 * @param {string} [props.defaultMode="simple_select"]
 */
export default function DrawControl({
  position = "top-right",
  onCreate,
  onUpdate,
  onDelete,
  controls = {
    line_string: true,
    trash: true,
  },
  displayControlsDefault = false,
  defaultMode = "simple_select",
  ...restOptions
}) {
  // Store latest callbacks in ref to avoid stale closures during event handling
  const callbacksRef = useRef({ onCreate, onUpdate, onDelete });
  useEffect(() => {
    callbacksRef.current = { onCreate, onUpdate, onDelete };
  }, [onCreate, onUpdate, onDelete]);

  // Stable event listener references so map.off can remove the exact listeners
  const listenersRef = useRef({
    create: (e) => callbacksRef.current.onCreate?.(e),
    update: (e) => callbacksRef.current.onUpdate?.(e),
    delete: (e) => callbacksRef.current.onDelete?.(e),
  });

  useControl(
    () =>
      new MapboxDraw({
        displayControlsDefault,
        controls,
        defaultMode,
        ...restOptions,
      }),
    ({ map }) => {
      map.on("draw.create", listenersRef.current.create);
      map.on("draw.update", listenersRef.current.update);
      map.on("draw.delete", listenersRef.current.delete);
    },
    ({ map }) => {
      map.off("draw.create", listenersRef.current.create);
      map.off("draw.update", listenersRef.current.update);
      map.off("draw.delete", listenersRef.current.delete);
    },
    { position }
  );

  return null;
}
