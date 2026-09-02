// (camtraj patch) Standalone keyframe-timeline overlay for the Auteur demo.
// Sibling to SegmentTimeline.tsx (same "bypass the panel/dock system"
// rationale -- see that file's header comment) but a different interaction
// model entirely: a fixed-width bar spanning the whole trajectory, with
// draggable pins at absolute frame positions (not resizable, sequentially-
// appended boxes), plus an embedded total-frame-count field.

import React, { useContext, useRef, useState } from "react";
import { ViewerContext } from "./ViewerContext";

const BAR_WIDTH = 480;
const BAR_HEIGHT = 8;
const PIN_COLOR = "rgb(190, 110, 40)";

type DragState = { index: number; frame: number } | null;

export function KeyframeTimeline() {
  const viewer = useContext(ViewerContext)!;
  const timeline = viewer.useGui((state) => state.keyframeTimeline);
  const viewerMutable = viewer.mutable.current;
  const barRef = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<DragState>(null);
  const [totalFramesText, setTotalFramesText] = useState<string | null>(null);

  if (timeline === null) {
    return null;
  }
  const { keyframes, selected, total_frames, total_frames_min, total_frames_max } = timeline;

  const send = (action: "select" | "add" | "remove" | "move" | "set_total_frames", index: number, frame: number) =>
    viewerMutable.sendMessage({ type: "KeyframeTimelineActionMessage", action, index, frame });

  const frameToFraction = (frame: number) => (total_frames <= 1 ? 0 : frame / (total_frames - 1));
  const xToFrame = (x: number) => {
    const fraction = Math.min(1, Math.max(0, x / BAR_WIDTH));
    return Math.round(fraction * (total_frames - 1));
  };

  const displayFrame = (index: number, original: number) => (drag !== null && drag.index === index ? drag.frame : original);

  const beginDrag = (index: number, startClientX: number) => {
    const bar = barRef.current;
    if (!bar) return;
    const rect = bar.getBoundingClientRect();
    const lo = keyframes[index - 1][1] + 1;
    const hi = index === keyframes.length - 1 ? total_frames - 1 : keyframes[index + 1][1] - 1;
    // Below this many pixels of movement, treat the gesture as a plain click
    // (selection) rather than a drag -- otherwise an ordinary click to
    // select a pin also fires a spurious "move to the same frame" message.
    const DRAG_THRESHOLD_PX = 3;
    let moved = false;

    const onMove = (event: PointerEvent) => {
      if (!moved && Math.abs(event.clientX - startClientX) < DRAG_THRESHOLD_PX) return;
      moved = true;
      const raw = xToFrame(event.clientX - rect.left);
      const clamped = Math.min(hi, Math.max(lo, raw));
      setDrag({ index, frame: clamped });
    };
    const onUp = (event: PointerEvent) => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      if (moved) {
        const raw = xToFrame(event.clientX - rect.left);
        const clamped = Math.min(hi, Math.max(lo, raw));
        send("move", index, clamped);
      }
      setDrag(null);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const commitTotalFrames = () => {
    if (totalFramesText === null) return;
    const parsed = Math.round(Number(totalFramesText));
    if (Number.isFinite(parsed)) {
      const clamped = Math.min(total_frames_max, Math.max(total_frames_min, parsed));
      send("set_total_frames", 0, clamped);
    }
    setTotalFramesText(null);
  };

  return (
    <div
      style={{
        position: "fixed",
        top: "0.75em",
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 50,
        display: "flex",
        alignItems: "center",
        gap: "0.75em",
        userSelect: "none",
        fontFamily: "ui-rounded, 'SF Pro Rounded', 'Segoe UI', system-ui, sans-serif",
      }}
    >
      <div
        ref={barRef}
        onClick={(event) => {
          const rect = (event.currentTarget as HTMLDivElement).getBoundingClientRect();
          send("add", 0, xToFrame(event.clientX - rect.left));
        }}
        style={{
          position: "relative",
          width: `${BAR_WIDTH}px`,
          height: `${BAR_HEIGHT}px`,
          borderRadius: `${BAR_HEIGHT / 2}px`,
          backgroundColor: "rgba(190, 110, 40, 0.25)",
          cursor: "copy",
          marginTop: "1.6em", // room for pins/labels above the bar
        }}
      >
        {keyframes.map(([label, frame], index) => {
          const isSelected = index === selected;
          const isFirst = index === 0;
          const shownFrame = displayFrame(index, frame);
          return (
            <div
              key={index}
              onClick={(event) => {
                event.stopPropagation();
                send("select", index, 0);
              }}
              onPointerDown={(event) => {
                if (isFirst) return;
                event.stopPropagation();
                beginDrag(index, event.clientX);
              }}
              style={{
                position: "absolute",
                left: `${frameToFraction(shownFrame) * 100}%`,
                top: "50%",
                transform: "translate(-50%, -100%)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                cursor: isFirst ? "pointer" : "ew-resize",
              }}
            >
              <div
                style={{
                  position: "relative",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.3em",
                  padding: "0.15em 0.5em",
                  borderRadius: "0.3em",
                  color: "#fff",
                  fontSize: "0.8em",
                  fontWeight: isSelected ? 600 : 500,
                  backgroundColor: PIN_COLOR,
                  opacity: isSelected ? 1.0 : 0.6,
                  boxShadow: isSelected ? "0 0 0 2px rgba(0,0,0,0.35) inset" : "none",
                  whiteSpace: "nowrap",
                }}
              >
                {label}
                {!isFirst && (
                  <span
                    // Stop the pointerdown here too, not just the click --
                    // otherwise the pin's own onPointerDown (above) sees it
                    // first (pointerdown fires before click) and starts a
                    // drag before the click on "x" ever gets a chance to
                    // register.
                    onPointerDown={(event) => event.stopPropagation()}
                    onClick={(event) => {
                      event.stopPropagation();
                      send("remove", index, 0);
                    }}
                    style={{ opacity: 0.7, fontWeight: 700, lineHeight: 1 }}
                  >
                    &times;
                  </span>
                )}
              </div>
              {/* the "pointy thing" connecting the label down to the bar */}
              <div
                style={{
                  width: 0,
                  height: 0,
                  borderLeft: "4px solid transparent",
                  borderRight: "4px solid transparent",
                  borderTop: `5px solid ${PIN_COLOR}`,
                  opacity: isSelected ? 1.0 : 0.6,
                }}
              />
            </div>
          );
        })}
      </div>
      <input
        type="number"
        value={totalFramesText ?? total_frames}
        min={total_frames_min}
        max={total_frames_max}
        onChange={(event) => setTotalFramesText(event.target.value)}
        onBlur={commitTotalFrames}
        onKeyDown={(event) => {
          if (event.key === "Enter") (event.target as HTMLInputElement).blur();
        }}
        style={{
          width: "4em",
          height: `${BAR_HEIGHT + 22}px`,
          textAlign: "center",
          borderRadius: "0.3em",
          border: `1px solid ${PIN_COLOR}`,
          fontFamily: "inherit",
          fontSize: "0.85em",
        }}
      />
    </div>
  );
}
