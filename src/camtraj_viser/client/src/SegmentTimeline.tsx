// (camtraj patch) Standalone segment-timeline overlay. Deliberately bypasses
// viser's GUI panel/dock system (see dock/FloatingWindowView.tsx,
// dock/TabGroupFrame.tsx): that system always renders window chrome (a header
// handle, a background Paper, drag/resize affordances), none of which this
// widget wants. Mounted unconditionally at the App root, like Titlebar or
// CommandPalette -- it renders nothing when the server hasn't sent a
// SegmentTimelineMessage yet.

import React, { useContext, useState } from "react";
import { ViewerContext } from "./ViewerContext";
import { toMantineColor } from "./components/colorUtils";

const BOX_HEIGHT = "2.1em";
const ADD_COLOR = "#8a8a8a";
const LABEL_FONT =
  "ui-rounded, 'SF Pro Rounded', 'Segoe UI', system-ui, sans-serif";
// Below this extent, the box shows the short form ("seg N"); at/above it,
// the long form ("segment N") plus extra padding for the "spacious" widest
// look -- box width and label text grow together, not independently.
const LONG_LABEL_THRESHOLD = 0.4;

export function SegmentTimeline() {
  const viewer = useContext(ViewerContext)!;
  const timeline = viewer.useGui((state) => state.segmentTimeline);
  const viewerMutable = viewer.mutable.current;
  const [hovered, setHovered] = useState<number | null>(null);

  if (timeline === null) {
    return null;
  }
  const { segments, selected } = timeline;

  const send = (action: "select" | "add" | "remove", index: number) =>
    viewerMutable.sendMessage({ type: "SegmentTimelineActionMessage", action, index });

  return (
    <div
      style={{
        position: "fixed",
        top: "0.75em",
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 50,
        display: "flex",
        gap: "0.3em",
        userSelect: "none",
      }}
    >
      {segments.map(([shortLabel, longLabel, color, extent, removable], index) => {
        const isSelected = index === selected;
        const base = toMantineColor(color)!;
        const useLong = extent >= LONG_LABEL_THRESHOLD;
        const extraPad = useLong ? (extent - LONG_LABEL_THRESHOLD) / (1 - LONG_LABEL_THRESHOLD) : 0;
        return (
          <div
            key={index}
            onClick={() => send("select", index)}
            onMouseEnter={() => setHovered(index)}
            onMouseLeave={() => setHovered((h) => (h === index ? null : h))}
            style={{
              position: "relative",
              flexGrow: 1.3 + extent * 2.7,
              flexBasis: 0,
              minWidth: "fit-content",
              height: BOX_HEIGHT,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: `0 ${0.6 + extraPad * 1.8}em`,
              borderRadius: "0.4em",
              cursor: "pointer",
              color: "#fff",
              fontFamily: LABEL_FONT,
              letterSpacing: "0.02em",
              fontSize: "0.85em",
              fontWeight: isSelected ? 600 : 500,
              backgroundColor: base,
              opacity: isSelected ? 1.0 : 0.62,
              boxShadow: isSelected ? "0 0 0 2px rgba(0,0,0,0.35) inset" : "none",
              transition: "opacity 0.1s, box-shadow 0.1s, padding 0.1s",
              whiteSpace: "nowrap",
            }}
          >
            {useLong ? longLabel : shortLabel}
            {removable && (
              <span
                onClick={(event) => {
                  event.stopPropagation();
                  send("remove", index);
                }}
                style={{
                  marginLeft: "0.5em",
                  opacity: hovered === index ? 0.9 : 0.35,
                  transition: "opacity 0.1s",
                  fontWeight: 700,
                  lineHeight: 1,
                }}
              >
                &times;
              </span>
            )}
          </div>
        );
      })}
      <div
        onClick={() => send("add", segments.length)}
        style={{
          height: BOX_HEIGHT,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "0 0.7em",
          borderRadius: "0.4em",
          cursor: "pointer",
          color: ADD_COLOR,
          fontFamily: LABEL_FONT,
          fontSize: "0.95em",
          fontWeight: 600,
          border: `1px dashed ${ADD_COLOR}`,
          opacity: 0.75,
        }}
      >
        +
      </div>
    </div>
  );
}
