"use client";

import React, { useState } from "react";
import { cn } from "@/lib/utils";
import { debugImageUrl } from "@/lib/api";
import type { DebugData } from "@/lib/api";

interface FragmentTimelineProps {
  debug: DebugData;
  selectedId?: number | null;
  onSelect?: (id: number) => void;
}

export function FragmentTimeline({ debug, selectedId, onSelect }: FragmentTimelineProps) {
  const fragments = debug.fragments ?? [];
  const translations = debug.translations ?? {};
  const steps = debug.steps ?? [];
  const paths = debug.paths;

  // Build connection map from merge steps
  const connections = new Map<number, Set<number>>();
  for (const step of steps) {
    if (!connections.has(step.anchor))  connections.set(step.anchor, new Set());
    if (!connections.has(step.attached)) connections.set(step.attached, new Set());
    connections.get(step.anchor)!.add(step.attached);
    connections.get(step.attached)!.add(step.anchor);
  }

  return (
    <div className="w-full overflow-x-auto no-scrollbar py-4">
      <div className="flex items-start gap-3 min-w-max px-2">
        {fragments.map((frag, idx) => {
          const isSelected = selectedId === frag.id;
          const tx = translations[String(frag.id)];
          const connectedTo = connections.get(frag.id);
          const cropUrl = paths?.fragment_crops?.[idx]
            ? debugImageUrl(paths.fragment_crops[idx])
            : null;
          const isAnchor = steps.length > 0 && steps[0].anchor === frag.id;

          return (
            <div key={frag.id} className="flex items-center gap-3">
              {/* Fragment card */}
              <button
                onClick={() => onSelect?.(frag.id)}
                className={cn(
                  "flex flex-col items-center gap-2 p-3 rounded-2xl border-2 transition-all duration-150 w-28",
                  "hover:shadow-card-hover focus:outline-none focus:ring-2 focus:ring-accent/40",
                  isSelected
                    ? "border-accent bg-accent-light shadow-soft"
                    : "border-border bg-surface hover:border-accent/40"
                )}
              >
                {/* Image preview */}
                <div className="w-20 h-20 rounded-xl overflow-hidden bg-muted flex-shrink-0 relative">
                  {cropUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={cropUrl}
                      alt={`Fragment ${frag.id}`}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-secondary/30 text-xs">
                      #{frag.id}
                    </div>
                  )}
                  {/* Anchor badge */}
                  {isAnchor && (
                    <div className="absolute top-1 right-1 w-4 h-4 bg-accent rounded-full flex items-center justify-center">
                      <span className="text-white text-[8px] font-bold">A</span>
                    </div>
                  )}
                </div>

                {/* Fragment ID */}
                <span className={cn(
                  "text-xs font-semibold",
                  isSelected ? "text-accent" : "text-primary"
                )}>
                  #{frag.id}
                </span>

                {/* Stats */}
                <div className="text-[10px] text-secondary text-center leading-tight">
                  <div>{(frag.area / 1000).toFixed(1)}k px</div>
                  {tx && (
                    <div className="font-mono text-[9px] text-accent/70">
                      {tx.dx > 0 ? "+" : ""}{tx.dx.toFixed(0)}, {tx.dy > 0 ? "+" : ""}{tx.dy.toFixed(0)}
                    </div>
                  )}
                </div>

                {/* Connection dots */}
                {connectedTo && connectedTo.size > 0 && (
                  <div className="flex gap-0.5 flex-wrap justify-center max-w-[72px]">
                    {[...connectedTo].slice(0, 4).map((cid) => (
                      <span
                        key={cid}
                        className="text-[9px] bg-accent/10 text-accent rounded-full px-1"
                      >
                        #{cid}
                      </span>
                    ))}
                    {connectedTo.size > 4 && (
                      <span className="text-[9px] text-secondary">+{connectedTo.size - 4}</span>
                    )}
                  </div>
                )}
              </button>

              {/* Arrow connector between cards */}
              {idx < fragments.length - 1 && (
                <div className="flex items-center text-border">
                  <div className="w-6 h-px bg-border" />
                  <div className="w-0 h-0 border-y-2 border-y-transparent border-l-4 border-l-border" />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
