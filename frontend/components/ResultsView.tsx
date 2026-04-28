"use client";

import React, { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { fetchDebug, debugImageUrl } from "@/lib/api";
import type { DebugData } from "@/lib/api";
import { OverviewView }       from "@/components/views/OverviewView";
import { SegmentationView }   from "@/components/views/SegmentationView";
import { ContoursView }       from "@/components/views/ContoursView";
import { ReconstructionView } from "@/components/views/ReconstructionView";
import { CompositionView }    from "@/components/views/CompositionView";
import { AssemblyView }       from "@/components/views/AssemblyView";
import { Badge } from "@/components/ui/badge";

type Tab = "overview" | "segmentation" | "contours" | "reconstruction" | "composition" | "assembly";

const TABS: { key: Tab; label: string }[] = [
  { key: "overview",       label: "Обзор" },
  { key: "segmentation",   label: "Сегментация" },
  { key: "contours",       label: "Контуры" },
  { key: "reconstruction", label: "Реконструкция" },
  { key: "composition",    label: "Компоновка" },
  { key: "assembly",       label: "Сборка" },
];

interface ResultsViewProps {
  jobId: string;
  onReset: () => void;
}

/** Eagerly preload all debug images in the background so tabs render instantly. */
function preloadDebugImages(debug: DebugData) {
  const paths = debug.paths;
  if (!paths) return;

  const urls: string[] = [];

  // Single images
  const singles = [
    paths.input, paths.sam_overlay, paths.segmentation_overlay,
    paths.contours_overlay, paths.neighbor_graph,
    paths.composition_raw, paths.composition_gap, paths.composition_inpainted,
    paths.inpainting_before, paths.inpainting_mask, paths.inpainting_cleaned,
  ];
  for (const p of singles) {
    if (p) urls.push(debugImageUrl(p));
  }

  // Array images
  const arrays = [
    paths.fragment_crops, paths.fragment_masks,
    paths.fragment_sdfs, paths.fragment_support,
    paths.reconstruction_steps,
  ];
  for (const arr of arrays) {
    if (arr) {
      for (const p of arr) {
        if (p) urls.push(debugImageUrl(p));
      }
    }
  }

  // Fire off background loads via Image objects
  for (const url of urls) {
    const img = new window.Image();
    img.src = url;
  }
}

export function ResultsView({ jobId, onReset }: ResultsViewProps) {
  const [tab,   setTab]   = useState<Tab>("overview");
  const [debug, setDebug] = useState<DebugData | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Track whether assembly tab has been visited (lazy-mount, then keep alive)
  const [assemblyMounted, setAssemblyMounted] = useState(false);

  useEffect(() => {
    fetchDebug(jobId)
      .then((data) => {
        setDebug(data);
        // Eagerly preload all images in background
        preloadDebugImages(data);
      })
      .catch((e) => setError(e.message));
  }, [jobId]);

  // Mount assembly on first visit, keep it alive after
  useEffect(() => {
    if (tab === "assembly") setAssemblyMounted(true);
  }, [tab]);

  if (error) {
    return (
      <div className="text-center py-16 text-danger">
        Ошибка загрузки данных: {error}
      </div>
    );
  }

  if (!debug) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="w-full animate-slide-up">
      {/* Header bar */}
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
        <div>
          <h2 className="text-xl font-bold text-primary">
            Реконструкция завершена
          </h2>
          {debug.pipeline_meta && (
            <p className="text-sm text-secondary mt-0.5">
              {debug.pipeline_meta.n_fragments} фрагментов · {debug.pipeline_meta.timings.total}с всего
            </p>
          )}
          <div className="flex flex-wrap gap-2 mt-2">
            {debug.pipeline_meta?.edge_matcher_loaded && (
              <Badge variant="info">Siamese gate активен</Badge>
            )}
            {debug.inpainting?.status === "OK" && (
              <Badge variant="success">LaMa очистка</Badge>
            )}
            {debug.inpainting?.status === "SKIPPED_NO_MODEL" && (
              <Badge variant="warning">LaMa недоступна</Badge>
            )}
          </div>
        </div>
        <button
          onClick={onReset}
          className="text-sm text-secondary hover:text-primary transition-colors"
        >
          ← Новое изображение
        </button>
      </div>

      {/* Navigation pills */}
      <div className="flex gap-1 mb-6 bg-muted rounded-2xl p-1 w-fit max-w-full overflow-x-auto no-scrollbar">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={cn(
              "px-4 py-2 text-sm font-medium rounded-xl transition-all duration-150 whitespace-nowrap",
              tab === t.key
                ? "bg-white text-primary shadow-card"
                : "text-secondary hover:text-primary"
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content — assembly stays mounted (hidden) to preserve state */}
      {tab !== "assembly" && (
        <div key={tab} className="animate-fade-in">
          {tab === "overview"       && <OverviewView       debug={debug} jobId={jobId} />}
          {tab === "segmentation"   && <SegmentationView   debug={debug} />}
          {tab === "contours"       && <ContoursView       debug={debug} />}
          {tab === "reconstruction" && <ReconstructionView debug={debug} />}
          {tab === "composition"    && <CompositionView    debug={debug} jobId={jobId} />}
        </div>
      )}

      {/* Assembly view: mounted once, hidden when not active */}
      {assemblyMounted && (
        <div
          style={{ display: tab === "assembly" ? "block" : "none" }}
          className={cn(
            "transition-all",
            tab === "assembly"
              ? "relative left-1/2 right-1/2 -ml-[50vw] -mr-[50vw] w-screen px-4"
              : ""
          )}
        >
          <AssemblyView jobId={jobId} />
        </div>
      )}
    </div>
  );
}
