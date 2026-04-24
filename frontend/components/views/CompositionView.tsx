"use client";

import React, { useState } from "react";
import { Card, CardHeader, CardTitle, CardBody } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { debugImageUrl, resultImageUrl } from "@/lib/api";
import type { DebugData } from "@/lib/api";

type CompositionLayer = "raw" | "gap" | "inpainted" | "cleaned";

interface CompositionViewProps {
  debug: DebugData;
  jobId: string;
}

export function CompositionView({ debug, jobId }: CompositionViewProps) {
  const paths = debug.paths;
  const meta  = debug.composition;
  const inp   = debug.inpainting;
  const gapPixels =
    meta?.gap_pixels_inpainted ??
    // Compatibility with newer backend metadata naming.
    (meta as { gap_pixels_detected?: number } | undefined)?.gap_pixels_detected ??
    0;

  // Default to cleaned if it's available, otherwise the classical inpaint.
  const hasCleaned =
    !!paths?.inpainting_cleaned &&
    inp?.status !== "SKIPPED_NO_MODEL" &&
    inp?.status !== "FAILED";
  const [layer, setLayer] = useState<CompositionLayer>(hasCleaned ? "cleaned" : "inpainted");

  const layers: { key: CompositionLayer; label: string; desc: string }[] = [
    { key: "raw",       label: "Необработанная компоновка", desc: "Фрагменты размещены на вычисленных позициях" },
    { key: "gap",       label: "Маска пробелов",            desc: "Определены недостающие области" },
    { key: "inpainted", label: "Классическая дорисовка",    desc: "Быстрое заполнение пробелов (cv2.inpaint)" },
    { key: "cleaned",   label: "Чистая бумага",             desc: "LaMa убрал швы; текст сохранён" },
  ];

  const layerUrl: Record<CompositionLayer, string | undefined> = {
    raw:       paths?.composition_raw        ? debugImageUrl(paths.composition_raw)        : undefined,
    gap:       paths?.composition_gap        ? debugImageUrl(paths.composition_gap)        : undefined,
    inpainted: paths?.composition_inpainted  ? debugImageUrl(paths.composition_inpainted)  : undefined,
    cleaned:   paths?.inpainting_cleaned     ? debugImageUrl(paths.inpainting_cleaned)     : undefined,
  };

  // If LaMa was skipped or failed, hide the cleaned pill entirely
  const visibleLayers = hasCleaned ? layers : layers.filter((l) => l.key !== "cleaned");

  const coveragePct =
    meta && meta.canvas_w > 0 && meta.canvas_h > 0
      ? ((gapPixels / (meta.canvas_w * meta.canvas_h)) * 100).toFixed(1)
      : null;

  // Stats row
  const stats: { label: string; value: string }[] = [];
  if (meta) {
    stats.push({ label: "Размер холста", value: `${meta.canvas_w}×${meta.canvas_h}` });
    stats.push({ label: "Пиксели пробелов", value: gapPixels.toLocaleString() });
    stats.push({ label: "Покрытие пробелов", value: coveragePct !== null ? `${coveragePct}%` : "-" });
  }
  if (inp && inp.status === "OK") {
    stats.push({
      label: "Швы закрашены (LaMa)",
      value: `${(inp.mask_pixels ?? 0).toLocaleString()} px`,
    });
    stats.push({
      label: "Время LaMa",
      value: `${inp.duration_s ?? "?"} с${inp.refine ? " · refine" : ""}`,
    });
  } else if (inp && inp.status === "SKIPPED_NO_MODEL") {
    stats.push({ label: "LaMa", value: "не загружена" });
  } else if (inp && inp.status === "FAILED") {
    stats.push({ label: "LaMa", value: "ошибка" });
  }

  return (
    <div className="space-y-4">
      {/* Stats row */}
      {stats.length > 0 && (
        <div className={`grid gap-3 ${stats.length >= 4 ? "grid-cols-2 md:grid-cols-5" : "grid-cols-3"}`}>
          {stats.map((s) => (
            <Card key={s.label}>
              <CardBody className="pt-4">
                <div className="text-xs text-secondary">{s.label}</div>
                <div className="text-xl font-bold text-primary mt-1 truncate">{s.value}</div>
              </CardBody>
            </Card>
          ))}
        </div>
      )}

      {/* Layer viewer */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between flex-wrap gap-2">
            <CardTitle>Слои компоновки</CardTitle>
            {/* Layer pills */}
            <div className="flex gap-1 flex-wrap">
              {visibleLayers.map((l) => (
                <button
                  key={l.key}
                  onClick={() => setLayer(l.key)}
                  className={`text-xs px-3 py-1.5 rounded-full font-medium transition-all ${
                    layer === l.key
                      ? "bg-accent text-white shadow-soft"
                      : "bg-muted text-secondary hover:bg-border"
                  }`}
                >
                  {l.label}
                </button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardBody>
          <p className="text-xs text-secondary mb-3">
            {visibleLayers.find((l) => l.key === layer)?.desc}
          </p>
          {layerUrl[layer] ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={layer}
              src={layerUrl[layer]}
              alt={layer}
              className="w-full rounded-xl object-contain bg-muted max-h-[480px] animate-fade-in"
            />
          ) : (
            <div className="h-48 bg-muted rounded-xl flex items-center justify-center text-secondary text-sm">
              Изображение недоступно
            </div>
          )}
        </CardBody>
      </Card>

      {/* Side-by-side comparison */}
      <Card>
        <CardHeader>
          <CardTitle>До и после</CardTitle>
        </CardHeader>
        <CardBody>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs text-secondary">Исходное изображение</span>
              </div>
              {paths?.input ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={debugImageUrl(paths.input)}
                  alt="Input"
                  className="w-full rounded-xl object-contain bg-muted"
                />
              ) : (
                <div className="h-40 bg-muted rounded-xl" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs text-secondary">Итоговый результат</span>
                {hasCleaned ? (
                  <Badge variant="success">LaMa · без швов</Badge>
                ) : (
                  <Badge variant="success">Восстановлено</Badge>
                )}
              </div>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={resultImageUrl(jobId)}
                alt="Result"
                className="w-full rounded-xl object-contain bg-muted"
              />
            </div>
          </div>
        </CardBody>
      </Card>

      {/* Scar-mask inspection (only if LaMa ran successfully) */}
      {hasCleaned && paths?.inpainting_mask && paths?.inpainting_before && (
        <Card>
          <CardHeader>
            <CardTitle>Маска швов (LaMa)</CardTitle>
          </CardHeader>
          <CardBody>
            <p className="text-xs text-secondary mb-3">
              Узкая полоса вдоль краёв фрагментов (текст исключён). Именно эти пиксели перерисовывает LaMa.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {[
                { src: paths.inpainting_before,  label: "До (с швами)" },
                { src: paths.inpainting_mask,    label: "Маска (что перерисовать)" },
                { src: paths.inpainting_cleaned, label: "После (чистая бумага)" },
              ].map((x) => (
                <div key={x.label}>
                  <div className="text-xs text-secondary mb-1">{x.label}</div>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={debugImageUrl(x.src!)}
                    alt={x.label}
                    className="w-full rounded-xl object-contain bg-muted"
                  />
                </div>
              ))}
            </div>
          </CardBody>
        </Card>
      )}
    </div>
  );
}
