"use client";

import React from "react";
import { Card, CardHeader, CardTitle, CardBody } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { formatSeconds, debugImageUrl, resultImageUrl } from "@/lib/utils";
import type { DebugData } from "@/lib/api";

// Re-export from utils
function fmtS(s: number) { return formatSeconds(s); }

interface OverviewViewProps {
  debug: DebugData;
  jobId: string;
}

export function OverviewView({ debug, jobId }: OverviewViewProps) {
  const meta = debug.pipeline_meta;
  const timings = meta?.timings ?? {};

  const statItems = [
    { label: "Фрагменты",           value: meta?.n_fragments ?? "—" },
    { label: "Исходный размер",     value: meta ? `${meta.image_size.w}×${meta.image_size.h}` : "—" },
    { label: "Рабочий размер",      value: meta ? `${meta.working_size.w}×${meta.working_size.h}` : "—" },
    { label: "Коэффициент масштаба",value: meta ? `${meta.scale_factor}×` : "—" },
  ];

  const timingItems = [
    { label: "Предобработка",  key: "preprocess",      color: "#F59E0B" },
    { label: "Сегментация",    key: "segmentation",    color: "#3B82F6" },
    { label: "Контуры",        key: "contour_analysis",color: "#8B5CF6" },
    { label: "Реконструкция",  key: "reconstruction",  color: "#F97316" },
    { label: "Компоновка",     key: "composition",     color: "#10B981" },
  ];
  const total = timings["total"] ?? Object.values(timings).reduce((a, b) => a + b, 0);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Stats */}
      <Card>
        <CardHeader>
          <CardTitle>Итоги конвейера</CardTitle>
        </CardHeader>
        <CardBody className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            {statItems.map((item) => (
              <div key={item.label} className="bg-muted rounded-xl p-3">
                <div className="text-xs text-secondary">{item.label}</div>
                <div className="text-lg font-semibold text-primary mt-0.5">{String(item.value)}</div>
              </div>
            ))}
          </div>
          {meta && (
            <div className="flex items-center gap-2 mt-1">
              <Badge variant={meta.status === "SUCCESS" ? "success" : "danger"}>
                {meta.status}
              </Badge>
              <span className="text-xs text-secondary">Итого: {fmtS(total)}</span>
            </div>
          )}
        </CardBody>
      </Card>

      {/* Timing breakdown */}
      <Card>
        <CardHeader>
          <CardTitle>Время фаз</CardTitle>
        </CardHeader>
        <CardBody className="space-y-2.5">
          {timingItems.map(({ label, key, color }) => {
            const t = timings[key] ?? 0;
            const pct = total > 0 ? (t / total) * 100 : 0;
            return (
              <div key={key}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-secondary">{label}</span>
                  <span className="font-medium text-primary">{fmtS(t)}</span>
                </div>
                <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${pct}%`, backgroundColor: color }}
                  />
                </div>
              </div>
            );
          })}
        </CardBody>
      </Card>

      {/* Result image */}
      <Card className="md:col-span-2">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Итоговая реконструкция</CardTitle>
            <a
              href={resultImageUrl(jobId)}
              download
              className="text-xs text-accent hover:underline"
            >
              Скачать ↓
            </a>
          </div>
        </CardHeader>
        <CardBody>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={resultImageUrl(jobId)}
            alt="Reconstructed result"
            className="w-full rounded-xl object-contain max-h-[420px] bg-muted"
          />
        </CardBody>
      </Card>
    </div>
  );
}
