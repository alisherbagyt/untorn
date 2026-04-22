"use client";

import React, { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine,
} from "recharts";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Card, CardHeader, CardTitle, CardBody } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { debugImageUrl } from "@/lib/api";
import type { DebugData } from "@/lib/api";

interface ReconstructionViewProps {
  debug: DebugData;
}

export function ReconstructionView({ debug }: ReconstructionViewProps) {
  const steps = (debug.steps ?? []).map((raw, idx) => {
    const step = Number(raw?.step ?? idx + 1);
    const anchor = Number(raw?.anchor ?? -1);
    const attached = Number(raw?.attached ?? -1);
    const gapScoreRaw = (raw as { gap_score?: number; stotal?: number })?.gap_score
      ?? (raw as { stotal?: number })?.stotal
      ?? 0;
    const dx = Number(raw?.dx ?? 0);
    const dy = Number(raw?.dy ?? 0);

    return {
      step: Number.isFinite(step) ? step : idx + 1,
      phase: raw?.phase ?? "merge",
      anchor: Number.isFinite(anchor) ? anchor : -1,
      attached: Number.isFinite(attached) ? attached : -1,
      dx: Number.isFinite(dx) ? dx : 0,
      dy: Number.isFinite(dy) ? dy : 0,
      gap_score: Number.isFinite(Number(gapScoreRaw)) ? Number(gapScoreRaw) : 0,
    };
  });
  const paths = debug.paths;
  const neighbors = debug.neighbors ?? [];
  const translations = debug.translations ?? {};

  const stepImages = paths?.reconstruction_steps ?? [];
  const [stepIdx, setStepIdx] = useState(0);

  const currentStep = steps[stepIdx];
  const currentImage = stepImages[stepIdx];

  // Gap score chart
  const scoreData = steps.map((s) => ({
    step: s.step,
    score: parseFloat(s.gap_score.toFixed(2)),
    pair: `${s.anchor}→${s.attached}`,
  }));

  // Translation scatter data
  const translationData = Object.entries(translations).map(([id, t]) => ({
    id: `#${id}`,
    dx: t.dx,
    dy: t.dy,
  }));

  return (
    <div className="space-y-4">
      {/* Step viewer */}
      {stepImages.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <CardTitle>Шаги реконструкции</CardTitle>
                {currentStep && (
                  <Badge variant="info">
                    {currentStep.anchor} → {currentStep.attached}
                  </Badge>
                )}
              </div>
              <span className="text-sm text-secondary tabular-nums">
                {stepIdx + 1} / {stepImages.length}
              </span>
            </div>
          </CardHeader>
          <CardBody className="space-y-4">
            {/* Image viewer */}
            <div className="relative bg-muted rounded-xl overflow-hidden" style={{ minHeight: "260px" }}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                key={currentImage}
                src={debugImageUrl(currentImage)}
                alt={`Step ${stepIdx + 1}`}
                className="w-full object-contain max-h-[400px] animate-fade-in"
              />

              {/* Step info overlay */}
              {currentStep && (
                <div className="absolute bottom-3 left-3 bg-white/90 backdrop-blur-sm rounded-xl px-3 py-2 text-xs shadow-soft">
                  <div className="font-semibold text-primary">Шаг {currentStep.step}</div>
                  <div className="text-secondary">
                    Якорь #{currentStep.anchor} + Фрагмент #{currentStep.attached}
                  </div>
                  <div className="text-accent font-mono mt-0.5">
                    оценка: {currentStep.gap_score.toFixed(2)} · Δ({currentStep.dx.toFixed(1)}, {currentStep.dy.toFixed(1)})
                  </div>
                </div>
              )}
            </div>

            {/* Slider + nav */}
            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setStepIdx((i) => Math.max(0, i - 1))}
                disabled={stepIdx === 0}
              >
                <ChevronLeft size={14} />
              </Button>
              <div className="flex-1">
                <Slider
                  value={stepIdx}
                  min={0}
                  max={Math.max(0, stepImages.length - 1)}
                  onChange={setStepIdx}
                  label="Step"
                />
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setStepIdx((i) => Math.min(stepImages.length - 1, i + 1))}
                disabled={stepIdx === stepImages.length - 1}
              >
                <ChevronRight size={14} />
              </Button>
            </div>
          </CardBody>
        </Card>
      )}

      {/* Gap score chart */}
      {scoreData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Оценка разрыва по шагам объединения</CardTitle>
          </CardHeader>
          <CardBody>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={scoreData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E8E8E3" />
                <XAxis dataKey="step" tick={{ fontSize: 11, fill: "#6B6B71" }} label={{ value: "Шаг", position: "insideBottom", offset: -2, fontSize: 11, fill: "#6B6B71" }} />
                <YAxis tick={{ fontSize: 11, fill: "#6B6B71" }} width={40} />
                <Tooltip
                  formatter={(val: number) => [val.toFixed(2), "Оценка разрыва"]}
                  labelFormatter={(l) => {
                    const s = scoreData.find((d) => d.step === l);
                    return `Шаг ${l}: ${s?.pair ?? ""}`;
                  }}
                  contentStyle={{ borderRadius: 12, border: "1px solid #E8E8E3", fontSize: 12 }}
                />
                <ReferenceLine
                  y={scoreData.reduce((a, b) => a + b.score, 0) / scoreData.length}
                  stroke="#F59E0B"
                  strokeDasharray="4 4"
                  label={{ value: "ср", position: "right", fontSize: 10, fill: "#F59E0B" }}
                />
                <Line
                  type="monotone"
                  dataKey="score"
                  stroke="#4F46E5"
                  strokeWidth={2}
                  dot={{ fill: "#4F46E5", r: 3 }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardBody>
        </Card>
      )}

      {/* Merge log table */}
      {steps.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Журнал объединений</CardTitle>
          </CardHeader>
          <CardBody>
            <div className="overflow-x-auto no-scrollbar">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border">
                    {["Шаг", "Фаза", "Якорь", "Прикреплён", "dx", "dy", "Оценка"].map((h) => (
                      <th key={h} className="text-left text-secondary font-medium pb-2 pr-4">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {steps.map((s) => (
                    <tr
                      key={s.step}
                      onClick={() => setStepIdx(s.step - 1)}
                      className={`border-b border-border/50 cursor-pointer hover:bg-muted transition-colors ${
                        stepIdx === s.step - 1 ? "bg-accent-light" : ""
                      }`}
                    >
                      <td className="py-2 pr-4 font-semibold text-primary">{s.step}</td>
                      <td className="py-2 pr-4 text-secondary">{s.phase}</td>
                      <td className="py-2 pr-4 font-mono text-accent">#{s.anchor}</td>
                      <td className="py-2 pr-4 font-mono text-accent">#{s.attached}</td>
                      <td className="py-2 pr-4 font-mono">{s.dx.toFixed(1)}</td>
                      <td className="py-2 pr-4 font-mono">{s.dy.toFixed(1)}</td>
                      <td className="py-2 pr-4 font-semibold text-success">{s.gap_score.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardBody>
        </Card>
      )}

      {/* Neighbor graph */}
      {paths?.neighbor_graph && (
        <Card>
          <CardHeader>
            <CardTitle>Граф смежности</CardTitle>
          </CardHeader>
          <CardBody>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={debugImageUrl(paths.neighbor_graph)}
              alt="Neighbor graph"
              className="w-full rounded-xl object-contain bg-muted max-h-[360px]"
            />
          </CardBody>
        </Card>
      )}
    </div>
  );
}
