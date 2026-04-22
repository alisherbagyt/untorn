"use client";

import React from "react";
import { CheckCircle, Circle, Loader } from "lucide-react";
import { cn, phaseLabel } from "@/lib/utils";
import { Progress } from "@/components/ui/progress";
import { Card, CardBody, CardHeader, CardTitle } from "@/components/ui/card";
import type { JobStatus } from "@/lib/api";

interface ProcessingViewProps {
  status: JobStatus;
  onCancel?: () => void;
}

const PHASES = [
  { key: "preprocessing",  label: "Предобработка",    desc: "Уменьшение изображения для GPU" },
  { key: "segmentation",   label: "Сегментация",      desc: "Обнаружение фрагментов SAM 2.1" },
  { key: "contours",       label: "Анализ контуров",  desc: "Извлечение краёв и SDF" },
  { key: "reconstruction", label: "Реконструкция",    desc: "Попарное сопоставление и объединение" },
  { key: "composition",    label: "Компоновка",       desc: "Дорисовка в полном разрешении" },
];

type PhaseStatus = "done" | "active" | "pending";

function getPhaseStatus(phaseKey: string, currentPhase: string, overallStatus: string): PhaseStatus {
  const order = PHASES.map((p) => p.key);
  const currentIdx = order.indexOf(currentPhase);
  const phaseIdx   = order.indexOf(phaseKey);

  if (overallStatus === "done") return "done";
  if (phaseIdx < currentIdx)  return "done";
  if (phaseIdx === currentIdx) return "active";
  return "pending";
}

export function ProcessingView({ status }: ProcessingViewProps) {
  const isError = status.status === "error";

  return (
    <div className="w-full max-w-2xl mx-auto flex flex-col gap-5 animate-slide-up">
      {/* Main progress card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">
              {isError ? "Ошибка конвейера" : "Реконструкция..."}
            </CardTitle>
            <span className="text-2xl font-bold text-accent tabular-nums">
              {status.progress}%
            </span>
          </div>
        </CardHeader>
        <CardBody className="space-y-4">
          <Progress value={status.progress} animated={!isError} barClassName={isError ? "bg-danger" : undefined} />

          <div className="text-sm text-secondary">
            {isError
              ? `Ошибка: ${status.error ?? "Неизвестная ошибка"}`
              : `Фаза: ${phaseLabel(status.current_phase)}`}
          </div>

          {/* Phase timeline */}
          <div className="mt-2 space-y-2">
            {PHASES.map((phase, idx) => {
              const ps = getPhaseStatus(phase.key, status.current_phase, status.status);
              return (
                <div key={phase.key} className="flex items-center gap-3">
                  {/* Connector line */}
                  <div className="flex flex-col items-center">
                    <div
                      className={cn(
                        "w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 transition-all",
                        ps === "done"   && "bg-success/10 text-success",
                        ps === "active" && "bg-accent/10 text-accent",
                        ps === "pending"&& "bg-muted text-secondary/40"
                      )}
                    >
                      {ps === "done"   && <CheckCircle size={14} />}
                      {ps === "active" && <Loader size={14} className="animate-spin" />}
                      {ps === "pending"&& <Circle size={14} />}
                    </div>
                    {idx < PHASES.length - 1 && (
                      <div className={cn("w-px h-3 mt-0.5",
                        ps === "done" ? "bg-success/30" : "bg-border"
                      )} />
                    )}
                  </div>

                  <div className={cn("flex-1 min-w-0 pb-2", idx < PHASES.length - 1 && "border-b border-border/50")}>
                    <div className="flex items-baseline gap-2">
                      <span className={cn(
                        "text-sm font-medium",
                        ps === "done"   && "text-success",
                        ps === "active" && "text-primary",
                        ps === "pending"&& "text-secondary/50"
                      )}>
                        {phase.label}
                      </span>
                      {ps === "active" && (
                        <span className="text-xs text-secondary">выполняется…</span>
                      )}
                    </div>
                    <p className={cn("text-xs mt-0.5",
                      ps === "pending" ? "text-secondary/30" : "text-secondary"
                    )}>
                      {phase.desc}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </CardBody>
      </Card>

    </div>
  );
}
