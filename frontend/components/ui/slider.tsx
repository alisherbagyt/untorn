"use client";
import * as React from "react";
import { cn } from "@/lib/utils";

interface SliderProps {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
  className?: string;
  label?: string;
}

export const Slider = ({ value, min = 0, max = 100, step = 1, onChange, className, label }: SliderProps) => {
  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div className={cn("flex items-center gap-3", className)}>
      {label && <span className="text-xs text-secondary whitespace-nowrap">{label}</span>}
      <div className="relative flex-1 h-5 flex items-center">
        <div className="w-full h-1.5 bg-muted rounded-full">
          <div
            className="h-full bg-accent rounded-full transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        {/* Thumb */}
        <div
          className="absolute w-4 h-4 bg-white border-2 border-accent rounded-full shadow -translate-y-0 pointer-events-none"
          style={{ left: `calc(${pct}% - 8px)` }}
        />
      </div>
      <span className="text-xs font-medium text-primary w-6 text-right">{value}</span>
    </div>
  );
};
