import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatSeconds(s: number): string {
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(0);
  return `${m}m ${sec}s`;
}

export function phaseLabel(phase: string): string {
  const map: Record<string, string> = {
    queued:         "В очереди",
    preprocessing:  "Предобработка",
    segmentation:   "Сегментация",
    contours:       "Анализ контуров",
    reconstruction: "Реконструкция",
    composition:    "Компоновка",
    done:           "Завершено",
    error:          "Ошибка",
  };
  return map[phase] ?? phase;
}

export function phaseColor(phase: string): string {
  const map: Record<string, string> = {
    queued:         "text-secondary",
    preprocessing:  "text-warning",
    segmentation:   "text-blue-500",
    contours:       "text-purple-500",
    reconstruction: "text-orange-500",
    composition:    "text-green-500",
    done:           "text-success",
    error:          "text-danger",
  };
  return map[phase] ?? "text-secondary";
}

function _backendHost(): string {
  if (typeof window === "undefined") return "localhost";
  return window.location.hostname;
}

export function getApiBase(): string {
  return `http://${_backendHost()}:8000`;
}

export function getWsBase(): string {
  return `ws://${_backendHost()}:8000`;
}

export const API_BASE = "http://localhost:8000";
export const WS_BASE  = "ws://localhost:8000";

export function debugImageUrl(path: string): string {
  return `${getApiBase()}${path}`;
}

export function resultImageUrl(jobId: string): string {
  return `${getApiBase()}/result/${jobId}`;
}
