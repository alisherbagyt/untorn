"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";
import { uploadImage, fetchStatus, createWebSocket } from "@/lib/api";
import type { JobStatus } from "@/lib/api";
import { UploadZone }    from "@/components/UploadZone";
import { ProcessingView } from "@/components/ProcessingView";
import { ResultsView }    from "@/components/ResultsView";

type AppView = "upload" | "processing" | "results";

export default function Home() {
  const [view,   setView]   = useState<AppView>("upload");
  const [jobId,  setJobId]  = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);

  // Global drag-over state for the full-page overlay
  const [globalDrag, setGlobalDrag] = useState(false);

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null);

  // ── Upload handler ────────────────────────────────────────────────────────
  const handleUpload = useCallback(async (file: File) => {
    const { job_id } = await uploadImage(file);
    setJobId(job_id);
    setView("processing");
    setStatus({
      job_id,
      status:        "queued",
      progress:      0,
      current_phase: "queued",
      logs:          [],
    });

    // Connect WebSocket
    const ws = createWebSocket(job_id);
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data) as Partial<JobStatus>;
      setStatus((prev) => ({ ...(prev as JobStatus), ...data, job_id }));

      if (data.status === "done") {
        ws.close();
        setView("results");
      } else if (data.status === "error") {
        ws.close();
      }
    };

    ws.onerror = () => {
      // Fallback to polling if WS fails
      ws.close();
      startPolling(job_id);
    };

    ws.onclose = () => {
      wsRef.current = null;
    };
  }, []);

  // ── Polling fallback ──────────────────────────────────────────────────────
  function startPolling(id: string) {
    const interval = setInterval(async () => {
      try {
        const s = await fetchStatus(id);
        setStatus(s);
        if (s.status === "done") {
          clearInterval(interval);
          setView("results");
        } else if (s.status === "error") {
          clearInterval(interval);
        }
      } catch {
        clearInterval(interval);
      }
    }, 800);
  }

  // ── Reset ─────────────────────────────────────────────────────────────────
  const handleReset = useCallback(() => {
    wsRef.current?.close();
    setView("upload");
    setJobId(null);
    setStatus(null);
  }, []);

  // ── Global drag-and-drop overlay ──────────────────────────────────────────
  const { getRootProps: getGlobalRootProps, getInputProps: getGlobalInputProps, isDragActive: isGlobalDrag } =
    useDropzone({
      accept: { "image/*": [".tif", ".tiff", ".jpg", ".jpeg", ".png"] },
      noClick: true,
      disabled: view === "processing",
      onDropAccepted: ([file]) => {
        if (view !== "processing") handleUpload(file);
      },
    });

  return (
    <div
      {...getGlobalRootProps()}
      className="min-h-screen bg-background relative"
    >
      <input {...getGlobalInputProps()} />

      {/* Global drag overlay */}
      {isGlobalDrag && view !== "processing" && (
        <div className="fixed inset-0 z-50 bg-accent/5 border-4 border-dashed border-accent/40 rounded-none flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <div className="w-20 h-20 bg-accent rounded-3xl flex items-center justify-center mx-auto mb-4">
              <Upload size={36} className="text-white" />
            </div>
            <p className="text-xl font-bold text-accent">Бросьте для обработки</p>
            <p className="text-sm text-secondary mt-1">Отпустите для начала реконструкции</p>
          </div>
        </div>
      )}

      {/* App shell */}
      <div className="max-w-5xl mx-auto px-4 py-8">
        {/* Header */}
        <header className="mb-10">
          <div className="flex items-center gap-3">
            <div>
              <h1 className="text-lg font-bold text-primary leading-none">UNTORN</h1>
              <p className="text-xs text-secondary mt-0.5">Цифровая Реставрация</p>
            </div>
          </div>
        </header>

        {/* Views */}
        {view === "upload" && (
          <div className="flex flex-col items-center gap-8 animate-slide-up">
            <div className="text-center max-w-lg">
              <h2 className="text-3xl font-bold text-primary text-balance leading-tight">
                Цифровая Реставрация
              </h2>
              <p className="text-secondary mt-3 text-balance">
                Загрузите фото фрагментов разорванной бумаги.<br /> UNTORN использует SAM 2.1
                для сегментации каждого фрагмента и алгоритмически собирает их в
                исходный документ.
              </p>
            </div>

            <UploadZone onUpload={handleUpload} />

            <p className="text-xs text-secondary/50 text-center">
              Работает полностью офлайн · Поддержка TIFF, JPEG, PNG · Ускорение GPU
            </p>
          </div>
        )}

        {view === "processing" && status && (
          <ProcessingView status={status} onCancel={handleReset} />
        )}

        {view === "results" && jobId && (
          <ResultsView jobId={jobId} onReset={handleReset} />
        )}
      </div>
    </div>
  );
}
