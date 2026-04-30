"use client";

import React, { useCallback, useEffect, useRef, useState, useMemo } from "react";
import {
  Download,
  RotateCcw,
  Loader,
  ZoomIn,
  ZoomOut,
  Maximize,
  Layers,
  Eye,
  EyeOff,
  Lock,
  Unlock,
  ChevronUp,
  ChevronDown,
  Grid3X3,
  Undo2,
  X,
  Move,
  Settings,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  fetchBoardData,
  boardFragmentUrl,
  exportBoard,
} from "@/lib/api";
import type { BoardData } from "@/lib/api";

// ── Types ───────────────────────────────────────────────────────────────────

interface FragmentState {
  id: number;
  x: number;
  y: number;
  rotation: number;
  width: number;
  height: number;
  placed: boolean;
  zIndex: number;
  visible: boolean;
  locked: boolean;
}

interface AssemblyViewProps {
  jobId: string;
}

// ── Constants ───────────────────────────────────────────────────────────────

const ROTATION_STEP = 0.15; // degrees per px of horizontal drag
const MIN_ZOOM = 0.2;
const MAX_ZOOM = 3.5;
const ZOOM_SENSITIVITY = 0.001;
const ZOOM_BUTTON_STEP = 0.15;
const GRID_SIZE = 20;
const NUDGE_PX = 1;
const NUDGE_PX_SHIFT = 10;
const MAX_UNDO = 50;

// ── Component ───────────────────────────────────────────────────────────────

export function AssemblyView({ jobId }: AssemblyViewProps) {
  /* ── Board data ──────────────────────────────────────────────────────── */
  const [boardData, setBoardData] = useState<BoardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  /* ── Fragments ───────────────────────────────────────────────────────── */
  const [fragments, setFragments] = useState<FragmentState[]>([]);
  const [initialFragments, setInitialFragments] = useState<FragmentState[]>([]);

  /* ── Blob image cache ────────────────────────────────────────────────── */
  const blobUrls = useRef<Map<number, string>>(new Map());
  const [imagesLoaded, setImagesLoaded] = useState(0);

  /* ── Interaction state ───────────────────────────────────────────────── */
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [dragState, setDragState] = useState<{
    fragmentId: number;
    startMouseX: number;
    startMouseY: number;
    startFragX: number;
    startFragY: number;
  } | null>(null);
  const [rotateState, setRotateState] = useState<{
    fragmentId: number;
    startMouseX: number;
    startRotation: number;
  } | null>(null);

  /* ── Canvas pan / zoom ───────────────────────────────────────────────── */
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(0.5);
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0, panX: 0, panY: 0 });

  /* ── UI toggles ──────────────────────────────────────────────────────── */
  const [showLayers, setShowLayers] = useState(true);
  const [showInspector, setShowInspector] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [exportScale, setExportScale] = useState(2);
  const [exporting, setExporting] = useState(false);
  const [showRaw, setShowRaw] = useState(false);   // true => disable LaMa cleaning
  const [refine, setRefine] = useState(false);     // true => LaMa refinement (slow)
  const [showAdvanced, setShowAdvanced] = useState(false);

  /* ── Space-to-pan ─────────────────────────────────────────────────────── */
  const isSpaceDown = useRef(false);
  const [spaceHeld, setSpaceHeld] = useState(false);

  /* ── Scrollbar drag ──────────────────────────────────────────────────── */
  const [scrollDrag, setScrollDrag] = useState<{
    axis: "x" | "y";
    startMouse: number;
    startPan: number;
    scale: number; // maps px of scrollbar drag → px of pan
  } | null>(null);

  /* ── Undo stack ──────────────────────────────────────────────────────── */
  const undoStack = useRef<FragmentState[][]>([]);
  const [undoCount, setUndoCount] = useState(0); // just to trigger re-renders

  /* ── Undo helpers ────────────────────────────────────────────────────── */

  const saveUndo = useCallback(() => {
    setFragments((current) => {
      undoStack.current.push(current.map((f) => ({ ...f })));
      if (undoStack.current.length > MAX_UNDO) undoStack.current.shift();
      setUndoCount(undoStack.current.length);
      return current;
    });
  }, []);

  const undo = useCallback(() => {
    const prev = undoStack.current.pop();
    setUndoCount(undoStack.current.length);
    if (prev) setFragments(prev);
  }, []);

  /* ── Refs ─────────────────────────────────────────────────────────────── */
  const containerRef = useRef<HTMLDivElement>(null);
  const topZRef = useRef(100);

  // Keep zoom/pan in refs so the native wheel listener always sees latest
  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  useEffect(() => { zoomRef.current = zoom; }, [zoom]);
  useEffect(() => { panRef.current = pan; }, [pan]);

  /* ── Compute fit zoom ────────────────────────────────────────────────── */

  const clampZoom = useCallback(
    (z: number) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z)),
    []
  );

  const computeContentBounds = useCallback((items: FragmentState[]) => {
    if (!items.length) {
      return {
        minX: 0,
        minY: 0,
        maxX: boardData?.canvas.width ?? 0,
        maxY: boardData?.canvas.height ?? 0,
      };
    }
    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (const f of items) {
      minX = Math.min(minX, f.x);
      minY = Math.min(minY, f.y);
      maxX = Math.max(maxX, f.x + f.width);
      maxY = Math.max(maxY, f.y + f.height);
    }
    return { minX, minY, maxX, maxY };
  }, [boardData]);

  const computeFitZoom = useCallback((items: FragmentState[]) => {
    if (!containerRef.current) return 0.5;
    const rect = containerRef.current.getBoundingClientRect();
    const bounds = computeContentBounds(items);
    const width = Math.max(1, bounds.maxX - bounds.minX);
    const height = Math.max(1, bounds.maxY - bounds.minY);
    const scaleX = (rect.width - 120) / width;
    const scaleY = (rect.height - 120) / height;
    return clampZoom(Math.min(scaleX, scaleY, 1));
  }, [computeContentBounds, clampZoom]);

  const fitToView = useCallback((items: FragmentState[] = fragments) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const bounds = computeContentBounds(items);
    const fitZoom = computeFitZoom(items);
    const contentW = Math.max(1, bounds.maxX - bounds.minX);
    const contentH = Math.max(1, bounds.maxY - bounds.minY);
    setZoom(fitZoom);
    setPan({
      x: (rect.width - contentW * fitZoom) / 2 - bounds.minX * fitZoom,
      y: (rect.height - contentH * fitZoom) / 2 - bounds.minY * fitZoom,
    });
  }, [computeContentBounds, computeFitZoom, fragments]);

  /* ── Load board data ─────────────────────────────────────────────────── */

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);
    fetchBoardData(jobId)
      .then((data) => {
        if (!mounted) return;
        setBoardData(data);
        const minX = Math.min(...data.fragments.map((f) => f.x));
        const minY = Math.min(...data.fragments.map((f) => f.y));
        const initial = data.fragments.map((f, i) => ({
          id: f.id,
          x: f.x - minX,
          y: f.y - minY,
          rotation: f.rotation,
          width: f.width,
          height: f.height,
          placed: f.placed,
          zIndex: i + 1,
          visible: true,
          locked: false,
        }));
        setFragments(initial);
        setInitialFragments(initial.map((f) => ({ ...f })));
        setLoading(false);
        requestAnimationFrame(() => {
          if (!mounted) return;
          fitToView(initial);
        });
      })
      .catch((e) => {
        if (!mounted) return;
        setError(e?.message ?? "Failed to load board data");
        setLoading(false);
      });
    return () => { mounted = false; };
  }, [jobId]);

  /* ── Load fragment images (blob cache) ───────────────────────────────── */

  useEffect(() => {
    let mounted = true;
    if (!boardData) return () => { mounted = false; };

    setImagesLoaded(0);
    blobUrls.current.forEach((url) => URL.revokeObjectURL(url));
    blobUrls.current.clear();

    const load = async () => {
      for (const frag of boardData.fragments) {
        try {
          const res = await fetch(boardFragmentUrl(jobId, frag.id));
          const blob = await res.blob();
          if (!mounted) return;
          const url = URL.createObjectURL(blob);
          blobUrls.current.set(frag.id, url);
          setImagesLoaded((c) => c + 1);
        } catch {
          if (!mounted) return;
          setImagesLoaded((c) => c + 1);
        }
      }
    };

    load();
    return () => {
      mounted = false;
      blobUrls.current.forEach((url) => URL.revokeObjectURL(url));
      blobUrls.current.clear();
    };
  }, [boardData, jobId]);

  /* ── Scroll-wheel zoom ───────────────────────────────────────────────── */

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const handler = (e: WheelEvent) => {
      if (!boardData) return;
      e.preventDefault();

      const rect = el.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const prevZoom = zoomRef.current;
      const newZoom = clampZoom(prevZoom - e.deltaY * ZOOM_SENSITIVITY);
      if (newZoom === prevZoom) return;

      const ratio = newZoom / prevZoom;
      const prevPan = panRef.current;
      const newPanX = mouseX - ratio * (mouseX - prevPan.x);
      const newPanY = mouseY - ratio * (mouseY - prevPan.y);

      setZoom(newZoom);
      setPan({ x: newPanX, y: newPanY });
    };

    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, [boardData, clampZoom]);

  /* ── Resize → re-fit ─────────────────────────────────────────────────── */

  useEffect(() => {
    if (!boardData) return;
    const onResize = () => fitToView();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [boardData, fitToView]);

  /* ── Keyboard shortcuts ──────────────────────────────────────────────── */

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      // Space — hold to pan
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        isSpaceDown.current = true;
        setSpaceHeld(true);
        return;
      }

      // Undo
      if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault();
        undo();
        return;
      }

      if (e.key === "Escape") {
        setSelectedId(null);
        return;
      }

      // Delete — deselect (don't remove fragments)
      if (e.key === "Delete" || e.key === "Backspace") {
        if (selectedId !== null) {
          setSelectedId(null);
        }
        return;
      }

      // Arrow keys — nudge selected fragment
      if (selectedId !== null && ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)) {
        e.preventDefault();
        const step = e.shiftKey ? NUDGE_PX_SHIFT : NUDGE_PX;
        const dx = e.key === "ArrowLeft" ? -step : e.key === "ArrowRight" ? step : 0;
        const dy = e.key === "ArrowUp" ? -step : e.key === "ArrowDown" ? step : 0;
        saveUndo();
        setFragments((prev) =>
          prev.map((f) =>
            f.id === selectedId && !f.locked
              ? { ...f, x: f.x + dx, y: f.y + dy }
              : f
          )
        );
        return;
      }

      // Bracket keys — rotate selected fragment
      if (selectedId !== null && (e.key === "[" || e.key === "]")) {
        e.preventDefault();
        const deg = e.shiftKey ? 15 : 1;
        const delta = e.key === "[" ? -deg : deg;
        saveUndo();
        setFragments((prev) =>
          prev.map((f) =>
            f.id === selectedId && !f.locked
              ? { ...f, rotation: f.rotation + delta }
              : f
          )
        );
        return;
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        isSpaceDown.current = false;
        setSpaceHeld(false);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [selectedId, undo, saveUndo]);

  /* ── Fragment dragging ───────────────────────────────────────────────── */

  const handleFragmentMouseDown = useCallback(
    (e: React.MouseEvent, fragId: number) => {
      e.stopPropagation();
      e.preventDefault();

      // Space held → pan instead of dragging fragment
      if (isSpaceDown.current) {
        const p = panRef.current;
        setIsPanning(true);
        setPanStart({ x: e.clientX, y: e.clientY, panX: p.x, panY: p.y });
        return;
      }

      setSelectedId(fragId);
      topZRef.current += 1;

      saveUndo();

      setFragments((prev) => {
        const frag = prev.find((f) => f.id === fragId);
        if (frag && !frag.locked) {
          setDragState({
            fragmentId: fragId,
            startMouseX: e.clientX,
            startMouseY: e.clientY,
            startFragX: frag.x,
            startFragY: frag.y,
          });
        }
        return prev.map((f) =>
          f.id === fragId ? { ...f, zIndex: topZRef.current } : f
        );
      });
    },
    [saveUndo]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (scrollDrag) {
        const mousePos = scrollDrag.axis === "x" ? e.clientX : e.clientY;
        const delta = mousePos - scrollDrag.startMouse;
        const newVal = scrollDrag.startPan - delta * scrollDrag.scale;
        if (scrollDrag.axis === "x") {
          setPan((prev) => ({ ...prev, x: newVal }));
        } else {
          setPan((prev) => ({ ...prev, y: newVal }));
        }
      } else if (dragState) {
        const dx = (e.clientX - dragState.startMouseX) / zoom;
        const dy = (e.clientY - dragState.startMouseY) / zoom;
        setFragments((prev) =>
          prev.map((f) =>
            f.id === dragState.fragmentId
              ? { ...f, x: dragState.startFragX + dx, y: dragState.startFragY + dy }
              : f
          )
        );
      } else if (rotateState) {
        const dx = e.clientX - rotateState.startMouseX;
        setFragments((prev) =>
          prev.map((f) =>
            f.id === rotateState.fragmentId
              ? { ...f, rotation: rotateState.startRotation + dx * ROTATION_STEP }
              : f
          )
        );
      } else if (isPanning) {
        setPan({
          x: panStart.panX + (e.clientX - panStart.x),
          y: panStart.panY + (e.clientY - panStart.y),
        });
      }
    },
    [scrollDrag, dragState, rotateState, isPanning, zoom, panStart]
  );

  const handleMouseUp = useCallback(() => {
    setDragState(null);
    setRotateState(null);
    setIsPanning(false);
    setScrollDrag(null);
  }, []);

  /* ── Canvas panning (click empty area or middle-click) ───────────────── */

  const handleCanvasMouseDown = useCallback(
    (e: React.MouseEvent) => {
      const shouldPan =
        isSpaceDown.current ||
        e.button === 1 ||
        (e.button === 0 && e.target === e.currentTarget);

      if (shouldPan) {
        if (!isSpaceDown.current && e.button === 0) setSelectedId(null);
        setIsPanning(true);
        setPanStart({ x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y });
        e.preventDefault();
      }
    },
    [pan]
  );

  /* ── Right-click rotate on fragment ──────────────────────────────────── */

  const handleFragmentContextMenu = useCallback(
    (e: React.MouseEvent, fragId: number) => {
      e.preventDefault();
      e.stopPropagation();
      setSelectedId(fragId);
      saveUndo();
      setFragments((prev) => {
        const frag = prev.find((f) => f.id === fragId);
        if (frag && !frag.locked) {
          setRotateState({
            fragmentId: fragId,
            startMouseX: e.clientX,
            startRotation: frag.rotation,
          });
        }
        return prev;
      });
    },
    [saveUndo]
  );

  /* ── Rotation handle ─────────────────────────────────────────────────── */

  const handleRotateMouseDown = useCallback(
    (e: React.MouseEvent, fragId: number) => {
      e.stopPropagation();
      e.preventDefault();
      saveUndo();
      setFragments((prev) => {
        const frag = prev.find((f) => f.id === fragId);
        if (frag) {
          setRotateState({
            fragmentId: fragId,
            startMouseX: e.clientX,
            startRotation: frag.rotation,
          });
        }
        return prev;
      });
    },
    [saveUndo]
  );

  /* ── Zoom buttons ────────────────────────────────────────────────────── */

  const handleZoomIn = useCallback(() => {
    setZoom((prev) => clampZoom(prev + ZOOM_BUTTON_STEP));
  }, [clampZoom]);

  const handleZoomOut = useCallback(() => {
    setZoom((prev) => clampZoom(prev - ZOOM_BUTTON_STEP));
  }, [clampZoom]);

  const handleZoomSlider = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setZoom(clampZoom(parseFloat(e.target.value)));
    },
    [clampZoom]
  );

  /* ── Layer operations ────────────────────────────────────────────────── */

  const toggleVisible = useCallback((id: number) => {
    setFragments((prev) =>
      prev.map((f) => (f.id === id ? { ...f, visible: !f.visible } : f))
    );
  }, []);

  const toggleLock = useCallback((id: number) => {
    setFragments((prev) =>
      prev.map((f) => (f.id === id ? { ...f, locked: !f.locked } : f))
    );
  }, []);

  const moveLayerUp = useCallback((id: number) => {
    saveUndo();
    setFragments((prev) => {
      const sorted = [...prev].sort((a, b) => a.zIndex - b.zIndex);
      const idx = sorted.findIndex((f) => f.id === id);
      if (idx < sorted.length - 1) {
        const myZ = sorted[idx].zIndex;
        const aboveZ = sorted[idx + 1].zIndex;
        return prev.map((f) => {
          if (f.id === sorted[idx].id) return { ...f, zIndex: aboveZ };
          if (f.id === sorted[idx + 1].id) return { ...f, zIndex: myZ };
          return f;
        });
      }
      return prev;
    });
  }, [saveUndo]);

  const moveLayerDown = useCallback((id: number) => {
    saveUndo();
    setFragments((prev) => {
      const sorted = [...prev].sort((a, b) => a.zIndex - b.zIndex);
      const idx = sorted.findIndex((f) => f.id === id);
      if (idx > 0) {
        const myZ = sorted[idx].zIndex;
        const belowZ = sorted[idx - 1].zIndex;
        return prev.map((f) => {
          if (f.id === sorted[idx].id) return { ...f, zIndex: belowZ };
          if (f.id === sorted[idx - 1].id) return { ...f, zIndex: myZ };
          return f;
        });
      }
      return prev;
    });
  }, [saveUndo]);

  /* ── Reset ───────────────────────────────────────────────────────────── */

  const handleReset = useCallback(() => {
    saveUndo();
    setFragments(initialFragments.map((f) => ({ ...f })));
    setSelectedId(null);
    topZRef.current = initialFragments.length + 1;
  }, [initialFragments, saveUndo]);

  /* ── Export ──────────────────────────────────────────────────────────── */

  const handleExport = useCallback(async () => {
    if (!boardData || fragments.length === 0) return;
    setExporting(true);
    try {
      const visibleFrags = fragments.filter((f) => f.visible);
      const minX = Math.min(...visibleFrags.map((f) => f.x));
      const minY = Math.min(...visibleFrags.map((f) => f.y));
      const maxX = Math.max(...visibleFrags.map((f) => f.x + f.width));
      const maxY = Math.max(...visibleFrags.map((f) => f.y + f.height));

      const width = Math.ceil(maxX - minX);
      const height = Math.ceil(maxY - minY);

      const placements = visibleFrags.map((f) => ({
        id: f.id,
        x: f.x - minX,
        y: f.y - minY,
        rotation: f.rotation,
      }));

      const clean = !showRaw;
      const blob = await exportBoard(jobId, placements, width, height, exportScale, {
        clean,
        refine: clean && refine,
      });

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const tag = `${exportScale}x${clean ? (refine ? "_refined" : "_clean") : "_raw"}`;
      a.download = `untorn_assembly_${tag}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Export failed:", err);
    } finally {
      setExporting(false);
    }
  }, [fragments, jobId, boardData, exportScale, showRaw, refine]);

  /* ── Sorted fragments for layers panel (highest z on top) ────────────── */

  const layerOrder = useMemo(
    () => [...fragments].sort((a, b) => b.zIndex - a.zIndex),
    [fragments]
  );

  const updateSelected = useCallback((patch: Partial<FragmentState>) => {
    if (selectedId === null) return;
    setFragments((prev) =>
      prev.map((f) => (f.id === selectedId ? { ...f, ...patch } : f))
    );
  }, [selectedId]);

  /* ── Scrollbar helpers ─────────────────────────────────────────────────── */

  const startScrollDrag = useCallback(
    (e: React.MouseEvent, axis: "x" | "y") => {
      e.stopPropagation();
      e.preventDefault();
      const el = containerRef.current;
      if (!el) return;
      const containerW = el.clientWidth;
      const containerH = el.clientHeight;
      const contentW = (boardData?.canvas.width ?? 0) * zoom;
      const contentH = (boardData?.canvas.height ?? 0) * zoom;

      // The scrollbar track maps to the full content range
      // scale = how many content-px per 1 scrollbar-px of drag
      const trackPad = 16; // px padding on each end of the track
      if (axis === "x") {
        const trackLen = containerW - trackPad * 2;
        const totalRange = Math.max(contentW, containerW) + containerW;
        setScrollDrag({
          axis: "x",
          startMouse: e.clientX,
          startPan: pan.x,
          scale: totalRange / trackLen,
        });
      } else {
        const trackLen = containerH - trackPad * 2;
        const totalRange = Math.max(contentH, containerH) + containerH;
        setScrollDrag({
          axis: "y",
          startMouse: e.clientY,
          startPan: pan.y,
          scale: totalRange / trackLen,
        });
      }
    },
    [boardData, zoom, pan]
  );

  // Compute scrollbar thumb position/size for the current pan/zoom
  const scrollbarInfo = useMemo(() => {
    const el = containerRef.current;
    if (!el || !boardData) return null;
    const cW = el.clientWidth;
    const cH = el.clientHeight;
    const contentW = boardData.canvas.width * zoom;
    const contentH = boardData.canvas.height * zoom;

    const computeAxis = (panVal: number, contentSize: number, viewportSize: number) => {
      const rangeStart = Math.min(0, panVal);
      const rangeEnd = Math.max(viewportSize, panVal + contentSize);
      const total = rangeEnd - rangeStart;
      if (total <= 0) return null;
      const thumbStart = (0 - rangeStart) / total;
      const thumbSize = viewportSize / total;
      // Hide if content fits entirely
      if (thumbSize >= 0.98) return null;
      return { thumbStart, thumbSize };
    };

    return {
      h: computeAxis(pan.x, contentW, cW),
      v: computeAxis(pan.y, contentH, cH),
      cW,
      cH,
    };
  }, [boardData, zoom, pan]);

  /* ── Render ──────────────────────────────────────────────────────────── */

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-16 text-danger">
        Error loading board data: {error}
      </div>
    );
  }

  const totalFragments = boardData?.fragments.length ?? 0;
  const allLoaded = imagesLoaded >= totalFragments;
  const selectedFrag = fragments.find((f) => f.id === selectedId) ?? null;
  const placedCount = fragments.filter((f) => f.placed).length;

  return (
    <div className="space-y-3">
      {/* ── Top bar ─────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between flex-wrap gap-2 rounded-2xl border border-border/60 bg-white/80 backdrop-blur px-3 py-2 shadow-card">
        <div className="flex items-center gap-1.5">
          <Button variant="outline" size="sm" onClick={handleReset} title="Reset all positions">
            <RotateCcw size={14} />
            <span className="hidden sm:inline">Сброс</span>
          </Button>
          <Button variant="outline" size="sm" onClick={() => fitToView()} title="Fit to view">
            <Maximize size={14} />
            <span className="hidden sm:inline">Вписать</span>
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={undo}
            disabled={undoCount === 0}
            title="Undo (Ctrl+Z)"
          >
            <Undo2 size={14} />
          </Button>
          <div className="w-px h-5 bg-border mx-0.5" />
          <Button
            variant={showGrid ? "default" : "outline"}
            size="sm"
            onClick={() => setShowGrid((v) => !v)}
            title="Toggle grid"
          >
            <Grid3X3 size={14} />
          </Button>
          <Button
            variant={showLayers ? "default" : "outline"}
            size="sm"
            onClick={() => setShowLayers((v) => !v)}
            title="Toggle layers panel"
          >
            <Layers size={14} />
          </Button>
          <Button
            variant={showInspector ? "default" : "outline"}
            size="sm"
            onClick={() => setShowInspector((v) => !v)}
            title="Toggle inspector"
          >
            <Settings size={14} />
          </Button>
        </div>

        <div className="flex items-center gap-2 text-xs text-secondary">
          <span className="hidden md:inline">{placedCount}/{totalFragments} размещено</span>
          {selectedId !== null && (
            <span className="rounded-full bg-accent/10 text-accent px-2 py-0.5">Выбрано #{selectedId}</span>
          )}
          <span className="rounded-full bg-muted px-2 py-0.5">{Math.round(zoom * 100)}%</span>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-0.5 bg-muted rounded-lg p-0.5">
            {[1, 2, 4].map((s) => (
              <button
                key={s}
                onClick={() => setExportScale(s)}
                className={cn(
                  "px-2 py-1 text-[11px] font-medium rounded-md transition-all",
                  exportScale === s
                    ? "bg-white text-primary shadow-sm"
                    : "text-secondary hover:text-primary"
                )}
                title={`Export at ${s}x resolution`}
              >
                {s}x
              </button>
            ))}
          </div>
          <Button
            variant="default"
            size="sm"
            onClick={handleExport}
            disabled={exporting}
            title={showRaw
              ? "Экспорт без чистки швов"
              : refine ? "Чистка LaMa (refine) — медленно" : "Чистка LaMa"}
          >
            {exporting ? (
              <Loader size={14} className="animate-spin" />
            ) : (
              <Download size={14} />
            )}
            {exporting
              ? (refine && !showRaw ? "Обработка..." : "Экспорт...")
              : (showRaw ? "Скачать (raw)" : refine ? "Скачать (refined)" : "Скачать")}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[260px_minmax(0,1fr)_300px] gap-3">
        {/* ── Layers sidebar ───────────────────────────────────────────── */}
        <aside className={cn("flex flex-col gap-3", showLayers ? "flex" : "hidden", "lg:flex")}
          style={{ minHeight: 240 }}>
          <div className="rounded-2xl border border-border/60 bg-white/90 shadow-card overflow-hidden">
            <div className="flex items-center justify-between px-3 py-2 border-b border-border/50">
              <span className="text-xs font-semibold text-primary flex items-center gap-1.5">
                <Layers size={12} />
                Слои
                <span className="text-secondary font-normal">({totalFragments})</span>
              </span>
              <button
                onClick={() => setShowLayers(false)}
                className="w-6 h-6 flex items-center justify-center rounded-md hover:bg-black/5 text-secondary hover:text-primary transition-colors lg:hidden"
              >
                <X size={12} />
              </button>
            </div>
            <div className="max-h-[calc(100vh-340px)] overflow-y-auto p-2 space-y-1">
              {layerOrder.map((frag) => {
                const isSelected = selectedId === frag.id;
                const thumbSrc = blobUrls.current.get(frag.id);
                return (
                  <div
                    key={frag.id}
                    className={cn(
                      "flex items-center gap-2 px-2 py-1.5 rounded-lg cursor-pointer transition-all text-xs group",
                      isSelected
                        ? "bg-accent/10 ring-1 ring-accent/20"
                        : "hover:bg-black/[0.03]"
                    )}
                    onClick={() => setSelectedId(frag.id)}
                  >
                    <div
                      className={cn(
                        "w-8 h-8 rounded-md overflow-hidden flex-shrink-0 flex items-center justify-center",
                        frag.visible ? "bg-neutral-100" : "bg-neutral-100/50"
                      )}
                    >
                      {thumbSrc && (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={thumbSrc}
                          alt=""
                          className="w-full h-full object-contain"
                          style={{ opacity: frag.visible ? 1 : 0.3 }}
                          draggable={false}
                        />
                      )}
                    </div>
                    <span
                      className={cn(
                        "flex-1 font-medium tabular-nums",
                        isSelected ? "text-accent" : "text-primary",
                        !frag.visible && "text-secondary"
                      )}
                    >
                      Fragment #{frag.id}
                    </span>
                    {isSelected && (
                      <div className="flex flex-col -my-0.5">
                        <button
                          onClick={(e) => { e.stopPropagation(); moveLayerUp(frag.id); }}
                          className="p-0.5 hover:bg-accent/10 rounded text-secondary hover:text-accent transition-colors"
                          title="Move up"
                        >
                          <ChevronUp size={10} />
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); moveLayerDown(frag.id); }}
                          className="p-0.5 hover:bg-accent/10 rounded text-secondary hover:text-accent transition-colors"
                          title="Move down"
                        >
                          <ChevronDown size={10} />
                        </button>
                      </div>
                    )}
                    <button
                      onClick={(e) => { e.stopPropagation(); toggleVisible(frag.id); }}
                      className={cn(
                        "p-1 rounded-md transition-colors",
                        frag.visible
                          ? "text-secondary hover:text-primary hover:bg-black/5"
                          : "text-secondary/40 hover:text-secondary hover:bg-black/5"
                      )}
                      title={frag.visible ? "Hide" : "Show"}
                    >
                      {frag.visible ? <Eye size={12} /> : <EyeOff size={12} />}
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); toggleLock(frag.id); }}
                      className={cn(
                        "p-1 rounded-md transition-colors",
                        frag.locked
                          ? "text-warning hover:text-warning/80 hover:bg-warning/5"
                          : "text-secondary/30 hover:text-secondary hover:bg-black/5"
                      )}
                      title={frag.locked ? "Unlock" : "Lock"}
                    >
                      {frag.locked ? <Lock size={12} /> : <Unlock size={12} />}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        </aside>

        {/* ── Canvas area ─────────────────────────────────────────────── */}
        <div
          ref={containerRef}
          className="relative w-full rounded-2xl overflow-hidden select-none"
          style={{
            height: "calc(100vh - 180px)",
            minHeight: 520,
            cursor: isPanning || scrollDrag
              ? "grabbing"
              : spaceHeld
                ? "grab"
                : dragState
                  ? "grabbing"
                  : "default",
            background: "#1a1a1e",
            backgroundImage: "radial-gradient(circle, rgba(255,255,255,0.03) 1px, transparent 1px)",
            backgroundSize: "24px 24px",
          }}
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onContextMenu={(e) => e.preventDefault()}
        >
        {/* Loading overlay */}
        {!allLoaded && (
          <div className="absolute inset-0 bg-[#1a1a1e]/90 z-50 flex flex-col items-center justify-center gap-3">
            <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-white/60">
              Loading fragments... {imagesLoaded}/{totalFragments}
            </span>
          </div>
        )}

        {/* ── Transformed canvas layer ──────────────────────────────────── */}
        <div
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: "0 0",
            position: "absolute",
            width: boardData?.canvas.width ?? 0,
            height: boardData?.canvas.height ?? 0,
          }}
        >
          {/* Canvas background — white board with optional grid */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              background: "white",
              borderRadius: 2,
              boxShadow: "0 0 0 1px rgba(255,255,255,0.08), 0 8px 40px rgba(0,0,0,0.4)",
              ...(showGrid
                ? {
                    backgroundImage:
                      "radial-gradient(circle, rgba(0,0,0,0.12) 1px, transparent 1px)",
                    backgroundSize: `${GRID_SIZE}px ${GRID_SIZE}px`,
                  }
                : {}),
            }}
          />

          {/* ── Fragments ───────────────────────────────────────────────── */}
          {fragments.map((frag) => {
            if (!frag.visible) return null;
            const isSelected = selectedId === frag.id;
            const imgSrc =
              blobUrls.current.get(frag.id) ??
              boardFragmentUrl(jobId, frag.id);

            return (
              <div
                key={frag.id}
                className="absolute"
                style={{
                  left: frag.x,
                  top: frag.y,
                  width: frag.width,
                  height: frag.height,
                  zIndex: frag.zIndex,
                  transform: `rotate(${frag.rotation}deg)`,
                  transformOrigin: "center center",
                  cursor: frag.locked
                    ? "not-allowed"
                    : dragState?.fragmentId === frag.id
                      ? "grabbing"
                      : "grab",
                  opacity: frag.locked && !isSelected ? 0.7 : 1,
                  transition: dragState?.fragmentId === frag.id ? "none" : "opacity 0.15s ease",
                }}
                onMouseDown={(e) => {
                  if (e.button === 0) handleFragmentMouseDown(e, frag.id);
                }}
                onContextMenu={(e) => handleFragmentContextMenu(e, frag.id)}
              >
                {/* Fragment image */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={imgSrc}
                  alt={`Fragment ${frag.id}`}
                  className="w-full h-full pointer-events-none"
                  style={{ imageRendering: "auto" }}
                  draggable={false}
                />

                {/* Selection ring + controls */}
                {isSelected && (
                  <>
                    {/* Glow outline */}
                    <div
                      className="absolute pointer-events-none"
                      style={{
                        inset: -2,
                        borderRadius: 3,
                        border: "2px solid rgba(79, 70, 229, 0.7)",
                        boxShadow:
                          "0 0 0 1px rgba(79, 70, 229, 0.2), 0 0 16px rgba(79, 70, 229, 0.15), inset 0 0 16px rgba(79, 70, 229, 0.05)",
                      }}
                    />

                    {/* Corner dots */}
                    {[
                      { top: -4, left: -4 },
                      { top: -4, right: -4 },
                      { bottom: -4, left: -4 },
                      { bottom: -4, right: -4 },
                    ].map((pos, i) => (
                      <div
                        key={i}
                        className="absolute w-2 h-2 bg-white border-2 border-accent rounded-full pointer-events-none"
                        style={pos}
                      />
                    ))}

                    {/* Rotation handle — line + circle */}
                    <div
                      className="absolute left-1/2 pointer-events-none"
                      style={{
                        top: -28,
                        width: 1,
                        height: 20,
                        background: "rgba(79, 70, 229, 0.4)",
                        transform: "translateX(-50%)",
                      }}
                    />
                    <div
                      className="absolute left-1/2 flex items-center justify-center cursor-grab active:cursor-grabbing"
                      style={{
                        top: -36,
                        width: 20,
                        height: 20,
                        transform: "translateX(-50%)",
                        background: "#4F46E5",
                        borderRadius: "50%",
                        boxShadow: "0 2px 8px rgba(79, 70, 229, 0.35)",
                        transition: "transform 0.1s ease",
                        zIndex: 9999,
                        pointerEvents: "auto",
                      }}
                      onMouseDown={(e) => handleRotateMouseDown(e, frag.id)}
                      onMouseEnter={(e) => {
                        (e.target as HTMLElement).style.transform = "translateX(-50%) scale(1.15)";
                      }}
                      onMouseLeave={(e) => {
                        (e.target as HTMLElement).style.transform = "translateX(-50%) scale(1)";
                      }}
                      title="Drag to rotate"
                    >
                      <RotateCcw size={10} className="text-white" />
                    </div>

                    {/* Info badge */}
                    <div
                      className="absolute left-1/2 whitespace-nowrap pointer-events-none"
                      style={{
                        bottom: -24,
                        transform: "translateX(-50%)",
                        fontSize: 10,
                        fontWeight: 600,
                        color: "white",
                        background: "rgba(79, 70, 229, 0.85)",
                        borderRadius: 10,
                        padding: "2px 8px",
                        backdropFilter: "blur(4px)",
                        letterSpacing: "0.01em",
                      }}
                    >
                      #{frag.id} · {frag.rotation.toFixed(1)}°
                      {frag.locked && " · Locked"}
                    </div>
                  </>
                )}
              </div>
            );
          })}
        </div>

        {/* ── Layers panel (floating) ───────────────────────────────────── */}
        {showLayers && (
          <div
            className="absolute top-3 right-3 z-50 animate-fade-in"
            style={{ width: 220 }}
          >
            <div
              className="rounded-xl overflow-hidden"
              style={{
                background: "rgba(255,255,255,0.92)",
                backdropFilter: "blur(16px)",
                boxShadow: "0 4px 24px rgba(0,0,0,0.15), 0 0 0 1px rgba(0,0,0,0.06)",
              }}
            >
              {/* Header */}
              <div className="flex items-center justify-between px-3 py-2 border-b border-black/5">
                <span className="text-xs font-semibold text-primary flex items-center gap-1.5">
                  <Layers size={12} />
                  Layers
                  <span className="text-secondary font-normal">({totalFragments})</span>
                </span>
                <button
                  onClick={() => setShowLayers(false)}
                  className="w-5 h-5 flex items-center justify-center rounded-md hover:bg-black/5 text-secondary hover:text-primary transition-colors"
                >
                  <X size={12} />
                </button>
              </div>

              {/* Fragment list */}
              <div className="max-h-72 overflow-y-auto p-1.5 space-y-0.5">
                {layerOrder.map((frag) => {
                  const isSelected = selectedId === frag.id;
                  const thumbSrc = blobUrls.current.get(frag.id);
                  return (
                    <div
                      key={frag.id}
                      className={cn(
                        "flex items-center gap-2 px-2 py-1.5 rounded-lg cursor-pointer transition-all text-xs group",
                        isSelected
                          ? "bg-accent/10 ring-1 ring-accent/20"
                          : "hover:bg-black/[0.03]"
                      )}
                      onClick={() => setSelectedId(frag.id)}
                    >
                      {/* Thumbnail */}
                      <div
                        className={cn(
                          "w-8 h-8 rounded-md overflow-hidden flex-shrink-0 flex items-center justify-center",
                          frag.visible ? "bg-neutral-100" : "bg-neutral-100/50"
                        )}
                      >
                        {thumbSrc && (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img
                            src={thumbSrc}
                            alt=""
                            className="w-full h-full object-contain"
                            style={{ opacity: frag.visible ? 1 : 0.3 }}
                            draggable={false}
                          />
                        )}
                      </div>

                      {/* Label */}
                      <span
                        className={cn(
                          "flex-1 font-medium tabular-nums",
                          isSelected ? "text-accent" : "text-primary",
                          !frag.visible && "text-secondary"
                        )}
                      >
                        Fragment #{frag.id}
                      </span>

                      {/* Reorder buttons (show on selected) */}
                      {isSelected && (
                        <div className="flex flex-col -my-0.5">
                          <button
                            onClick={(e) => { e.stopPropagation(); moveLayerUp(frag.id); }}
                            className="p-0.5 hover:bg-accent/10 rounded text-secondary hover:text-accent transition-colors"
                            title="Move up"
                          >
                            <ChevronUp size={10} />
                          </button>
                          <button
                            onClick={(e) => { e.stopPropagation(); moveLayerDown(frag.id); }}
                            className="p-0.5 hover:bg-accent/10 rounded text-secondary hover:text-accent transition-colors"
                            title="Move down"
                          >
                            <ChevronDown size={10} />
                          </button>
                        </div>
                      )}

                      {/* Visibility */}
                      <button
                        onClick={(e) => { e.stopPropagation(); toggleVisible(frag.id); }}
                        className={cn(
                          "p-1 rounded-md transition-colors",
                          frag.visible
                            ? "text-secondary hover:text-primary hover:bg-black/5"
                            : "text-secondary/40 hover:text-secondary hover:bg-black/5"
                        )}
                        title={frag.visible ? "Hide" : "Show"}
                      >
                        {frag.visible ? <Eye size={12} /> : <EyeOff size={12} />}
                      </button>

                      {/* Lock */}
                      <button
                        onClick={(e) => { e.stopPropagation(); toggleLock(frag.id); }}
                        className={cn(
                          "p-1 rounded-md transition-colors",
                          frag.locked
                            ? "text-warning hover:text-warning/80 hover:bg-warning/5"
                            : "text-secondary/30 hover:text-secondary hover:bg-black/5"
                        )}
                        title={frag.locked ? "Unlock" : "Lock"}
                      >
                        {frag.locked ? <Lock size={12} /> : <Unlock size={12} />}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* ── Zoom controls — bottom center ─────────────────────────────── */}
        <div
          className="absolute bottom-4 left-1/2 -translate-x-1/2 z-40 flex items-center gap-2 px-3 py-2 rounded-xl"
          style={{
            background: "rgba(255,255,255,0.92)",
            backdropFilter: "blur(12px)",
            boxShadow: "0 2px 12px rgba(0,0,0,0.1), 0 0 0 1px rgba(0,0,0,0.05)",
          }}
        >
          <button
            onClick={handleZoomOut}
            className="w-7 h-7 flex items-center justify-center rounded-lg hover:bg-black/5 text-secondary hover:text-primary transition-colors"
            title="Zoom out"
          >
            <ZoomOut size={15} />
          </button>
          <input
            type="range"
            min={MIN_ZOOM}
            max={MAX_ZOOM}
            step={0.01}
            value={zoom}
            onChange={handleZoomSlider}
            className="w-28 h-1 accent-accent cursor-pointer"
            title={`${Math.round(zoom * 100)}%`}
          />
          <button
            onClick={handleZoomIn}
            className="w-7 h-7 flex items-center justify-center rounded-lg hover:bg-black/5 text-secondary hover:text-primary transition-colors"
            title="Zoom in"
          >
            <ZoomIn size={15} />
          </button>
          <div className="w-px h-4 bg-black/10 mx-0.5" />
          <span className="text-[11px] font-medium text-secondary tabular-nums w-10 text-center">
            {Math.round(zoom * 100)}%
          </span>
        </div>

        {/* ── Canvas size badge — bottom left ───────────────────────────── */}
        <div
          className="absolute bottom-4 left-4 text-[10px] text-white/30 z-40 pointer-events-none"
          style={{
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {boardData?.canvas.width} × {boardData?.canvas.height} px
        </div>

        {/* ── Scrollbars ────────────────────────────────────────────────── */}

        {/* Horizontal scrollbar — bottom edge */}
        {scrollbarInfo?.h && (
          <div
            className="absolute z-40 opacity-0 hover:opacity-100 transition-opacity duration-200"
            style={{
              left: 16,
              right: 16,
              bottom: 36,
              height: 10, // thicker
            }}
            onMouseDown={(e) => startScrollDrag(e, "x")}
          >
            {/* Track */}
            <div
              className="absolute inset-0 rounded-full"
              style={{
                background: "rgba(59,130,246,0.15)",
              }}
            />

            {/* Thumb */}
            <div
              className="absolute top-1 rounded-full transition-all duration-150"
              style={{
                left: `${scrollbarInfo.h.thumbStart * 100}%`,
                width: `${Math.max(scrollbarInfo.h.thumbSize * 100, 4)}%`,
                height: 6,
                background:
                  scrollDrag?.axis === "x"
                    ? "#3B82F6"
                    : "rgba(59,130,246,0.7)",
                boxShadow: "0 0 8px rgba(59,130,246,0.9)",
                cursor: "grab",
              }}
            />
          </div>
        )}

        {/* Vertical scrollbar — right edge */}
        {scrollbarInfo?.v && (
          <div
            className="absolute z-40 opacity-0 hover:opacity-100 transition-opacity duration-200"
            style={{
              top: 16,
              bottom: 16,
              right: 4,
              width: 10, // thicker
            }}
            onMouseDown={(e) => startScrollDrag(e, "y")}
          >
            {/* Track */}
            <div
              className="absolute inset-0 rounded-full"
              style={{
                background: "rgba(59,130,246,0.15)",
              }}
            />

            {/* Thumb */}
            <div
              className="absolute left-1 rounded-full transition-all duration-150"
              style={{
                top: `${scrollbarInfo.v.thumbStart * 100}%`,
                height: `${Math.max(scrollbarInfo.v.thumbSize * 100, 4)}%`,
                width: 6,
                background:
                  scrollDrag?.axis === "y"
                    ? "#3B82F6"
                    : "rgba(59,130,246,0.7)",
                boxShadow: "0 0 8px rgba(59,130,246,0.9)",
                cursor: "grab",
              }}
            />
          </div>
        )}
      </div>

        {/* ── Inspector sidebar ────────────────────────────────────────── */}
        <aside className={cn("flex flex-col gap-3", showInspector ? "flex" : "hidden", "lg:flex")}
          style={{ minHeight: 240 }}>
          <div className="rounded-2xl border border-border/60 bg-white/90 shadow-card overflow-hidden">
            <div className="flex items-center justify-between px-3 py-2 border-b border-border/50">
              <span className="text-xs font-semibold text-primary">Инспектор</span>
              <button
                onClick={() => setShowInspector(false)}
                className="w-6 h-6 flex items-center justify-center rounded-md hover:bg-black/5 text-secondary hover:text-primary transition-colors lg:hidden"
              >
                <X size={12} />
              </button>
            </div>
            <div className="p-3 space-y-3">
              {selectedFrag ? (
                <>
                  <div className="text-xs text-secondary">Фрагмент #{selectedFrag.id}</div>
                  <div className="grid grid-cols-2 gap-2">
                    <label className="text-xs text-secondary">
                      X
                      <input
                        type="number"
                        className="mt-1 w-full rounded-md border border-border/60 px-2 py-1 text-xs text-primary"
                        value={selectedFrag.x.toFixed(1)}
                        onChange={(e) => updateSelected({ x: parseFloat(e.target.value) || 0 })}
                      />
                    </label>
                    <label className="text-xs text-secondary">
                      Y
                      <input
                        type="number"
                        className="mt-1 w-full rounded-md border border-border/60 px-2 py-1 text-xs text-primary"
                        value={selectedFrag.y.toFixed(1)}
                        onChange={(e) => updateSelected({ y: parseFloat(e.target.value) || 0 })}
                      />
                    </label>
                    <label className="text-xs text-secondary">
                      Rotation
                      <input
                        type="number"
                        className="mt-1 w-full rounded-md border border-border/60 px-2 py-1 text-xs text-primary"
                        value={selectedFrag.rotation.toFixed(1)}
                        onChange={(e) => updateSelected({ rotation: parseFloat(e.target.value) || 0 })}
                      />
                    </label>
                    <div className="text-xs text-secondary">
                      Size
                      <div className="mt-1 rounded-md border border-border/60 px-2 py-1 text-xs text-primary">
                        {selectedFrag.width}×{selectedFrag.height}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => toggleVisible(selectedFrag.id)}
                    >
                      {selectedFrag.visible ? <Eye size={14} /> : <EyeOff size={14} />}
                      {selectedFrag.visible ? "Скрыть" : "Показать"}
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => toggleLock(selectedFrag.id)}
                    >
                      {selectedFrag.locked ? <Lock size={14} /> : <Unlock size={14} />}
                      {selectedFrag.locked ? "Разблок." : "Блокировать"}
                    </Button>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm" onClick={() => moveLayerUp(selectedFrag.id)}>
                      <ChevronUp size={14} />
                      Вверх
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => moveLayerDown(selectedFrag.id)}>
                      <ChevronDown size={14} />
                      Вниз
                    </Button>
                  </div>
                </>
              ) : (
                <div className="text-xs text-secondary">Выберите фрагмент на холсте.</div>
              )}
            </div>
          </div>

          <div className="rounded-2xl border border-border/60 bg-white/90 shadow-card overflow-hidden">
            <div className="flex items-center justify-between px-3 py-2 border-b border-border/50">
              <span className="text-xs font-semibold text-primary">Экспорт</span>
              <button
                onClick={() => setShowAdvanced((v) => !v)}
                className="w-6 h-6 flex items-center justify-center rounded-md hover:bg-black/5 text-secondary hover:text-primary transition-colors"
                title="Дополнительные параметры"
              >
                {showAdvanced ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              </button>
            </div>
            <div className="p-3 space-y-3">
              <div className="flex items-center gap-1 bg-muted rounded-lg p-0.5 w-fit">
                {[1, 2, 4].map((s) => (
                  <button
                    key={s}
                    onClick={() => setExportScale(s)}
                    className={cn(
                      "px-2 py-1 text-[11px] font-medium rounded-md transition-all",
                      exportScale === s
                        ? "bg-white text-primary shadow-sm"
                        : "text-secondary hover:text-primary"
                    )}
                  >
                    {s}x
                  </button>
                ))}
              </div>
              <label className="flex items-start gap-2 text-xs text-secondary">
                <input
                  type="checkbox"
                  className="mt-0.5"
                  checked={showRaw}
                  onChange={(e) => setShowRaw(e.target.checked)}
                />
                Экспорт без LaMa (сырой результат)
              </label>
              {showAdvanced && (
                <label className={cn(
                  "flex items-start gap-2 text-xs text-secondary",
                  showRaw && "opacity-40 pointer-events-none"
                )}>
                  <input
                    type="checkbox"
                    className="mt-0.5"
                    checked={refine}
                    onChange={(e) => setRefine(e.target.checked)}
                    disabled={showRaw}
                  />
                  Refinement LaMa (высокое качество, медленно)
                </label>
              )}
              <Button
                variant="default"
                size="sm"
                onClick={handleExport}
                disabled={exporting}
                className="w-full"
              >
                {exporting ? (
                  <Loader size={14} className="animate-spin" />
                ) : (
                  <Download size={14} />
                )}
                {exporting ? "Экспорт..." : "Скачать"}
              </Button>
            </div>
          </div>

          <div className="rounded-2xl border border-border/60 bg-white/90 shadow-card p-3 text-xs text-secondary space-y-2">
            <div className="font-semibold text-primary">Подсказки</div>
            <div>Перетащите фрагмент — перемещение.</div>
            <div>ПКМ и перетаскивание — поворот.</div>
            <div>Колёсико — масштаб, Space — панорама.</div>
          </div>
        </aside>
      </div>

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between text-xs text-secondary">
        <span className="flex items-center gap-1.5">
          <Move size={11} />
          {totalFragments} fragments
          {selectedId !== null && (
            <span className="text-accent font-medium"> · #{selectedId} selected</span>
          )}
        </span>
        <span className="flex items-center gap-3 text-[11px]">
          <span>
            <kbd className="px-1 py-0.5 rounded bg-muted text-[10px] font-mono">Esc</kbd> deselect
          </span>
          <span>
            <kbd className="px-1 py-0.5 rounded bg-muted text-[10px] font-mono">Arrows</kbd> nudge
          </span>
          <span>
            <kbd className="px-1 py-0.5 rounded bg-muted text-[10px] font-mono">[ ]</kbd> rotate
          </span>
          <span>
            <kbd className="px-1 py-0.5 rounded bg-muted text-[10px] font-mono">Ctrl+Z</kbd> undo
          </span>
          <span>
            <kbd className="px-1 py-0.5 rounded bg-muted text-[10px] font-mono">Space</kbd> pan
          </span>
        </span>
      </div>
    </div>
  );
}
