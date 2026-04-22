"use client";

import React, { useState } from "react";
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { Card, CardHeader, CardTitle, CardBody } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { debugImageUrl } from "@/lib/api";
import type { DebugData } from "@/lib/api";

interface ContoursViewProps {
  debug: DebugData;
}

export function ContoursView({ debug }: ContoursViewProps) {
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const contours = debug.contours ?? [];
  const fragments = debug.fragments ?? [];
  const paths = debug.paths;

  const selectedContour = contours.find((c) => c.id === selectedId);
  const selectedIdx     = contours.findIndex((c) => c.id === selectedId);

  // Chart data: perimeters per fragment
  const perimeterData = contours.map((c) => ({
    name: `#${c.id}`,
    perimeter: Math.round(c.total_perimeter),
    segments: c.n_edge_segments,
  }));

  // Radar chart data for selected fragment
  const selectedFrag = fragments.find((f) => f.id === selectedId);
  const radarData = selectedContour
    ? [
        { subject: "Опор. точки",  value: selectedContour.n_support_points, max: 30 },
        { subject: "Сегм. краёв",  value: selectedContour.n_edge_segments,  max: 30 },
        { subject: "Периметр",     value: Math.round(selectedContour.total_perimeter / 50), max: 30 },
        { subject: "Гран. пикс.",  value: Math.round(selectedContour.n_boundary_pixels / 100), max: 30 },
      ]
    : [];

  return (
    <div className="space-y-4">
      {/* Support points overview */}
      <Card>
        <CardHeader>
          <CardTitle>Наложение всех опорных точек</CardTitle>
        </CardHeader>
        <CardBody>
          {paths?.contours_overlay ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={debugImageUrl(paths.contours_overlay)}
              alt="All support points"
              className="w-full rounded-xl object-contain bg-muted max-h-[360px]"
            />
          ) : (
            <div className="h-48 bg-muted rounded-xl flex items-center justify-center text-secondary text-sm">
              Нет наложения контуров
            </div>
          )}
        </CardBody>
      </Card>

      {/* Perimeter bar chart */}
      <Card>
        <CardHeader>
          <CardTitle>Периметры фрагментов</CardTitle>
        </CardHeader>
        <CardBody>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={perimeterData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E8E8E3" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#6B6B71" }} />
              <YAxis tick={{ fontSize: 11, fill: "#6B6B71" }} width={40} />
              <Tooltip
                contentStyle={{ borderRadius: 12, border: "1px solid #E8E8E3", fontSize: 12 }}
                cursor={{ fill: "#F3F3F0" }}
              />
              <Bar dataKey="perimeter" name="Периметр (пк)" fill="#4F46E5" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </CardBody>
      </Card>

      {/* Fragment grid */}
      <Card>
        <CardHeader>
          <CardTitle>Выбрать фрагмент</CardTitle>
        </CardHeader>
        <CardBody>
          <div className="grid grid-cols-4 md:grid-cols-6 gap-2">
            {contours.map((c, idx) => (
              <button
                key={c.id}
                onClick={() => setSelectedId(selectedId === c.id ? null : c.id)}
                className={`rounded-xl border-2 p-2 text-xs transition-all ${
                  selectedId === c.id
                    ? "border-accent bg-accent-light"
                    : "border-border hover:border-accent/40 bg-surface"
                }`}
              >
                <div className="font-semibold text-primary">#{c.id}</div>
                <div className="text-secondary mt-0.5">{c.n_support_points} тч</div>
              </button>
            ))}
          </div>
        </CardBody>
      </Card>

      {/* Selected contour detail */}
      {selectedContour && selectedIdx >= 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <CardTitle>Фрагмент #{selectedContour.id} — Детали контура</CardTitle>
              <Badge variant="info">{selectedContour.n_support_points} опорных точек</Badge>
            </div>
          </CardHeader>
          <CardBody>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Radar chart */}
              <div>
                <p className="text-xs text-secondary mb-2">Профиль контура</p>
                <ResponsiveContainer width="100%" height={200}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#E8E8E3" />
                    <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: "#6B6B71" }} />
                    <Radar
                      name="Fragment"
                      dataKey="value"
                      stroke="#4F46E5"
                      fill="#4F46E5"
                      fillOpacity={0.15}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              {/* Images */}
              <div className="space-y-3">
                {paths?.fragment_sdfs?.[selectedIdx] && (
                  <div>
                    <p className="text-xs text-secondary mb-1">Визуализация SDF</p>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={debugImageUrl(paths.fragment_sdfs[selectedIdx])}
                      alt={`SDF ${selectedContour.id}`}
                      className="w-full rounded-xl object-contain bg-muted"
                    />
                  </div>
                )}
                {paths?.fragment_support?.[selectedIdx] && (
                  <div>
                    <p className="text-xs text-secondary mb-1">Опорные точки</p>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={debugImageUrl(paths.fragment_support[selectedIdx])}
                      alt={`Support ${selectedContour.id}`}
                      className="w-full rounded-xl object-contain bg-muted"
                    />
                  </div>
                )}
              </div>
            </div>

            {/* Edge lengths chart */}
            {selectedContour.edge_lengths.length > 0 && (
              <div className="mt-4">
                <p className="text-xs text-secondary mb-2">Длины сегментов краёв</p>
                <ResponsiveContainer width="100%" height={120}>
                  <BarChart
                    data={selectedContour.edge_lengths.map((l, i) => ({ seg: `S${i}`, length: Math.round(l) }))}
                    margin={{ top: 4, right: 8, left: 0, bottom: 4 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#E8E8E3" vertical={false} />
                    <XAxis dataKey="seg" tick={{ fontSize: 10, fill: "#6B6B71" }} />
                    <YAxis tick={{ fontSize: 10, fill: "#6B6B71" }} width={32} />
                    <Tooltip
                      contentStyle={{ borderRadius: 12, border: "1px solid #E8E8E3", fontSize: 11 }}
                    />
                    <Bar dataKey="length" name="Длина (пк)" fill="#8B5CF6" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Stats */}
            <div className="grid grid-cols-4 gap-2 mt-4">
              {[
                { label: "Опор. точки",  value: selectedContour.n_support_points },
                { label: "Сегм. краёв",  value: selectedContour.n_edge_segments },
                { label: "Гран. пикс.",  value: selectedContour.n_boundary_pixels.toLocaleString() },
                { label: "Периметр",     value: `${selectedContour.total_perimeter.toFixed(0)}px` },
              ].map((s) => (
                <div key={s.label} className="bg-muted rounded-xl p-3 text-center">
                  <div className="text-xs text-secondary">{s.label}</div>
                  <div className="text-sm font-bold text-primary mt-0.5">{String(s.value)}</div>
                </div>
              ))}
            </div>
          </CardBody>
        </Card>
      )}
    </div>
  );
}
