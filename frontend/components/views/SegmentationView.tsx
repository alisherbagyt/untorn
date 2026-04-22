"use client";

import React, { useState } from "react";
import { Card, CardHeader, CardTitle, CardBody } from "@/components/ui/card";
import { FragmentTimeline } from "@/components/FragmentTimeline";
import { Badge } from "@/components/ui/badge";
import { debugImageUrl } from "@/lib/api";
import type { DebugData } from "@/lib/api";

interface SegmentationViewProps {
  debug: DebugData;
}

export function SegmentationView({ debug }: SegmentationViewProps) {
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const paths = debug.paths;
  const fragments = debug.fragments ?? [];
  const selectedFrag = fragments.find((f) => f.id === selectedId);
  const selectedIdx  = fragments.findIndex((f) => f.id === selectedId);

  return (
    <div className="space-y-4">
      {/* SAM overlay */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Наложение сегментации SAM 2.1</CardTitle>
            <Badge variant="info">{fragments.length} фрагментов</Badge>
          </div>
        </CardHeader>
        <CardBody>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {paths?.sam_overlay && (
              <div>
                <p className="text-xs text-secondary mb-1.5">Необработанные маски SAM</p>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={debugImageUrl(paths.sam_overlay)}
                  alt="SAM overlay"
                  className="w-full rounded-xl object-contain bg-muted"
                />
              </div>
            )}
            {paths?.segmentation_overlay && (
              <div>
                <p className="text-xs text-secondary mb-1.5">Финальное наложение фрагментов</p>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={debugImageUrl(paths.segmentation_overlay)}
                  alt="Fragments overlay"
                  className="w-full rounded-xl object-contain bg-muted"
                />
              </div>
            )}
          </div>
        </CardBody>
      </Card>

      {/* Fragment timeline */}
      <Card>
        <CardHeader>
          <CardTitle>Просмотр фрагментов</CardTitle>
        </CardHeader>
        <CardBody>
          <FragmentTimeline
            debug={debug}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        </CardBody>
      </Card>

      {/* Selected fragment detail */}
      {selectedFrag && selectedIdx >= 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <CardTitle>Фрагмент #{selectedFrag.id} — Детали</CardTitle>
              <Badge variant="info">
                {(selectedFrag.area / 1000).toFixed(1)}k px²
              </Badge>
            </div>
          </CardHeader>
          <CardBody>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {/* Crop */}
              {paths?.fragment_crops?.[selectedIdx] && (
                <div>
                  <p className="text-xs text-secondary mb-1.5">Обрезка</p>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={debugImageUrl(paths.fragment_crops[selectedIdx])}
                    alt={`Crop ${selectedFrag.id}`}
                    className="w-full rounded-xl object-contain bg-muted"
                  />
                </div>
              )}
              {/* Mask */}
              {paths?.fragment_masks?.[selectedIdx] && (
                <div>
                  <p className="text-xs text-secondary mb-1.5">Маска</p>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={debugImageUrl(paths.fragment_masks[selectedIdx])}
                    alt={`Mask ${selectedFrag.id}`}
                    className="w-full rounded-xl object-contain bg-muted"
                  />
                </div>
              )}
              {/* SDF */}
              {paths?.fragment_sdfs?.[selectedIdx] && (
                <div>
                  <p className="text-xs text-secondary mb-1.5">SDF</p>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={debugImageUrl(paths.fragment_sdfs[selectedIdx])}
                    alt={`SDF ${selectedFrag.id}`}
                    className="w-full rounded-xl object-contain bg-muted"
                  />
                </div>
              )}
              {/* Support points */}
              {paths?.fragment_support?.[selectedIdx] && (
                <div>
                  <p className="text-xs text-secondary mb-1.5">Опорные точки</p>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={debugImageUrl(paths.fragment_support[selectedIdx])}
                    alt={`Support ${selectedFrag.id}`}
                    className="w-full rounded-xl object-contain bg-muted"
                  />
                </div>
              )}
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-3 gap-3 mt-4">
              {[
                { label: "Площадь",                 value: `${selectedFrag.area.toLocaleString()} px²` },
                { label: "Центроид",                value: `(${selectedFrag.centroid[0].toFixed(0)}, ${selectedFrag.centroid[1].toFixed(0)})` },
                { label: "Габаритный прямоугольник",value: `${selectedFrag.bbox_xywh[2]}×${selectedFrag.bbox_xywh[3]}` },
              ].map((s) => (
                <div key={s.label} className="bg-muted rounded-xl p-3">
                  <div className="text-xs text-secondary">{s.label}</div>
                  <div className="text-sm font-semibold text-primary mt-0.5 font-mono">{s.value}</div>
                </div>
              ))}
            </div>
          </CardBody>
        </Card>
      )}
    </div>
  );
}
