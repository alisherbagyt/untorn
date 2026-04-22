"use client";

import React, { useCallback, useRef, useState } from "react";
import { Upload, FileImage, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface UploadZoneProps {
  onUpload: (file: File) => Promise<void>;
  disabled?: boolean;
}

const ACCEPTED_EXTENSIONS = [".tif", ".tiff", ".jpg", ".jpeg", ".png"];

export function UploadZone({ onUpload, disabled }: UploadZoneProps) {
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
      if (!ACCEPTED_EXTENSIONS.includes(ext)) {
        setError("Неподдерживаемый тип файла. Загрузите изображение в формате TIFF, JPG или PNG.");
        return;
      }
      setUploading(true);
      try {
        await onUpload(file);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Ошибка загрузки");
      } finally {
        setUploading(false);
      }
    },
    [onUpload]
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
      e.target.value = "";
    },
    [handleFile]
  );

  return (
    <div className="w-full flex flex-col items-center gap-5">
      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        accept=".tif,.tiff,.jpg,.jpeg,.png"
        className="hidden"
        onChange={handleInputChange}
        disabled={disabled || uploading}
      />

      {/* Upload button */}
      <Button
        variant="default"
        size="lg"
        disabled={disabled || uploading}
        onClick={() => inputRef.current?.click()}
        className="gap-2.5"
      >
        {uploading ? (
          <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
        ) : (
          <FileImage size={18} />
        )}
        {uploading ? "Загрузка..." : "Выбрать изображение"}
      </Button>

      {/* Hint */}
      <p className="text-sm text-secondary">
        <Upload size={13} className="inline mr-1.5 -mt-0.5" />
        Или перетащите файл в любое место на странице
      </p>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 text-sm text-danger bg-red-50 rounded-xl px-4 py-3 w-full max-w-2xl">
          <AlertCircle size={16} />
          {error}
        </div>
      )}
    </div>
  );
}
