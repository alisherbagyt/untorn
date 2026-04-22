import * as React from "react";
import { cn } from "@/lib/utils";

interface ProgressProps {
  value: number;  // 0-100
  className?: string;
  barClassName?: string;
  animated?: boolean;
}

export const Progress = ({ value, className, barClassName, animated = true }: ProgressProps) => {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <div className={cn("w-full bg-muted rounded-full overflow-hidden", className)} style={{ height: "6px" }}>
      <div
        className={cn(
          "h-full rounded-full transition-all duration-500 ease-out",
          animated && "bg-gradient-to-r from-accent to-indigo-400",
          !animated && "bg-accent",
          barClassName
        )}
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
};
