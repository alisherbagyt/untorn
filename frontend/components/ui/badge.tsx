import * as React from "react";
import { cn } from "@/lib/utils";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "success" | "warning" | "danger" | "info" | "muted";
}

export const Badge = ({ className, variant = "default", children, ...props }: BadgeProps) => {
  const variants = {
    default: "bg-primary/10 text-primary",
    success: "bg-success/10 text-success",
    warning: "bg-warning/10 text-warning",
    danger:  "bg-danger/10 text-danger",
    info:    "bg-accent/10 text-accent",
    muted:   "bg-muted text-secondary",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full",
        variants[variant],
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
};
