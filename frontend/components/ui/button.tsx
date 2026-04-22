import * as React from "react";
import { cn } from "@/lib/utils";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "md", children, disabled, ...props }, ref) => {
    const base = "inline-flex items-center justify-center gap-2 font-medium rounded-xl transition-all duration-150 select-none focus:outline-none focus:ring-2 focus:ring-accent/40 disabled:opacity-50 disabled:pointer-events-none";

    const variants = {
      default: "bg-primary text-white hover:bg-primary/90 active:scale-[0.98]",
      outline: "border border-border bg-surface text-primary hover:bg-muted active:scale-[0.98]",
      ghost:   "text-primary hover:bg-muted active:scale-[0.98]",
      danger:  "bg-danger text-white hover:bg-danger/90 active:scale-[0.98]",
    };

    const sizes = {
      sm: "text-xs px-3 py-1.5",
      md: "text-sm px-4 py-2",
      lg: "text-base px-6 py-3",
    };

    return (
      <button
        ref={ref}
        className={cn(base, variants[variant], sizes[size], className)}
        disabled={disabled}
        {...props}
      >
        {children}
      </button>
    );
  }
);
Button.displayName = "Button";
