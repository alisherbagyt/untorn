import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#F7F7F5",
        surface:    "#FFFFFF",
        border:     "#E8E8E3",
        muted:      "#F3F3F0",
        primary:    "#1C1C1E",
        secondary:  "#6B6B71",
        accent:     "#4F46E5",
        "accent-light": "#EEF2FF",
        success:    "#10B981",
        warning:    "#F59E0B",
        danger:     "#EF4444",
      },
      borderRadius: {
        "2xl": "1rem",
        "3xl": "1.5rem",
        "4xl": "2rem",
      },
      boxShadow: {
        card:   "0 1px 3px 0 rgb(0 0 0 / 0.06), 0 1px 2px -1px rgb(0 0 0 / 0.04)",
        "card-hover": "0 4px 16px 0 rgb(0 0 0 / 0.08), 0 2px 4px -2px rgb(0 0 0 / 0.04)",
        soft:   "0 2px 8px 0 rgb(0 0 0 / 0.06)",
      },
      fontFamily: {
        sans: [
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
      },
      animation: {
        "pulse-dot": "pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-in":   "fadeIn 0.3s ease-out",
        "slide-up":  "slideUp 0.3s ease-out",
      },
      keyframes: {
        fadeIn:  { from: { opacity: "0" }, to: { opacity: "1" } },
        slideUp: { from: { opacity: "0", transform: "translateY(8px)" }, to: { opacity: "1", transform: "translateY(0)" } },
      },
    },
  },
  plugins: [],
};

export default config;
