import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        truth: { 400: "#22d3ee", 600: "#0891b2" },
        deepfake: { 500: "#a855f7", 600: "#9333ea" }
      }
    }
  },
  plugins: []
} satisfies Config;
