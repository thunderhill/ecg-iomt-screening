import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API paths are proxied to the FastAPI server in development.
const API = "http://127.0.0.1:8000";
const apiPaths = ["/health", "/status", "/live", "/ecg", "/metrics", "/screening", "/explain",
  "/recordings", "/models", "/evaluation", "/configuration", "/auth", "/devices"];

export default defineConfig({
  base: "/ui/",
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: Object.fromEntries(apiPaths.map((p) => [p, API])),
  },
});
