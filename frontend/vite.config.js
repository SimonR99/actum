import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/settings": "http://localhost:8000",
      "/state": "http://localhost:8000",
      "/capabilities": "http://localhost:8000",
      "/trigger": "http://localhost:8000",
      "/cron": "http://localhost:8000",
      "/command": "http://localhost:8000",
      "/events": {
        target: "ws://localhost:8000",
        ws: true
      }
    }
  }
});
