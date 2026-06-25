import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The Python UI server (web_stream.py) hosts the built bundle and synthesizes
// /config.json. In dev, Vite serves the app and proxies that one endpoint to the
// running Python server; the WebTransport/QUIC connection is opened directly to
// the port advertised in the config, so it does not pass through this proxy.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    proxy: {
      "/config.json": "http://127.0.0.1:8000",
    },
  },
});
