import { createRoot } from "react-dom/client";
import App from "./App.jsx";

// Intentionally no StrictMode: its dev-only double-mount would open and tear
// down two WebTransport sessions on load. The live session is the point here.
createRoot(document.getElementById("root")).render(<App />);
