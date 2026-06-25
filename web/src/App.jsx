import { useRef } from "react";
import { useStream } from "./hooks/useStream.js";
import { Viewport } from "./components/Viewport.jsx";
import { ControlPanel } from "./components/ControlPanel.jsx";
import "./theme.css";

export default function App() {
  const canvasRef = useRef(null);
  const { status, engineStatus, controls, send } = useStream(canvasRef);

  return (
    <div className="app">
      <aside className="sidebar">
        <header className="app__header">BalaGAN</header>
        <ControlPanel
          controls={controls}
          send={send}
          status={status}
          engineStatus={engineStatus}
        />
      </aside>
      <main className="stage">
        <Viewport canvasRef={canvasRef} controls={controls} send={send} />
      </main>
    </div>
  );
}
