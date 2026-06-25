import { useRef } from "react";
import { useStream } from "./hooks/useStream.js";
import { Viewport } from "./components/Viewport.jsx";
import { ControlPanel } from "./components/ControlPanel.jsx";
import "./theme.css";

export default function App() {
  const canvasRef = useRef(null);
  const { status, controls, send } = useStream(canvasRef);

  return (
    <div className="app">
      <header className="app__header">BalaGAN</header>
      <Viewport canvasRef={canvasRef} status={status} controls={controls} send={send} />
      <ControlPanel controls={controls} send={send} />
    </div>
  );
}
