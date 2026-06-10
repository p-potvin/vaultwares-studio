// VaultWares Studio splat viewport — GaussianSplats3D inside QWebEngineView.
// The Python side talks to this file via QWebChannel ("bridge" object) and
// window.* functions invoked through runJavaScript().

import * as GaussianSplats3D from './vendor/gaussian-splats-3d.module.js';

const statusEl = document.getElementById('status');
let bridge = null;
let viewer = null;

function setStatus(message) {
  statusEl.textContent = message;
  if (bridge) bridge.jsLog(message);
}

function initChannel() {
  return new Promise((resolve) => {
    if (typeof qt === 'undefined' || !qt.webChannelTransport) {
      resolve(); // standalone browser debugging without Qt
      return;
    }
    new QWebChannel(qt.webChannelTransport, (channel) => {
      bridge = channel.objects.bridge;
      resolve();
    });
  });
}

// Fly the viewer camera along sampled path frames ({position, lookAt} pairs).
// Any user input cancels playback.
let pathPlayback = null;
window.playPath = function playPath(framesJson, fps) {
  if (!viewer) return;
  const frames = typeof framesJson === 'string' ? JSON.parse(framesJson) : framesJson;
  if (!frames.length) return;
  if (pathPlayback) cancelAnimationFrame(pathPlayback.handle);
  const frameMs = 1000 / (fps || 30);
  const startedAt = performance.now();
  const playback = { handle: 0, cancelled: false };
  pathPlayback = playback;
  const cancel = () => { playback.cancelled = true; setStatus('Path preview stopped.'); };
  window.addEventListener('pointerdown', cancel, { once: true });
  window.addEventListener('keydown', cancel, { once: true });
  setStatus(`Previewing path (${(frames.length / (fps || 30)).toFixed(1)}s)…`);

  function step(now) {
    if (playback.cancelled) return;
    const index = Math.min(frames.length - 1, Math.floor((now - startedAt) / frameMs));
    const frame = frames[index];
    viewer.camera.position.set(frame.position[0], frame.position[1], frame.position[2]);
    if (viewer.controls) {
      viewer.controls.target.set(frame.lookAt[0], frame.lookAt[1], frame.lookAt[2]);
      viewer.controls.update();
    } else {
      viewer.camera.lookAt(frame.lookAt[0], frame.lookAt[1], frame.lookAt[2]);
    }
    if (index < frames.length - 1) {
      playback.handle = requestAnimationFrame(step);
    } else {
      setStatus('Path preview finished.');
    }
  }
  playback.handle = requestAnimationFrame(step);
};

// Called from Python (or the toolbar button) to capture the current view as
// a camera pose. Position + lookAt in scene coordinates.
window.captureCamera = function captureCamera() {
  if (!viewer) return;
  const camera = viewer.camera;
  const target = viewer.controls ? viewer.controls.target : { x: 0, y: 0, z: 0 };
  const pose = {
    position: [camera.position.x, camera.position.y, camera.position.z],
    lookAt: [target.x, target.y, target.z],
    up: [camera.up.x, camera.up.y, camera.up.z],
    fovDegrees: camera.fov,
  };
  if (bridge) bridge.cameraCaptured(JSON.stringify(pose));
  setStatus('Camera captured.');
};

async function main() {
  await initChannel();
  const params = new URLSearchParams(window.location.search);
  const scene = params.get('scene');
  if (!scene) {
    setStatus('No scene loaded — run a reconstruction, then reload.');
    if (bridge) bridge.viewerReady();
    return;
  }

  // Scene framing computed Python-side from the preview cloud.
  const center = ['cx', 'cy', 'cz'].map((k) => parseFloat(params.get(k)));
  const radius = parseFloat(params.get('r'));
  const framed = center.every(Number.isFinite) && Number.isFinite(radius);
  let lookAt = framed ? center : [0, 0, 0];
  let position = framed
    ? [center[0], center[1] + radius * 0.4, center[2] + radius * 1.2]
    : [0, 1.5, 4];
  // Explicit pose (camera authoring / screenshot validation) overrides framing.
  const explicitPos = ['px', 'py', 'pz'].map((k) => parseFloat(params.get(k)));
  const explicitLook = ['lx', 'ly', 'lz'].map((k) => parseFloat(params.get(k)));
  if (explicitPos.every(Number.isFinite) && explicitLook.every(Number.isFinite)) {
    position = explicitPos;
    lookAt = explicitLook;
  }

  viewer = new GaussianSplats3D.Viewer({
    cameraUp: [0, 1, 0],
    initialCameraPosition: position,
    initialCameraLookAt: lookAt,
    // No COOP/COEP headers on the custom scheme, so SharedArrayBuffer is
    // unavailable — sort on the main worker without shared memory.
    sharedMemoryForWorkers: false,
    selfDrivenMode: true,
  });

  setStatus('Loading splat scene…');
  try {
    await viewer.addSplatScene(scene, {
      format: GaussianSplats3D.SceneFormat.Ply,
      progressiveLoad: true,
      showLoadingUI: true,
      onProgress: (pct, label, stage) => {
        if (pct !== undefined && pct !== null) setStatus(`Loading splats: ${Math.round(pct)}%`);
      },
    });
    viewer.start();
    setStatus('Scene loaded — drag to orbit, scroll to zoom, WASD to fly.');
  } catch (error) {
    setStatus(`Failed to load scene: ${error.message || error}`);
    console.error(error);
  }
  if (bridge) bridge.viewerReady();
}

main();
