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
  const lookAt = framed ? center : [0, 0, 0];
  const position = framed
    ? [center[0], center[1] + radius * 0.4, center[2] + radius * 1.2]
    : [0, 1.5, 4];

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
