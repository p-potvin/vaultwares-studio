// VaultWares Studio splat viewport — GaussianSplats3D inside QWebEngineView.
// The Python side talks to this file via QWebChannel ("bridge" object) and
// window.* functions invoked through runJavaScript().

import * as THREE from 'three';
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

// Blender-style axis gizmo: corner widget mirroring the camera orientation.
// Click an axis ball to snap the view; drag the widget to orbit the camera.
function setupAxisGizmo(getCamera, getControls) {
  const size = 130;
  const canvas = document.createElement('canvas');
  Object.assign(canvas.style, {
    position: 'absolute', top: '10px', right: '10px',
    width: `${size}px`, height: `${size}px`, zIndex: 20, cursor: 'grab',
    borderRadius: '50%', background: 'rgba(20,20,26,0.35)',
  });
  document.body.appendChild(canvas);

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setSize(size, size, false);
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  const scene = new THREE.Scene();
  const gizmoCam = new THREE.OrthographicCamera(-1.9, 1.9, 1.9, -1.9, 0.1, 10);
  gizmoCam.position.set(0, 0, 5);

  const makeLabel = (text, color) => {
    const labelCanvas = document.createElement('canvas');
    labelCanvas.width = labelCanvas.height = 64;
    const ctx = labelCanvas.getContext('2d');
    ctx.font = 'bold 40px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = color;
    ctx.fillText(text, 32, 34);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: new THREE.CanvasTexture(labelCanvas), depthTest: false, transparent: true,
    }));
    sprite.scale.setScalar(0.62);
    return sprite;
  };

  const pickables = [];
  const axisDefs = [
    { dir: new THREE.Vector3(1, 0, 0), color: 0xe5544b, css: '#ffd9d6', label: 'X' },
    { dir: new THREE.Vector3(0, 1, 0), color: 0x7ab32a, css: '#e4f7c7', label: 'Y' },
    { dir: new THREE.Vector3(0, 0, 1), color: 0x4286e0, css: '#d8e9ff', label: 'Z' },
  ];
  for (const axis of axisDefs) {
    const lineGeometry = new THREE.BufferGeometry().setFromPoints(
      [new THREE.Vector3(0, 0, 0), axis.dir.clone().multiplyScalar(1.05)]
    );
    scene.add(new THREE.Line(lineGeometry, new THREE.LineBasicMaterial({ color: axis.color })));

    const tip = new THREE.Mesh(
      new THREE.SphereGeometry(0.30, 16, 16),
      new THREE.MeshBasicMaterial({ color: axis.color })
    );
    tip.position.copy(axis.dir).multiplyScalar(1.35);
    tip.userData.dir = axis.dir.clone();
    const label = makeLabel(axis.label, axis.css);
    label.position.copy(tip.position);
    scene.add(tip, label);
    pickables.push(tip);

    const negativeTip = new THREE.Mesh(
      new THREE.SphereGeometry(0.20, 16, 16),
      new THREE.MeshBasicMaterial({ color: axis.color, transparent: true, opacity: 0.45 })
    );
    negativeTip.position.copy(axis.dir).multiplyScalar(-1.35);
    negativeTip.userData.dir = axis.dir.clone().negate();
    scene.add(negativeTip);
    pickables.push(negativeTip);
  }

  const controlsTarget = () => {
    const controls = getControls();
    return controls ? controls.target.clone() : new THREE.Vector3();
  };

  const snapTo = (dir) => {
    const camera = getCamera();
    const target = controlsTarget();
    const distance = Math.max(camera.position.distanceTo(target), 0.5);
    const direction = dir.clone().normalize();
    if (Math.abs(direction.y) > 0.999) {
      direction.add(new THREE.Vector3(0, 0, 0.02)).normalize(); // dodge gimbal at the poles
    }
    camera.position.copy(target).addScaledVector(direction, distance);
    camera.up.set(0, 1, 0);
    camera.lookAt(target);
    const controls = getControls();
    if (controls) controls.update();
  };

  const orbit = (dx, dy) => {
    const camera = getCamera();
    const target = controlsTarget();
    const offset = camera.position.clone().sub(target);
    const spherical = new THREE.Spherical().setFromVector3(offset);
    spherical.theta -= dx * 0.012;
    spherical.phi = Math.min(Math.PI - 0.05, Math.max(0.05, spherical.phi - dy * 0.012));
    offset.setFromSpherical(spherical);
    camera.position.copy(target).add(offset);
    camera.lookAt(target);
    const controls = getControls();
    if (controls) controls.update();
  };

  let dragging = false;
  let movedDistance = 0;
  let last = [0, 0];
  canvas.addEventListener('pointerdown', (event) => {
    dragging = true;
    movedDistance = 0;
    last = [event.clientX, event.clientY];
    canvas.setPointerCapture(event.pointerId);
    canvas.style.cursor = 'grabbing';
    event.stopPropagation();
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    const dx = event.clientX - last[0];
    const dy = event.clientY - last[1];
    movedDistance += Math.abs(dx) + Math.abs(dy);
    last = [event.clientX, event.clientY];
    if (movedDistance > 4) orbit(dx, dy);
    event.stopPropagation();
  });
  canvas.addEventListener('pointerup', (event) => {
    dragging = false;
    canvas.style.cursor = 'grab';
    if (movedDistance <= 4) {
      const rect = canvas.getBoundingClientRect();
      const pointer = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -(((event.clientY - rect.top) / rect.height) * 2 - 1)
      );
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(pointer, gizmoCam);
      const hits = raycaster.intersectObjects(pickables, false);
      if (hits.length) snapTo(hits[0].object.userData.dir);
    }
    event.stopPropagation();
  });

  (function tick() {
    scene.quaternion.copy(getCamera().quaternion).invert();
    renderer.render(scene, gizmoCam);
    requestAnimationFrame(tick);
  })();
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
    setupAxisGizmo(() => viewer.camera, () => viewer.controls);
    setStatus('Scene loaded — drag to orbit, scroll to zoom, WASD to fly.');
  } catch (error) {
    setStatus(`Failed to load scene: ${error.message || error}`);
    console.error(error);
  }
  if (bridge) bridge.viewerReady();
}

main();
