// VaultWares Studio splat viewport — GaussianSplats3D inside QWebEngineView.
// The Python side talks to this file via QWebChannel ("bridge" object) and
// window.* functions invoked through runJavaScript().

import * as THREE from 'three';
import * as GaussianSplats3D from './vendor/gaussian-splats-3d.module.js';

const statusEl = document.getElementById('status');
let bridge = null;
let viewer = null;
// Mutable world-up reference shared between flipCameraUp and the FPS / WASD
// movement code. Starts as +Y; flipCameraUp inverts it so that after the user
// rights the scene, FPS yaw axis and the WASD right-vector cross product both
// flip together — otherwise "right" ends up pointing left, which is what
// "L/R inverted after Flip Up" was.
const worldUp = new THREE.Vector3(0, 1, 0);

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
    // requestAnimationFrame's `now` and performance.now() can use slightly
    // different time origins in QtWebEngine, so clamp the index into range
    // and skip the frame entirely if data is malformed.
    const raw = Math.floor((now - startedAt) / frameMs);
    const index = Math.max(0, Math.min(frames.length - 1, raw));
    const frame = frames[index];
    if (!frame || !frame.position || !frame.lookAt) {
      playback.handle = requestAnimationFrame(step);
      return;
    }
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

// Snap the orbit camera to a named view direction relative to its current
// target. Preserves the current orbit distance and forces world +Y as up so
// the result is always upright (use flipCameraUp afterwards if you want it
// inverted). Direction vectors are in world space; "top" looks DOWN at the
// scene from +Y, "front" looks at the scene from +Z, etc.
window.snapToView = function snapToView(viewName) {
  if (!viewer || !viewer.controls) return;
  const dirs = {
    top:    [0,  1, 0],
    bottom: [0, -1, 0],
    front:  [0,  0, 1],
    back:   [0,  0,-1],
    left:   [-1, 0, 0],
    right:  [1,  0, 0],
    iso:    [1,  0.6, 1],
  };
  const dir = dirs[String(viewName).toLowerCase()];
  if (!dir) return;
  const camera = viewer.camera;
  const target = viewer.controls.target.clone();
  const distance = Math.max(camera.position.distanceTo(target), 0.5);
  const direction = new THREE.Vector3(dir[0], dir[1], dir[2]).normalize();
  // Dodge the gimbal pole when snapping straight up/down (same trick the
  // axis gizmo already uses): nudge the direction slightly so OrbitControls
  // doesn't end up in the clamped-polar corner.
  if (Math.abs(direction.y) > 0.999) {
    direction.add(new THREE.Vector3(0, 0, 0.02)).normalize();
  }
  camera.position.copy(target).addScaledVector(direction, distance);
  camera.up.set(0, 1, 0);
  camera.lookAt(target);
  viewer.controls.update();
  setStatus(`Snapped to ${viewName}.`);
};

// Roll the camera 180 degrees around its forward axis. Useful when the orbit
// has somehow ended up upside-down (rare, but the pole clamp doesn't catch
// every weird state) — gives a one-click "right myself" button.
// Tracks the current flip state and notifies Python whenever it changes so
// the choice survives a viewer reload. Direct calls (e.g. the auto-flip on
// load) should pass {silent: true} to skip the bridge notification.
let isFlipped = false;
window.flipCameraUp = function flipCameraUp(options) {
  if (!viewer) return;
  const camera = viewer.camera;
  camera.up.set(-camera.up.x, -camera.up.y, -camera.up.z);
  // Flip the FPS / WASD up reference too, otherwise after righting the scene
  // the right-vector cross-product flips sign and A/D feel inverted.
  worldUp.set(-worldUp.x, -worldUp.y, -worldUp.z);
  const target = viewer.controls ? viewer.controls.target : { x: 0, y: 0, z: 0 };
  camera.lookAt(target.x, target.y, target.z);
  if (viewer.controls) viewer.controls.update();
  isFlipped = !isFlipped;
  if (!(options && options.silent) && bridge && bridge.viewerFlipChanged) {
    bridge.viewerFlipChanged(isFlipped);
  }
  setStatus('Camera up flipped.');
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
    // Pick the loader from the scene path extension so the same code path
    // serves cloud.ply (legacy) and the packed cloud.splat (current default).
    let format = GaussianSplats3D.SceneFormat.Ply;
    if (scene.endsWith('.splat')) format = GaussianSplats3D.SceneFormat.Splat;
    else if (scene.endsWith('.ksplat')) format = GaussianSplats3D.SceneFormat.KSplat;
    await viewer.addSplatScene(scene, {
      format,
      progressiveLoad: true,
      showLoadingUI: true,
      onProgress: (pct, label, stage) => {
        if (pct !== undefined && pct !== null) setStatus(`Loading splats: ${Math.round(pct)}%`);
      },
    });
    viewer.start();
    // Keep the orbit away from the poles. OrbitControls allows phi in [0, π]
    // by default, so dragging through straight-up or straight-down crosses the
    // gimbal pole and snaps the camera roll by 180° on two axes at once — reads
    // as the camera flipping "inverted on 2 axes". The axis gizmo already does
    // the same clamp at 0.05; mirror it here on the main controls.
    if (viewer.controls) {
      viewer.controls.minPolarAngle = 0.05;
      viewer.controls.maxPolarAngle = Math.PI - 0.05;
      // OrbitControls defaults feel hot inside QWebEngineView — DPI scaling
      // amplifies mouse deltas. The free-look path uses its own (lower)
      // sensitivity; left-drag orbit stays slow to match that calmer feel.
      viewer.controls.rotateSpeed = 0.25;
      viewer.controls.panSpeed = 0.5;
      viewer.controls.zoomSpeed = 0.6;
      // Damping stays ON (enableDamping=false breaks WASD walking by changing
      // how OrbitControls.update() handles externally-mutated camera state),
      // but the factor is bumped up so the post-release glide decays in a
      // few frames instead of half a second. 0.05 was the default → noticeable
      // overshoot; 0.3 is tight without feeling jerky.
      viewer.controls.dampingFactor = 0.3;
      // Enforce world +Y as camera up at load. Gravity alignment puts the
      // scene's up axis on world +Y, but GaussianSplats3D's auto-framed
      // camera occasionally lands with up=(0,-1,0) — which is why the view
      // started upside down. Setting it once here means the user doesn't
      // need to hit Flip Up on every load.
      viewer.camera.up.set(0, 1, 0);
      viewer.controls.update();
    }
    // Per-job sticky flip: Python passes flip=1 when the user previously hit
    // Flip Up on this job, so the orientation persists across reloads. Silent
    // call avoids re-notifying Python (we already know the persisted state).
    if (params.get('flip') === '1') {
      window.flipCameraUp({ silent: true });
    }
    setupAxisGizmo(() => viewer.camera, () => viewer.controls);
    // Infinite zoom: orbit controls asymptote at the target and feel like a
    // wall. When zooming in close, push the target forward so scrolling
    // carries the camera THROUGH the scene instead of stalling at the pivot.
    window.addEventListener('wheel', (event) => {
      if (event.deltaY >= 0 || !viewer.controls) return;
      const camera = viewer.camera;
      const target = viewer.controls.target;
      const distance = camera.position.distanceTo(target);
      if (distance < 0.4) {
        const forward = new THREE.Vector3().subVectors(target, camera.position).normalize();
        target.addScaledVector(forward, Math.max(distance, 0.06) * 0.35);
        viewer.controls.update();
      }
    }, { passive: true });

    // Pointer leave / re-enter teleport fix.
    //
    // OrbitControls calls setPointerCapture on pointerdown, but Qt's
    // WebEngineView does not reliably forward pointer events once the cursor
    // leaves the widget bounds. If the user releases the button outside the
    // viewport the embedded Chromium never sees pointerup; when they come
    // back in, the next pointermove delta is huge and the camera teleports.
    // Synthesise a pointerup on pointerleave so the drag ends cleanly.
    const rendererCanvas = viewer.renderer && viewer.renderer.domElement;
    if (rendererCanvas) {
      const activePointers = new Set();
      rendererCanvas.addEventListener('pointerdown', (event) => {
        activePointers.add(event.pointerId);
      });
      rendererCanvas.addEventListener('pointerup', (event) => {
        activePointers.delete(event.pointerId);
      });
      rendererCanvas.addEventListener('pointercancel', (event) => {
        activePointers.delete(event.pointerId);
      });
      rendererCanvas.addEventListener('pointerleave', () => {
        for (const pointerId of activePointers) {
          rendererCanvas.dispatchEvent(new PointerEvent('pointerup', {
            pointerId, bubbles: true, cancelable: true,
          }));
        }
        activePointers.clear();
      });
    }

    // Real WASD fly. OrbitControls only orbits around its fixed target, so the
    // user can't actually translate toward a point in the scene that isn't the
    // current target — they end up arcing past it. Mutating BOTH camera.position
    // and controls.target by the same delta moves the orbit pivot with the
    // camera, so WASD feels like first-person walking. QE raises/lowers along
    // world up; Shift speeds up.
    const heldKeys = new Set();
    const keyTargets = new Set(['w', 'a', 's', 'd', 'q', 'e', 'shift']);
    window.addEventListener('keydown', (event) => {
      const key = event.key.toLowerCase();
      if (keyTargets.has(key)) heldKeys.add(key);
    });
    window.addEventListener('keyup', (event) => {
      heldKeys.delete(event.key.toLowerCase());
    });
    window.addEventListener('blur', () => heldKeys.clear());

    // Free-look mode: hold right-mouse (or both buttons) to pivot the camera
    // in place — yaw + pitch, no translation, cursor stays visible (no pointer
    // lock). This mimics how a person turns their head: the body stays
    // planted, the eyes sweep around. WASD continues to walk in this mode, so
    // free-look + WASD together is the practical "first-person walking"
    // experience; right-click alone is the precise-framing tool. Pointer lock
    // turned out to be the wrong abstraction for precise framing — hiding the
    // cursor made it hard to know what you were pointing at.
    //
    // Math: snapshot the camera's quaternion at entry as `lookBaseQuat`, then
    // accumulate scalar yaw + pitch deltas. Final orientation per frame is
    // qPitch * qYaw * lookBaseQuat, where qYaw is around worldUp and qPitch
    // is around the right axis after yaw. Textbook roll-free FPS camera; the
    // pitch scalar is trivially clamped to ±89°.
    let lookMode = false;
    let lookYaw = 0;
    let lookPitch = 0;
    let leftDown = false;
    let rightDown = false;
    const LOOK_TARGET_DISTANCE = 3.0;
    const PITCH_LIMIT = Math.PI / 2 - 0.02; // ~88.8 degrees
    const lookBaseQuat = new THREE.Quaternion();
    const _vec = new THREE.Vector3();
    const _qYaw = new THREE.Quaternion();
    const _qPitch = new THREE.Quaternion();
    const _baseRight = new THREE.Vector3();
    const _rightAfterYaw = new THREE.Vector3();

    function applyLookOrientation() {
      _qYaw.setFromAxisAngle(worldUp, lookYaw);
      // The pitch axis is the base-camera's local +X transformed by qYaw —
      // i.e. the right axis the camera would have after yaw alone. Always
      // perpendicular to worldUp by construction, so pitch never induces roll.
      _baseRight.set(1, 0, 0).applyQuaternion(lookBaseQuat);
      _rightAfterYaw.copy(_baseRight).applyQuaternion(_qYaw);
      _qPitch.setFromAxisAngle(_rightAfterYaw, lookPitch);
      viewer.camera.quaternion
        .copy(lookBaseQuat)
        .premultiply(_qYaw)
        .premultiply(_qPitch);
      // Park the orbit target on the camera's forward ray so capture poses
      // and post-free-look orbit drag both have a coherent pivot in view.
      _vec.set(0, 0, -1).applyQuaternion(viewer.camera.quaternion);
      viewer.controls.target
        .copy(viewer.camera.position)
        .addScaledVector(_vec, LOOK_TARGET_DISTANCE);
    }

    function enterLookMode() {
      lookBaseQuat.copy(viewer.camera.quaternion);
      lookYaw = 0;
      lookPitch = 0;
      lookMode = true;
      if (viewer.controls) viewer.controls.enabled = false;
      setStatus('Free-look — drag to pivot in place, WASD to walk, release right-mouse to exit.');
    }

    function exitLookMode() {
      lookMode = false;
      if (viewer.controls) viewer.controls.enabled = true;
      setStatus('Scene loaded — right-click+drag for free-look, drag to orbit, WASD to fly, scroll to zoom.');
    }

    if (rendererCanvas) {
      // Suppress the native right-click menu so it doesn't fight free-look.
      // The contextmenu event fires on right-mouseUP, and in free-look the
      // cursor often wanders off the canvas before release — so the menu
      // would open on whatever document element ended up under the pointer.
      // Binding the listener at document level catches all of those.
      document.addEventListener('contextmenu', (event) => event.preventDefault());

      rendererCanvas.addEventListener('mousedown', (event) => {
        if (event.button === 0) leftDown = true;
        if (event.button === 2) rightDown = true;
        // Right-mouse-down (alone or with left) → enter free-look. Stop
        // propagation so OrbitControls' own right-button binding (pan) never
        // fires — it would fight our rotation and leave the target adrift.
        if (event.button === 2 && !lookMode) {
          event.preventDefault();
          event.stopPropagation();
          enterLookMode();
        }
      });

      window.addEventListener('mouseup', (event) => {
        if (event.button === 0) leftDown = false;
        if (event.button === 2) rightDown = false;
        // Exit free-look when right releases (the canonical binding). Both-
        // buttons callers exit on either release, which matches the original
        // both-buttons gesture and avoids stuck-in-look after they let go.
        if (lookMode && !rightDown) {
          exitLookMode();
        }
      });

      document.addEventListener('mousemove', (event) => {
        if (!lookMode) return;
        // Sensitivity tuned for QtWebEngine's DPI-amplified deltas; halve
        // for a slower feel or double for snappier.
        const sens = 0.0022;
        lookYaw -= event.movementX * sens;
        lookPitch -= event.movementY * sens;
        lookPitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, lookPitch));
        applyLookOrientation();
      });

      // Cancel free-look if the canvas loses focus mid-drag (alt-tab, etc.)
      // so the next pointerdown starts clean.
      window.addEventListener('blur', () => {
        leftDown = false;
        rightDown = false;
        if (lookMode) exitLookMode();
      });
    }

    let lastFrameMs = performance.now();
    function flyTick(now) {
      const dt = Math.min(0.1, (now - lastFrameMs) / 1000);
      lastFrameMs = now;
      requestAnimationFrame(flyTick);
      if (!viewer.controls || heldKeys.size === 0) return;
      const camera = viewer.camera;
      const target = viewer.controls.target;
      let forward;
      let distance;
      if (lookMode) {
        // In free-look mode movement is along the camera's HORIZONTAL forward
        // so looking down doesn't slow walking. Project camera forward onto
        // the plane perpendicular to worldUp; QE handles vertical separately.
        forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
        forward.addScaledVector(worldUp, -forward.dot(worldUp)).normalize();
        distance = LOOK_TARGET_DISTANCE;
      } else {
        forward = new THREE.Vector3().subVectors(target, camera.position);
        distance = forward.length();
        if (distance < 1e-4) return;
        forward.normalize();
      }
      const right = new THREE.Vector3().crossVectors(forward, worldUp).normalize();
      // Scale move speed with the orbit distance so the same key feel works at
      // both far and close framings. The multipliers were tuned down (0.6 ->
      // 0.35, floor 0.4 -> 0.25) after user feedback that fly was too hot;
      // Shift sprint still gives 3x for fast traversal.
      const baseSpeed = Math.max(0.25, distance * 0.35);
      const speed = baseSpeed * (heldKeys.has('shift') ? 3 : 1) * dt;
      const delta = new THREE.Vector3();
      if (heldKeys.has('w')) delta.addScaledVector(forward, speed);
      if (heldKeys.has('s')) delta.addScaledVector(forward, -speed);
      if (heldKeys.has('d')) delta.addScaledVector(right, speed);
      if (heldKeys.has('a')) delta.addScaledVector(right, -speed);
      // E/Q follow worldUp so they always feel "up/down" in the user's frame,
      // even after Flip Up — when worldUp is (0,-1,0) pressing E goes visually up.
      if (heldKeys.has('e')) delta.addScaledVector(worldUp, speed);
      if (heldKeys.has('q')) delta.addScaledVector(worldUp, -speed);
      if (delta.lengthSq() === 0) return;
      camera.position.add(delta);
      target.add(delta);
      viewer.controls.update();
    }
    requestAnimationFrame(flyTick);

    setStatus('Scene loaded — right-click+drag for free-look, drag to orbit, WASD to fly, scroll to zoom.');
  } catch (error) {
    setStatus(`Failed to load scene: ${error.message || error}`);
    console.error(error);
  }
  if (bridge) bridge.viewerReady();
}

main();
