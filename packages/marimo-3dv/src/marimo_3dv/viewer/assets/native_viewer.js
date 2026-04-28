function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalize(vec) {
  const length = Math.hypot(vec[0], vec[1], vec[2]);
  if (length < 1e-8) {
    return [0, 0, 1];
  }
  return [vec[0] / length, vec[1] / length, vec[2] / length];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(vec, scalar) {
  return [vec[0] * scalar, vec[1] * scalar, vec[2] * scalar];
}

function lookAtCamera(position, target, upDirection) {
  let forward = normalize(subtract(target, position));
  let right = cross(forward, upDirection);
  if (Math.hypot(right[0], right[1], right[2]) < 1e-8) {
    right = cross(forward, [0, 0, 1]);
  }
  right = normalize(right);
  const up = normalize(cross(forward, right));
  return [
    [right[0], up[0], forward[0], position[0]],
    [right[1], up[1], forward[1], position[1]],
    [right[2], up[2], forward[2], position[2]],
    [0, 0, 0, 1],
  ];
}

function matrixColumn(matrix, index) {
  return [matrix[0][index], matrix[1][index], matrix[2][index]];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function squaredDistance2D(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return dx * dx + dy * dy;
}

function dedupePoints2D(points, epsilon = 1e-4) {
  const deduped = [];
  const maxSquaredDistance = epsilon * epsilon;
  for (const point of points) {
    if (
      !deduped.some((existingPoint) =>
        squaredDistance2D(existingPoint, point) <= maxSquaredDistance)
    ) {
      deduped.push(point);
    }
  }
  return deduped;
}

function clipImplicitLineToViewport(a, b, c, width, height) {
  const intersections = [];
  if (Math.abs(b) > 1e-8) {
    intersections.push([0.0, (-c) / b]);
    intersections.push([width, (-c - a * width) / b]);
  }
  if (Math.abs(a) > 1e-8) {
    intersections.push([(-c) / a, 0.0]);
    intersections.push([(-c - b * height) / a, height]);
  }

  const visiblePoints = dedupePoints2D(
    intersections.filter(
      ([x, y]) =>
        Number.isFinite(x)
        && Number.isFinite(y)
        && x >= 0.0
        && x <= width
        && y >= 0.0
        && y <= height,
    ),
  );
  if (visiblePoints.length < 2) {
    return null;
  }

  let bestSegment = [visiblePoints[0], visiblePoints[1]];
  let bestSquaredDistance = squaredDistance2D(
    visiblePoints[0],
    visiblePoints[1],
  );
  for (let index = 0; index < visiblePoints.length; index += 1) {
    for (
      let otherIndex = index + 1;
      otherIndex < visiblePoints.length;
      otherIndex += 1
    ) {
      const candidateSquaredDistance = squaredDistance2D(
        visiblePoints[index],
        visiblePoints[otherIndex],
      );
      if (candidateSquaredDistance > bestSquaredDistance) {
        bestSquaredDistance = candidateSquaredDistance;
        bestSegment = [visiblePoints[index], visiblePoints[otherIndex]];
      }
    }
  }
  if (bestSquaredDistance <= 1e-8) {
    return null;
  }
  return bestSegment;
}

function conventionRotation(cameraConvention) {
  const rotations = {
    opencv: [
      [1, 0, 0],
      [0, -1, 0],
      [0, 0, 1],
    ],
    opengl: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, -1],
    ],
    blender: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, -1],
    ],
    colmap: [
      [1, 0, 0],
      [0, -1, 0],
      [0, 0, 1],
    ],
  };
  return rotations[cameraConvention] ?? rotations.opencv;
}

function multiplyMat3(a, b) {
  const result = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      let sum = 0;
      for (let index = 0; index < 3; index += 1) {
        sum += a[row][index] * b[index][col];
      }
      result[row][col] = sum;
    }
  }
  return result;
}

function multiplyMat3Vec3(matrix, vector) {
  return [
    matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
    matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
    matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
  ];
}

function axisAngleRotation(axis, angleRadians) {
  const [x, y, z] = normalize(axis);
  const c = Math.cos(angleRadians);
  const s = Math.sin(angleRadians);
  const t = 1.0 - c;
  return [
    [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
    [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
    [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
  ];
}

function rotationFromCamToWorld(camToWorld) {
  return [
    [camToWorld[0][0], camToWorld[0][1], camToWorld[0][2]],
    [camToWorld[1][0], camToWorld[1][1], camToWorld[1][2]],
    [camToWorld[2][0], camToWorld[2][1], camToWorld[2][2]],
  ];
}

function rotationMatrixXYZ(xDegrees, yDegrees, zDegrees) {
  const [xRadians, yRadians, zRadians] = [xDegrees, yDegrees, zDegrees]
    .map((degrees) => degrees * Math.PI / 180.0);
  const cx = Math.cos(xRadians);
  const sx = Math.sin(xRadians);
  const cy = Math.cos(yRadians);
  const sy = Math.sin(yRadians);
  const cz = Math.cos(zRadians);
  const sz = Math.sin(zRadians);
  const rotationX = [
    [1, 0, 0],
    [0, cx, -sx],
    [0, sx, cx],
  ];
  const rotationY = [
    [cy, 0, sy],
    [0, 1, 0],
    [-sy, 0, cy],
  ];
  const rotationZ = [
    [cz, -sz, 0],
    [sz, cz, 0],
    [0, 0, 1],
  ];
  return multiplyMat3(rotationZ, multiplyMat3(rotationY, rotationX));
}

function convertCamToWorldConvention(
  camToWorld,
  sourceConvention,
  targetConvention,
) {
  const sourceTransform = conventionRotation(sourceConvention);
  const targetTransform = conventionRotation(targetConvention);
  const rotation = [
    [camToWorld[0][0], camToWorld[0][1], camToWorld[0][2]],
    [camToWorld[1][0], camToWorld[1][1], camToWorld[1][2]],
    [camToWorld[2][0], camToWorld[2][1], camToWorld[2][2]],
  ];
  const internalRotation = multiplyMat3(rotation, sourceTransform);
  const targetRotation = multiplyMat3(internalRotation, targetTransform);
  return [
    [targetRotation[0][0], targetRotation[0][1], targetRotation[0][2], camToWorld[0][3]],
    [targetRotation[1][0], targetRotation[1][1], targetRotation[1][2], camToWorld[1][3]],
    [targetRotation[2][0], targetRotation[2][1], targetRotation[2][2], camToWorld[2][3]],
    [0, 0, 0, 1],
  ];
}

function parseCameraState(cameraStateJson) {
  const externalState = JSON.parse(cameraStateJson);
  const convention = externalState.camera_convention ?? "opencv";
  return {
    fov_degrees: externalState.fov_degrees,
    width: externalState.width,
    height: externalState.height,
    camera_convention: convention,
    cam_to_world: convertCamToWorldConvention(
      externalState.cam_to_world,
      convention,
      "opencv",
    ),
  };
}

function bufferToUint8Array(buffer) {
  if (buffer instanceof Uint8Array) {
    return buffer;
  }
  if (buffer instanceof ArrayBuffer) {
    return new Uint8Array(buffer);
  }
  if (ArrayBuffer.isView(buffer)) {
    return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  }
  return null;
}

function parseFramePacket(packet) {
  const bytes = bufferToUint8Array(packet);
  if (bytes === null || bytes.byteLength < 4) {
    return null;
  }
  const headerLength =
    (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
  if (headerLength < 0 || bytes.byteLength < 4 + headerLength) {
    return null;
  }
  const headerBytes = bytes.subarray(4, 4 + headerLength);
  const payload = bytes.subarray(4 + headerLength);
  const header = JSON.parse(new TextDecoder().decode(headerBytes));
  return { header, payload };
}

function render({ model, el }) {
  const root = document.createElement("div");
  root.className = "native-viewer-root";

  const frame = document.createElement("canvas");
  frame.className = "native-viewer-frame";
  frame.tabIndex = 0;
  frame.setAttribute("aria-label", "Native 3D viewer");
  root.appendChild(frame);
  const frameContext = frame.getContext("2d");
  if (frameContext === null) {
    throw new Error("Failed to acquire 2D canvas context.");
  }

  const overlay = document.createElement("div");
  overlay.className = "native-viewer-overlay";

  const latencyBadge = document.createElement("div");
  latencyBadge.className = "native-viewer-badge native-viewer-latency";
  latencyBadge.hidden = true;

  overlay.appendChild(latencyBadge);
  root.appendChild(overlay);

  const axesCanvas = document.createElement("canvas");
  axesCanvas.className = "native-viewer-axes";
  root.appendChild(axesCanvas);
  const axesContext = axesCanvas.getContext("2d");
  if (axesContext === null) {
    throw new Error("Failed to acquire 2D axis canvas context.");
  }

  const guidesCanvas = document.createElement("canvas");
  guidesCanvas.className = "native-viewer-guides";
  root.appendChild(guidesCanvas);
  const guidesContext = guidesCanvas.getContext("2d");
  if (guidesContext === null) {
    throw new Error("Failed to acquire 2D guides canvas context.");
  }
  el.appendChild(root);

  let cameraState = parseCameraState(model.get("camera_state_json"));
  let position = [
    cameraState.cam_to_world[0][3],
    cameraState.cam_to_world[1][3],
    cameraState.cam_to_world[2][3],
  ];
  let target = add(position, scale(matrixColumn(cameraState.cam_to_world, 2), 3.0));
  let orbitDistance = Math.max(1e-3, Math.hypot(...subtract(position, target)));
  let interaction = null;
  let animationFrame = null;
  let lastTickMs = null;
  const pressedKeys = new Set();
  const clickThresholdPixels = 4.0;
  let lastFrameRevision = -1;
  let averageLatencyMs = null;
  let lastLatencySampleMs = null;
  let lastLatencySampleAtMs = null;
  let lastRenderTimeMs = null;
  let lastDecodeTimeMs = null;
  let lastDrawTimeMs = null;
  let lastPresentWaitTimeMs = null;
  let lastReceiveQueueTimeMs = null;
  let lastPostReceiveTimeMs = null;
  let lastPacketSizeBytes = 0;
  let lastBackendToBrowserTimeMs = null;
  let smoothedRenderTimeMs = null;
  let smoothedDecodeTimeMs = null;
  let smoothedDrawTimeMs = null;
  let smoothedPresentWaitTimeMs = null;
  let smoothedReceiveQueueTimeMs = null;
  let smoothedPostReceiveTimeMs = null;
  let smoothedPacketSizeBytes = null;
  let smoothedBackendToBrowserTimeMs = null;
  const revisionSentAtMs = new Map();
  const recentDrawTimestamps = [];
  let viewerFps = null;
  let renderFps = null;
  let latestScheduledFrameRevision = -1;
  let interactionActive = Boolean(model.get("interaction_active"));
  let settleTimeoutId = null;
  const settleDelayMs = 150;
  let streamSocket = null;
  let reconnectTimeoutId = null;
  let closed = false;
  let nextClockSyncPingId = 0;
  let bestClockSyncRttMs = null;
  let backendClockOffsetMs = null;
  const pendingClockSyncPings = new Map();

  function setCameraRotation(rotation) {
    cameraState.cam_to_world = [
      [rotation[0][0], rotation[0][1], rotation[0][2], position[0]],
      [rotation[1][0], rotation[1][1], rotation[1][2], position[1]],
      [rotation[2][0], rotation[2][1], rotation[2][2], position[2]],
      [0, 0, 0, 1],
    ];
  }

  function viewerFrameRotation() {
    return rotationMatrixXYZ(
      Number(model.get("viewer_rotation_x_degrees")) || 0.0,
      Number(model.get("viewer_rotation_y_degrees")) || 0.0,
      Number(model.get("viewer_rotation_z_degrees")) || 0.0,
    );
  }

  function viewerUpVector() {
    return normalize(
      multiplyMat3Vec3(viewerFrameRotation(), [0, -1, 0]),
    );
  }

  function drawAxesGizmo() {
    const shouldShowAxes = Boolean(model.get("show_axes"));
    axesCanvas.hidden = !shouldShowAxes;
    if (!shouldShowAxes) {
      return;
    }

    const size = 96;
    const dpr = window.devicePixelRatio || 1;
    axesCanvas.width = Math.round(size * dpr);
    axesCanvas.height = Math.round(size * dpr);
    axesCanvas.style.width = `${size}px`;
    axesCanvas.style.height = `${size}px`;
    axesContext.setTransform(dpr, 0, 0, dpr, 0, 0);
    axesContext.clearRect(0, 0, size, size);

    const centerX = size / 2;
    const centerY = size / 2;
    const axisLength = 28;
    const markerRadius = 4;
    const viewRotation = cameraState.cam_to_world;
    const frameRotation = viewerFrameRotation();
    const cameraRight = matrixColumn(viewRotation, 0);
    const cameraUp = matrixColumn(viewRotation, 1);
    const cameraForward = matrixColumn(viewRotation, 2);
    const axes = [
      { label: "X", color: "#ef4444", world: [1, 0, 0] },
      { label: "Y", color: "#22c55e", world: [0, 1, 0] },
      { label: "Z", color: "#3b82f6", world: [0, 0, 1] },
    ].map((axis) => {
      const rotatedWorld = multiplyMat3Vec3(frameRotation, axis.world);
      const cameraX = dot(rotatedWorld, cameraRight);
      const cameraY = dot(rotatedWorld, cameraUp);
      const cameraZ = dot(rotatedWorld, cameraForward);
      return {
        ...axis,
        endX: centerX + cameraX * axisLength,
        endY: centerY + cameraY * axisLength,
        depth: cameraZ,
      };
    });

    axes.sort((first, second) => first.depth - second.depth);

    axesContext.lineWidth = 2.5;
    axesContext.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
    axesContext.textAlign = "center";
    axesContext.textBaseline = "middle";

    for (const axis of axes) {
      axesContext.strokeStyle = axis.color;
      axesContext.fillStyle = axis.color;
      axesContext.globalAlpha = axis.depth >= 0 ? 1.0 : 0.45;

      axesContext.beginPath();
      axesContext.moveTo(centerX, centerY);
      axesContext.lineTo(axis.endX, axis.endY);
      axesContext.stroke();

      axesContext.beginPath();
      axesContext.arc(axis.endX, axis.endY, markerRadius, 0, Math.PI * 2);
      axesContext.fill();

      axesContext.fillText(
        axis.label,
        axis.endX + (axis.endX >= centerX ? 10 : -10),
        axis.endY + (axis.endY >= centerY ? 10 : -10),
      );
    }

    axesContext.globalAlpha = 1.0;
  }

  function drawHorizon() {
    const shouldShowHorizon = Boolean(model.get("show_horizon"));
    const shouldShowOrigin = Boolean(model.get("show_origin"));
    guidesCanvas.hidden = !shouldShowHorizon && !shouldShowOrigin;
    if (!shouldShowHorizon && !shouldShowOrigin) {
      return;
    }

    const rect = root.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width || cameraState.width));
    const height = Math.max(1, Math.round(rect.height || cameraState.height));
    const dpr = window.devicePixelRatio || 1;
    guidesCanvas.width = Math.round(width * dpr);
    guidesCanvas.height = Math.round(height * dpr);
    guidesCanvas.style.width = `${width}px`;
    guidesCanvas.style.height = `${height}px`;
    guidesContext.setTransform(dpr, 0, 0, dpr, 0, 0);
    guidesContext.clearRect(0, 0, width, height);

    const cameraRight = matrixColumn(cameraState.cam_to_world, 0);
    const cameraUp = matrixColumn(cameraState.cam_to_world, 1);
    const cameraForward = matrixColumn(cameraState.cam_to_world, 2);

    if (shouldShowHorizon) {
      const up = viewerUpVector();
      const projectedUp = [
        dot(up, cameraRight),
        dot(up, cameraUp),
        dot(up, cameraForward),
      ];
      const halfFovRadians = (cameraState.fov_degrees * Math.PI / 180.0) / 2.0;
      const focal = (height / 2.0) / Math.tan(halfFovRadians);
      const centerX = width / 2.0;
      const centerY = height / 2.0;
      const a = projectedUp[0] / focal;
      const b = projectedUp[1] / focal;
      const c = projectedUp[2] - centerX * a - centerY * b;
      const clippedSegment = clipImplicitLineToViewport(
        a,
        b,
        c,
        width,
        height,
      );
      if (clippedSegment !== null) {
        guidesContext.strokeStyle = "rgba(255,255,255,0.8)";
        guidesContext.lineWidth = 2.0;
        guidesContext.beginPath();
        guidesContext.moveTo(clippedSegment[0][0], clippedSegment[0][1]);
        guidesContext.lineTo(clippedSegment[1][0], clippedSegment[1][1]);
        guidesContext.stroke();
      }
    }

    if (shouldShowOrigin) {
      const origin = [
        Number(model.get("origin_x")) || 0.0,
        Number(model.get("origin_y")) || 0.0,
        Number(model.get("origin_z")) || 0.0,
      ];
      const cameraPosition = [
        cameraState.cam_to_world[0][3],
        cameraState.cam_to_world[1][3],
        cameraState.cam_to_world[2][3],
      ];
      const relative = subtract(origin, cameraPosition);
      const cameraX = dot(relative, cameraRight);
      const cameraY = dot(relative, cameraUp);
      const cameraZ = dot(relative, cameraForward);
      if (cameraZ > 1e-5) {
        const halfFovRadians = (cameraState.fov_degrees * Math.PI / 180.0) / 2.0;
        const focal = (height / 2.0) / Math.tan(halfFovRadians);
        const pixelX = (cameraX / cameraZ) * focal + width / 2.0;
        const pixelY = (cameraY / cameraZ) * focal + height / 2.0;
        const radius = 6.0;

        guidesContext.strokeStyle = "rgba(255,191,36,0.95)";
        guidesContext.lineWidth = 2.0;
        guidesContext.beginPath();
        guidesContext.moveTo(pixelX - radius, pixelY);
        guidesContext.lineTo(pixelX + radius, pixelY);
        guidesContext.moveTo(pixelX, pixelY - radius);
        guidesContext.lineTo(pixelX, pixelY + radius);
        guidesContext.stroke();
      }
    }
  }

  function updateAspectRatio() {
    const explicitAspectRatio = Number(model.get("aspect_ratio"));
    const fallbackAspectRatio =
      Math.max(1, cameraState.width) / Math.max(1, cameraState.height);
    const aspectRatio =
      Number.isFinite(explicitAspectRatio) && explicitAspectRatio > 0.0
        ? explicitAspectRatio
        : fallbackAspectRatio;
    root.style.aspectRatio = `${aspectRatio}`;
  }

  function getViewportSize() {
    const rect = root.getBoundingClientRect();
    return {
      width: Math.max(1, Math.round(rect.width || cameraState.width)),
      height: Math.max(1, Math.round(rect.height || cameraState.height)),
    };
  }

  function updateLatencyBadge() {
    const shouldShowStats = Boolean(model.get("show_stats"));
    if (!shouldShowStats || averageLatencyMs === null) {
      latencyBadge.hidden = true;
      return;
    }
    latencyBadge.hidden = false;
    const viewerMs = `${Math.round(averageLatencyMs)}ms`;
    const viewerFpsStr = viewerFps !== null ? `${Math.round(viewerFps)}fps` : "—fps";
    const renderMs = lastRenderTimeMs !== null ? `${Math.round(lastRenderTimeMs)}ms` : "—ms";
    const renderFpsStr = renderFps !== null ? `${Math.round(renderFps)}fps` : "—fps";
    latencyBadge.innerHTML =
      `<strong>Viewer</strong> ${viewerMs} ${viewerFpsStr}<br><strong>Render</strong> ${renderMs} ${renderFpsStr}`;
  }

  function syncMetricsToModel() {
    if (
      averageLatencyMs === null
      || smoothedRenderTimeMs === null
      || smoothedReceiveQueueTimeMs === null
      || smoothedPostReceiveTimeMs === null
      || smoothedDecodeTimeMs === null
      || smoothedDrawTimeMs === null
      || smoothedPresentWaitTimeMs === null
      || smoothedPacketSizeBytes === null
    ) {
      return;
    }
    model.set("latency_ms", averageLatencyMs);
    model.set("latency_sample_ms", lastLatencySampleMs);
    model.set("render_time_ms", smoothedRenderTimeMs);
    model.set(
      "backend_to_browser_time_ms",
      smoothedBackendToBrowserTimeMs ?? 0.0,
    );
    model.set("packet_size_bytes", Math.round(smoothedPacketSizeBytes));
    model.set("browser_receive_queue_ms", smoothedReceiveQueueTimeMs);
    model.set("browser_post_receive_ms", smoothedPostReceiveTimeMs);
    model.set("browser_decode_time_ms", smoothedDecodeTimeMs);
    model.set("browser_draw_time_ms", smoothedDrawTimeMs);
    model.set("browser_present_wait_ms", smoothedPresentWaitTimeMs);
    model.save_changes();
  }

  function smoothMetric(previous, sample, shouldReset) {
    if (previous === null || shouldReset) {
      return sample;
    }
    return previous * 0.85 + sample * 0.15;
  }

  function updatePoseFromMatrix() {
    position = [
      cameraState.cam_to_world[0][3],
      cameraState.cam_to_world[1][3],
      cameraState.cam_to_world[2][3],
    ];
  }

  function serializeCameraState() {
    return JSON.stringify({
      fov_degrees: cameraState.fov_degrees,
      width: cameraState.width,
      height: cameraState.height,
      camera_convention: cameraState.camera_convention,
      cam_to_world: convertCamToWorldConvention(
        cameraState.cam_to_world,
        "opencv",
        cameraState.camera_convention,
      ),
    });
  }

function updateCameraMatrix() {
    setCameraRotation(rotationFromCamToWorld(cameraState.cam_to_world));
  }

  function syncSizeIntoCameraState() {
    const viewport = getViewportSize();
    cameraState.width = viewport.width;
    cameraState.height = viewport.height;
  }

  function pushCameraState() {
    syncSizeIntoCameraState();
    updateCameraMatrix();
    const nextJson = serializeCameraState();
    if (nextJson === model.get("camera_state_json")) {
      return;
    }
    const nextRevision = model.get("_camera_revision") + 1;
    revisionSentAtMs.set(nextRevision, performance.now());
    model.set("camera_state_json", nextJson);
    model.set("_camera_revision", nextRevision);
    model.save_changes();
  }

  function requestSettledRender() {
    const nextRevision = model.get("_camera_revision") + 1;
    revisionSentAtMs.set(nextRevision, performance.now());
    model.set("interaction_active", false);
    model.set("_camera_revision", nextRevision);
    model.save_changes();
    interactionActive = false;
  }

  function scheduleSettledRender() {
    if (settleTimeoutId !== null) {
      clearTimeout(settleTimeoutId);
    }
    settleTimeoutId = setTimeout(() => {
      settleTimeoutId = null;
      if (interaction !== null || pressedKeys.size > 0) {
        return;
      }
      if (interactionActive) {
        requestSettledRender();
      }
    }, settleDelayMs);
  }

  function markInteractionActive() {
    if (settleTimeoutId !== null) {
      clearTimeout(settleTimeoutId);
      settleTimeoutId = null;
    }
    if (interactionActive) {
      return;
    }
    interactionActive = true;
    model.set("interaction_active", true);
    model.save_changes();
  }

  function orbit(deltaX, deltaY) {
    const rotationSpeed = 0.008;
    const offset = subtract(position, target);
    const radius = Math.max(1e-3, Math.hypot(...offset));
    const upReference = viewerUpVector();
    const yawAxis = upReference;
    const orbitDeltaX = Boolean(model.get("orbit_invert_x")) ? deltaX : -deltaX;
    const orbitDeltaY = Boolean(model.get("orbit_invert_y")) ? deltaY : -deltaY;
    const yawRotation = axisAngleRotation(
      yawAxis,
      orbitDeltaX * rotationSpeed,
    );
    const yawedOffset = multiplyMat3Vec3(yawRotation, offset);
    const yawedForward = normalize(scale(yawedOffset, -1));
    const pitchAxis = normalize(cross(yawedForward, upReference));
    const pitchRotation = axisAngleRotation(
      pitchAxis,
      orbitDeltaY * rotationSpeed,
    );
    const orbitedOffset = multiplyMat3Vec3(pitchRotation, yawedOffset);
    position = add(target, orbitedOffset);
    cameraState.cam_to_world = lookAtCamera(position, target, upReference);
    orbitDistance = radius;
  }

  function pan(deltaX, deltaY) {
    const rotation = rotationFromCamToWorld(cameraState.cam_to_world);
    const right = normalize(matrixColumn(rotation, 0));
    const up = normalize(matrixColumn(rotation, 1));
    const panDeltaX = Boolean(model.get("pan_invert_x")) ? -deltaX : deltaX;
    const panDeltaY = Boolean(model.get("pan_invert_y")) ? -deltaY : deltaY;
    const scaleFactor =
      Math.max(1e-3, orbitDistance) *
      Math.tan((cameraState.fov_degrees * Math.PI / 180.0) / 2.0) /
      Math.max(1, frame.getBoundingClientRect().height) *
      2.0;
    const translation = add(
      scale(right, -panDeltaX * scaleFactor),
      scale(up, -panDeltaY * scaleFactor),
    );
    position = add(position, translation);
    target = add(target, translation);
  }

  function dolly(deltaY) {
    const zoomFactor = Math.exp(deltaY * 0.0015);
    const offset = subtract(position, target);
    orbitDistance = clamp(Math.hypot(...offset) * zoomFactor, 0.05, 1e5);
    const direction = normalize(offset);
    position = add(target, scale(direction, orbitDistance));
  }

  function stepKeyboard(deltaSeconds) {
    if (pressedKeys.size === 0) {
      return;
    }
    const rotation = rotationFromCamToWorld(cameraState.cam_to_world);
    const forward = normalize(matrixColumn(rotation, 2));
    const right = normalize(matrixColumn(rotation, 0));
    const up = normalize(matrixColumn(rotation, 1));
    const configuredMoveSpeed = Number(model.get("keyboard_move_speed")) || 0.125;
    const configuredSprintMultiplier =
      Number(model.get("keyboard_sprint_multiplier")) || 4.0;
    const sprintMultiplier =
      pressedKeys.has("shift") ? configuredSprintMultiplier : 1.0;
    const speed =
      Math.max(
        configuredMoveSpeed,
        orbitDistance * configuredMoveSpeed * 16.0,
      ) *
      sprintMultiplier *
      deltaSeconds;
    let motion = [0, 0, 0];
    if (pressedKeys.has("w")) motion = add(motion, scale(forward, speed));
    if (pressedKeys.has("s")) motion = add(motion, scale(forward, -speed));
    if (pressedKeys.has("a")) motion = add(motion, scale(right, -speed));
    if (pressedKeys.has("d")) motion = add(motion, scale(right, speed));
    if (pressedKeys.has("q")) motion = add(motion, scale(up, -speed));
    if (pressedKeys.has("e")) motion = add(motion, scale(up, speed));
    position = add(position, motion);
    target = add(target, motion);
    pushCameraState();
  }

  function tick(timestamp) {
    if (lastTickMs === null) {
      lastTickMs = timestamp;
    }
    const deltaSeconds = Math.min(0.05, (timestamp - lastTickMs) / 1000);
    lastTickMs = timestamp;
    stepKeyboard(deltaSeconds);
    if (pressedKeys.size > 0) {
      animationFrame = requestAnimationFrame(tick);
    } else {
      animationFrame = null;
    }
  }

  function ensureKeyboardLoop() {
    if (animationFrame === null) {
      lastTickMs = null;
      animationFrame = requestAnimationFrame(tick);
    }
  }

  function registerFrameMetrics(revision, renderTimeMs, shouldReset) {
    let latencySampleMs = null;
    const sentAtMs = revisionSentAtMs.get(revision);
    if (sentAtMs !== undefined) {
      const now = performance.now();
      const latencyMs = Math.max(0.0, now - sentAtMs);
      latencySampleMs = latencyMs;
      averageLatencyMs =
        averageLatencyMs === null || shouldReset
          ? latencyMs
          : averageLatencyMs * 0.85 + latencyMs * 0.15;
      lastLatencySampleAtMs = now;
      revisionSentAtMs.delete(revision);
    }
    if (typeof renderTimeMs === "number" && Number.isFinite(renderTimeMs)) {
      lastRenderTimeMs = renderTimeMs;
    }
    for (const pendingRevision of revisionSentAtMs.keys()) {
      if (pendingRevision < revision) {
        revisionSentAtMs.delete(pendingRevision);
      }
    }
    if (latencySampleMs !== null) {
      lastLatencySampleMs = latencySampleMs;
    }
    if (typeof renderTimeMs === "number" && Number.isFinite(renderTimeMs)) {
      smoothedRenderTimeMs = smoothMetric(
        smoothedRenderTimeMs,
        renderTimeMs,
        shouldReset,
      );
    }
    updateLatencyBadge();
    syncMetricsToModel();
  }

  async function drawFrame(
    bytes,
    width,
    height,
    revision,
    renderTimeMs,
    interactionActiveFrame,
    mimeType,
    messageReceivedAtMs,
    backendFrameSentPerfTimeMs,
  ) {
    latestScheduledFrameRevision = Math.max(latestScheduledFrameRevision, revision);
    const decodeEnqueueAt = performance.now();
    const shouldReset =
      lastLatencySampleAtMs === null
      || decodeEnqueueAt - lastLatencySampleAtMs > 1000.0;
    if (revision < latestScheduledFrameRevision || revision < lastFrameRevision) {
      return;
    }
    const blob = new Blob([bytes], { type: mimeType || "image/jpeg" });
    const decodeStartedAt = performance.now();
    const bitmap = await createImageBitmap(blob);
    const decodeFinishedAt = performance.now();
    if (revision < latestScheduledFrameRevision || revision < lastFrameRevision) {
      bitmap.close();
      return;
    }
    const drawStartedAt = performance.now();
    frame.width = width;
    frame.height = height;
    frameContext.clearRect(0, 0, width, height);
    frameContext.drawImage(bitmap, 0, 0, width, height);
    drawAxesGizmo();
    drawHorizon();
    bitmap.close();
    lastFrameRevision = revision;
    const nowMs = performance.now();
    recentDrawTimestamps.push(nowMs);
    const cutoff = nowMs - 1000.0;
    while (recentDrawTimestamps.length > 0 && recentDrawTimestamps[0] < cutoff) {
      recentDrawTimestamps.shift();
    }
    viewerFps = recentDrawTimestamps.length;
    updateLatencyBadge();
    const drawFinishedAt = performance.now();
    if (interactionActiveFrame) {
      const receiveQueueTimeMs = decodeEnqueueAt - messageReceivedAtMs;
      lastReceiveQueueTimeMs = receiveQueueTimeMs;
      lastPacketSizeBytes = bytes.byteLength;
      await new Promise((resolve) => {
        requestAnimationFrame(() => resolve(performance.now()));
      }).then((presentedAtNow) => {
        lastPresentWaitTimeMs = presentedAtNow - drawFinishedAt;
      });
      lastDecodeTimeMs = decodeFinishedAt - decodeStartedAt;
      lastDrawTimeMs = drawFinishedAt - drawStartedAt;
      lastPostReceiveTimeMs = performance.now() - messageReceivedAtMs;
      const shouldReset =
        lastLatencySampleAtMs === null
        || performance.now() - lastLatencySampleAtMs > 1000.0;
      if (
        backendClockOffsetMs !== null
        && typeof backendFrameSentPerfTimeMs === "number"
        && Number.isFinite(backendFrameSentPerfTimeMs)
      ) {
        const estimatedClientSentAtMs =
          backendFrameSentPerfTimeMs - backendClockOffsetMs;
        lastBackendToBrowserTimeMs = Math.max(
          0.0,
          messageReceivedAtMs - estimatedClientSentAtMs,
        );
        smoothedBackendToBrowserTimeMs = smoothMetric(
          smoothedBackendToBrowserTimeMs,
          lastBackendToBrowserTimeMs,
          shouldReset,
        );
      }
      smoothedDecodeTimeMs = smoothMetric(
        smoothedDecodeTimeMs,
        lastDecodeTimeMs,
        shouldReset,
      );
      smoothedDrawTimeMs = smoothMetric(
        smoothedDrawTimeMs,
        lastDrawTimeMs,
        shouldReset,
      );
      smoothedPresentWaitTimeMs = smoothMetric(
        smoothedPresentWaitTimeMs,
        lastPresentWaitTimeMs,
        shouldReset,
      );
      smoothedReceiveQueueTimeMs = smoothMetric(
        smoothedReceiveQueueTimeMs,
        receiveQueueTimeMs,
        shouldReset,
      );
      smoothedPostReceiveTimeMs = smoothMetric(
        smoothedPostReceiveTimeMs,
        lastPostReceiveTimeMs,
        shouldReset,
      );
      smoothedPacketSizeBytes = smoothMetric(
        smoothedPacketSizeBytes,
        lastPacketSizeBytes,
        shouldReset,
      );
      registerFrameMetrics(revision, renderTimeMs, shouldReset);
      if (interactionActive) {
        pushCameraState();
      }
      return;
    }
    revisionSentAtMs.delete(revision);
  }

  function handleFramePacket(rawPacket, messageReceivedAtMs) {
    const packet = parseFramePacket(rawPacket);
    if (packet === null) {
      return;
    }
    if (
      typeof packet.header.revision === "number"
      && packet.header.revision % 30 === 0
      && model.get("transport_mode") === "websocket"
    ) {
      sendClockSyncPing();
    }
    void drawFrame(
      packet.payload,
      packet.header.width ?? 0,
      packet.header.height ?? 0,
      packet.header.revision ?? -1,
      packet.header.render_time_ms,
      Boolean(packet.header.interaction_active),
      packet.header.mime_type,
      messageReceivedAtMs,
      packet.header.backend_frame_sent_perf_time_ms,
    );
  }

  function scheduleReconnect() {
    if (closed || reconnectTimeoutId !== null) {
      return;
    }
    reconnectTimeoutId = setTimeout(() => {
      reconnectTimeoutId = null;
      connectFrameStream();
    }, 250);
  }

  function sendClockSyncPing() {
    if (streamSocket === null || streamSocket.readyState !== WebSocket.OPEN) {
      return;
    }
    const pingId = nextClockSyncPingId;
    nextClockSyncPingId += 1;
    const clientSentAtMs = performance.now();
    pendingClockSyncPings.set(pingId, clientSentAtMs);
    streamSocket.send(JSON.stringify({
      type: "clock_sync_ping",
      ping_id: pingId,
      client_sent_at_ms: clientSentAtMs,
    }));
  }

  function disconnectFrameStream() {
    if (reconnectTimeoutId !== null) {
      clearTimeout(reconnectTimeoutId);
      reconnectTimeoutId = null;
    }
    pendingClockSyncPings.clear();
    if (streamSocket !== null) {
      const socket = streamSocket;
      streamSocket = null;
      socket.onopen = null;
      socket.onmessage = null;
      socket.onerror = null;
      socket.onclose = null;
      if (
        socket.readyState === WebSocket.OPEN
        || socket.readyState === WebSocket.CONNECTING
      ) {
        socket.close();
      }
    }
  }

  function connectFrameStream() {
    disconnectFrameStream();
    if (model.get("transport_mode") !== "websocket") {
      return;
    }
    const streamPort = Number(model.get("stream_port"));
    const streamPath = model.get("stream_path");
    const streamToken = model.get("stream_token");
    if (!Number.isFinite(streamPort) || streamPort <= 0 || !streamPath || !streamToken) {
      return;
    }
    const streamUrl =
      `ws://${window.location.hostname}:${streamPort}${streamPath}?token=${encodeURIComponent(streamToken)}`;
    const socket = new WebSocket(streamUrl);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {
      sendClockSyncPing();
    };
    socket.onmessage = (event) => {
      if (typeof event.data === "string") {
        let message = null;
        try {
          message = JSON.parse(event.data);
        } catch (_error) {
          return;
        }
        if (message.type !== "clock_sync_pong") {
          return;
        }
        const pingId = Number(message.ping_id);
        const clientSentAtMs = pendingClockSyncPings.get(pingId);
        if (clientSentAtMs === undefined) {
          return;
        }
        pendingClockSyncPings.delete(pingId);
        const clientReceivedAtMs = performance.now();
        const serverReceivedAtMs = Number(message.server_received_at_ms);
        if (!Number.isFinite(serverReceivedAtMs)) {
          return;
        }
        const rttMs = clientReceivedAtMs - clientSentAtMs;
        const offsetMs =
          serverReceivedAtMs - ((clientSentAtMs + clientReceivedAtMs) / 2.0);
        if (bestClockSyncRttMs === null || rttMs < bestClockSyncRttMs) {
          bestClockSyncRttMs = rttMs;
          backendClockOffsetMs = offsetMs;
        } else if (backendClockOffsetMs === null) {
          backendClockOffsetMs = offsetMs;
        } else {
          backendClockOffsetMs = backendClockOffsetMs * 0.9 + offsetMs * 0.1;
        }
        return;
      }
      handleFramePacket(event.data, performance.now());
    };
    socket.onerror = () => {
      scheduleReconnect();
    };
    socket.onclose = () => {
      if (streamSocket === socket) {
        streamSocket = null;
      }
      scheduleReconnect();
    };
    streamSocket = socket;
  }

  function applyCameraStateJson() {
    const incoming = model.get("camera_state_json");
    if (incoming === serializeCameraState()) {
      return;
    }
    cameraState = parseCameraState(incoming);
    updatePoseFromMatrix();
    updateAspectRatio();
    const forward = matrixColumn(cameraState.cam_to_world, 2);
    target = add(position, scale(forward, orbitDistance));
    drawAxesGizmo();
    drawHorizon();
  }

  frame.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });

  frame.addEventListener("pointerdown", (event) => {
    markInteractionActive();
    frame.focus();
    frame.setPointerCapture(event.pointerId);
    interaction = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      lastX: event.clientX,
      lastY: event.clientY,
      button: event.button,
      moved: false,
      mode: event.button === 2 ? "pan" : "orbit",
    };
    frame.classList.add("is-dragging");
  });

  frame.addEventListener("pointermove", (event) => {
    if (!interaction || interaction.pointerId !== event.pointerId) {
      return;
    }
    const deltaX = event.clientX - interaction.lastX;
    const deltaY = event.clientY - interaction.lastY;
    const dragDistance = Math.hypot(
      event.clientX - interaction.startX,
      event.clientY - interaction.startY,
    );
    if (dragDistance > clickThresholdPixels) {
      interaction.moved = true;
    }
    interaction.lastX = event.clientX;
    interaction.lastY = event.clientY;
    if (interaction.mode === "orbit") {
      orbit(deltaX, deltaY);
    } else {
      pan(deltaX, deltaY);
    }
    pushCameraState();
  });

  function endInteraction(event) {
    if (!interaction || interaction.pointerId !== event.pointerId) {
      return;
    }
    const shouldRegisterClick =
      interaction.button === 0 && !interaction.moved;
    if (shouldRegisterClick) {
      const rect = frame.getBoundingClientRect();
      const normalizedX = rect.width > 0
        ? (event.clientX - rect.left) / rect.width
        : 0.0;
      const normalizedY = rect.height > 0
        ? (event.clientY - rect.top) / rect.height
        : 0.0;
      const clickPayload = {
        x: Math.max(
          0,
          Math.min(
            cameraState.width - 1,
            Math.floor(normalizedX * cameraState.width),
          ),
        ),
        y: Math.max(
          0,
          Math.min(
            cameraState.height - 1,
            Math.floor(normalizedY * cameraState.height),
          ),
        ),
        width: cameraState.width,
        height: cameraState.height,
        camera_state: JSON.parse(serializeCameraState()),
      };
      model.set("last_click_json", JSON.stringify(clickPayload));
      model.save_changes();
    }
    frame.releasePointerCapture(event.pointerId);
    interaction = null;
    frame.classList.remove("is-dragging");
    scheduleSettledRender();
  }

  frame.addEventListener("pointerup", endInteraction);
  frame.addEventListener("pointercancel", endInteraction);

  frame.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      markInteractionActive();
      dolly(event.deltaY);
      pushCameraState();
      scheduleSettledRender();
    },
    { passive: false },
  );

  frame.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if (!["w", "a", "s", "d", "q", "e", "shift"].includes(key)) {
      return;
    }
    event.preventDefault();
    markInteractionActive();
    pressedKeys.add(key);
    ensureKeyboardLoop();
  });

  frame.addEventListener("keyup", (event) => {
    pressedKeys.delete(event.key.toLowerCase());
    if (pressedKeys.size === 0) {
      scheduleSettledRender();
    }
  });

  frame.addEventListener("blur", () => {
    pressedKeys.clear();
    scheduleSettledRender();
  });

  const resizeObserver = new ResizeObserver(() => {
    pushCameraState();
  });
  resizeObserver.observe(root);

  const onCameraChange = () => applyCameraStateJson();
  const onAspectRatioChange = () => {
    updateAspectRatio();
    pushCameraState();
  };
  const onInteractionActiveChange = () => {
    const wasActive = interactionActive;
    interactionActive = Boolean(model.get("interaction_active"));
    if (interactionActive && !wasActive) {
      scheduleSettledRender();
    }
  };
  const onStreamConfigChange = () => {
    connectFrameStream();
  };
  const onFramePacketChange = () => {
    if (model.get("transport_mode") !== "comm") {
      return;
    }
    handleFramePacket(model.get("frame_packet"), performance.now());
  };
  const onRenderFpsChange = () => {
    renderFps = Number(model.get("render_fps")) || null;
    updateLatencyBadge();
  };
  const onShowAxesChange = () => {
    drawAxesGizmo();
  };
  const onShowHorizonChange = () => {
    drawHorizon();
  };
  const onShowOriginChange = () => {
    drawHorizon();
  };
  const onShowStatsChange = () => {
    updateLatencyBadge();
  };
  const onViewerRotationChange = () => {
    drawAxesGizmo();
    drawHorizon();
  };
  const onOriginChange = () => {
    drawHorizon();
  };

  model.on("change:camera_state_json", onCameraChange);
  model.on("change:aspect_ratio", onAspectRatioChange);
  model.on("change:interaction_active", onInteractionActiveChange);
  model.on("change:stream_port", onStreamConfigChange);
  model.on("change:stream_path", onStreamConfigChange);
  model.on("change:stream_token", onStreamConfigChange);
  model.on("change:transport_mode", onStreamConfigChange);
  model.on("change:frame_packet", onFramePacketChange);
  model.on("change:render_fps", onRenderFpsChange);
  model.on("change:show_axes", onShowAxesChange);
  model.on("change:show_horizon", onShowHorizonChange);
  model.on("change:show_origin", onShowOriginChange);
  model.on("change:show_stats", onShowStatsChange);
  model.on("change:viewer_rotation_x_degrees", onViewerRotationChange);
  model.on("change:viewer_rotation_y_degrees", onViewerRotationChange);
  model.on("change:viewer_rotation_z_degrees", onViewerRotationChange);
  model.on("change:origin_x", onOriginChange);
  model.on("change:origin_y", onOriginChange);
  model.on("change:origin_z", onOriginChange);

  updateAspectRatio();
  updateLatencyBadge();
  drawAxesGizmo();
  drawHorizon();
  onFramePacketChange();
  connectFrameStream();
  pushCameraState();

  return () => {
    resizeObserver.disconnect();
    if (animationFrame !== null) {
      cancelAnimationFrame(animationFrame);
    }
    closed = true;
    disconnectFrameStream();
    model.off("change:camera_state_json", onCameraChange);
    model.off("change:aspect_ratio", onAspectRatioChange);
    model.off("change:interaction_active", onInteractionActiveChange);
    model.off("change:stream_port", onStreamConfigChange);
    model.off("change:stream_path", onStreamConfigChange);
    model.off("change:stream_token", onStreamConfigChange);
    model.off("change:transport_mode", onStreamConfigChange);
    model.off("change:frame_packet", onFramePacketChange);
    model.off("change:render_fps", onRenderFpsChange);
    model.off("change:show_axes", onShowAxesChange);
    model.off("change:show_horizon", onShowHorizonChange);
    model.off("change:show_origin", onShowOriginChange);
    model.off("change:show_stats", onShowStatsChange);
    model.off("change:viewer_rotation_x_degrees", onViewerRotationChange);
    model.off("change:viewer_rotation_y_degrees", onViewerRotationChange);
    model.off("change:viewer_rotation_z_degrees", onViewerRotationChange);
    model.off("change:origin_x", onOriginChange);
    model.off("change:origin_y", onOriginChange);
    model.off("change:origin_z", onOriginChange);
    if (settleTimeoutId !== null) {
      clearTimeout(settleTimeoutId);
    }
  };
}

const widget = { render };

export { render };
export default widget;
