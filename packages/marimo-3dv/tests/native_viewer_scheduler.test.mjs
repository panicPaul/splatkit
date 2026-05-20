import assert from "node:assert/strict";
import test from "node:test";

import { createInteractiveRevisionScheduler } from "../src/marimo_3dv/viewer/assets/native_viewer.js";

function makeSchedulerHarness({
  backpressure = true,
  frameIntervalMs = 0.0,
} = {}) {
  let cameraJson = "camera-0";
  let revision = 0;
  let nowMs = 1000.0;
  const sent = [];
  const timers = [];
  const adaptiveResetSamples = [];
  let clearedInteractionCount = 0;

  const scheduler = createInteractiveRevisionScheduler({
    currentCameraStateJson: () => cameraJson,
    sendInteractiveCameraStateJson: (nextJson) => {
      revision += 1;
      sent.push({ kind: "interactive", revision, cameraJson: nextJson });
      return revision;
    },
    sendNoninteractiveCameraStateJson: (nextJson) => {
      revision += 1;
      sent.push({ kind: "noninteractive", revision, cameraJson: nextJson });
      return revision;
    },
    sendSettledCameraStateJson: (nextJson) => {
      revision += 1;
      sent.push({ kind: "settled", revision, cameraJson: nextJson });
      return revision;
    },
    interactiveBackpressureEnabled: () => backpressure,
    interactiveFrameIntervalMs: () => frameIntervalMs,
    now: () => nowMs,
    setTimer: (callback, delayMs) => {
      const timer = { callback, delayMs, cleared: false };
      timers.push(timer);
      return timer;
    },
    clearTimer: (timer) => {
      timer.cleared = true;
    },
    clearInteractionActive: () => {
      clearedInteractionCount += 1;
    },
    resetAdaptiveFpsIfStale: (sampleNowMs) => {
      adaptiveResetSamples.push(sampleNowMs);
    },
  });

  return {
    adaptiveResetSamples,
    scheduler,
    sent,
    timers,
    get clearedInteractionCount() {
      return clearedInteractionCount;
    },
    get cameraJson() {
      return cameraJson;
    },
    set cameraJson(nextJson) {
      cameraJson = nextJson;
    },
    get nowMs() {
      return nowMs;
    },
    set nowMs(nextNowMs) {
      nowMs = nextNowMs;
    },
  };
}

test("queued settled render waits for the interactive revision", () => {
  const harness = makeSchedulerHarness();

  harness.cameraJson = "drag-camera";
  harness.scheduler.requestInteractiveCameraState();
  harness.cameraJson = "settled-camera";
  harness.scheduler.requestSettledRender();

  assert.deepEqual(harness.sent, [
    { kind: "interactive", revision: 1, cameraJson: "drag-camera" },
  ]);
  assert.equal(
    harness.scheduler.stateForTesting().pendingSettledRender,
    true,
  );

  harness.scheduler.completeRevision(1);

  assert.deepEqual(harness.sent, [
    { kind: "interactive", revision: 1, cameraJson: "drag-camera" },
    { kind: "settled", revision: 2, cameraJson: "settled-camera" },
  ]);
  assert.equal(
    harness.scheduler.stateForTesting().pendingSettledRender,
    false,
  );
  assert.equal(
    harness.scheduler.stateForTesting().interactiveInFlightRevision,
    null,
  );
});

test("queued settled render clears interaction before completion", () => {
  const harness = makeSchedulerHarness();

  harness.cameraJson = "drag-camera";
  harness.scheduler.requestInteractiveCameraState();
  harness.cameraJson = "settled-camera";
  harness.scheduler.requestSettledRender();

  assert.deepEqual(harness.sent, [
    { kind: "interactive", revision: 1, cameraJson: "drag-camera" },
  ]);
  assert.equal(harness.clearedInteractionCount, 1);
  assert.equal(
    harness.scheduler.stateForTesting().pendingSettledRender,
    true,
  );

  harness.scheduler.completeRevision(1);

  assert.deepEqual(harness.sent, [
    { kind: "interactive", revision: 1, cameraJson: "drag-camera" },
    { kind: "settled", revision: 2, cameraJson: "settled-camera" },
  ]);
  assert.equal(harness.clearedInteractionCount, 1);
});

test("revision completion drains a pending interactive camera", () => {
  const harness = makeSchedulerHarness();

  harness.cameraJson = "drag-camera-1";
  harness.scheduler.requestInteractiveCameraState();
  harness.cameraJson = "drag-camera-2";
  harness.scheduler.requestInteractiveCameraState();

  assert.deepEqual(harness.sent, [
    { kind: "interactive", revision: 1, cameraJson: "drag-camera-1" },
  ]);
  assert.equal(
    harness.scheduler.stateForTesting().interactiveInFlightRevision,
    1,
  );

  harness.scheduler.completeRevision(1);

  assert.deepEqual(harness.sent, [
    { kind: "interactive", revision: 1, cameraJson: "drag-camera-1" },
    { kind: "interactive", revision: 2, cameraJson: "drag-camera-2" },
  ]);
  assert.equal(
    harness.scheduler.stateForTesting().interactiveInFlightRevision,
    2,
  );
});

test("error revision completion unblocks a pending settled render", () => {
  const harness = makeSchedulerHarness();

  harness.cameraJson = "drag-before-error";
  harness.scheduler.requestInteractiveCameraState();
  harness.cameraJson = "settled-after-error";
  harness.scheduler.requestSettledRender();

  harness.scheduler.completeRevision(1);

  assert.deepEqual(harness.sent, [
    { kind: "interactive", revision: 1, cameraJson: "drag-before-error" },
    { kind: "settled", revision: 2, cameraJson: "settled-after-error" },
  ]);
});

test("no-frame revision completion unblocks a pending settled render", () => {
  const harness = makeSchedulerHarness();

  harness.cameraJson = "drag-before-drop";
  harness.scheduler.requestInteractiveCameraState();
  harness.cameraJson = "settled-after-drop";
  harness.scheduler.requestSettledRender();

  harness.scheduler.completeRevision(1);

  assert.deepEqual(harness.sent, [
    { kind: "interactive", revision: 1, cameraJson: "drag-before-drop" },
    { kind: "settled", revision: 2, cameraJson: "settled-after-drop" },
  ]);
  assert.equal(
    harness.scheduler.stateForTesting().interactiveInFlightRevision,
    null,
  );
});
