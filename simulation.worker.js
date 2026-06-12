import { runFibChloride } from "./simulation.mjs";

self.addEventListener("message", (event) => {
  const startedAt = performance.now();

  try {
    const data = runFibChloride(event.data.params, event.data.options, (progress) => {
      self.postMessage({ type: "progress", progress });
    });

    self.postMessage({
      type: "complete",
      data,
      elapsedMs: performance.now() - startedAt,
    });
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
});
