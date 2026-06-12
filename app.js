import { findTargetCrossing, toCsv } from "./simulation.mjs";

const form = document.querySelector("#model-form");
const runButton = document.querySelector("#run-button");
const errorMessage = document.querySelector("#error-message");
const emptyState = document.querySelector("#empty-state");
const progressCard = document.querySelector("#progress-card");
const progressBar = document.querySelector("#progress-bar");
const progressLabel = document.querySelector("#progress-label");
const resultContent = document.querySelector("#result-content");
const downloadButton = document.querySelector("#download-button");

let worker = null;
let currentData = [];
let latestElapsedMs = 0;

const alphaPresets = {
  pcc: [0.3, 0.12, 0, 1],
  "fly-ash": [0.6, 0.15, 0, 1],
  bfs: [0.45, 0.2, 0, 1],
  atmospheric: [0.65, 0.12, 0, 1],
};

const dxPresets = {
  zero: [0, 0, 0, 0, true],
  beta_submerged: [8.9, 5.6, 0, 50, true],
  beta_tidal: [10, 5, 0, 50, false],
};

function element(id) {
  return document.getElementById(id);
}

function numberValue(id) {
  const value = Number(element(id).value);
  if (!Number.isFinite(value)) throw new Error(`Enter a valid value for ${id}.`);
  return value;
}

function setNumber(id, value) {
  element(id).value = String(value);
}

function setDisabled(ids, disabled) {
  ids.forEach((id) => {
    element(id).disabled = disabled;
  });
}

function updateAlphaPreset() {
  const preset = alphaPresets[element("alpha_preset").value];
  const ids = ["alpha_mu", "alpha_sd", "alpha_L", "alpha_U"];

  if (preset) {
    ids.forEach((id, index) => setNumber(id, preset[index]));
    setDisabled(ids, true);
  } else {
    setDisabled(ids, false);
  }
}

function updateDxPreset() {
  const preset = dxPresets[element("dx_mode").value];
  const ids = ["dx_mu", "dx_sd", "dx_L", "dx_U"];

  if (!preset) {
    ids.forEach((id) => setNumber(id, 0));
    setDisabled(ids, true);
    return;
  }

  ids.forEach((id, index) => setNumber(id, preset[index]));
  setDisabled(ids, preset[4]);
}

function syncTemperature(sourceId, targetId, offset) {
  const source = element(sourceId);
  source.addEventListener("input", () => {
    const value = Number(source.value);
    if (Number.isFinite(value)) setNumber(targetId, (value + offset).toFixed(2));
  });
}

function collectModel() {
  const alphaPreset = element("alpha_preset").value;
  const dxMode = element("dx_mode").value;
  const tStart = numberValue("t_start");
  const tEnd = numberValue("t_end");

  if (!alphaPreset) throw new Error("Please choose an α preset or Custom values.");
  if (!dxMode) throw new Error("Please choose a Δx exposure mode.");
  if (!(tEnd > tStart)) {
    throw new Error("Plot end time must be greater than the display start time.");
  }

  const params = {
    Cs_mu: numberValue("Cs_mu"),
    Cs_sd: numberValue("Cs_sd"),
    alpha_mu: numberValue("alpha_mu"),
    alpha_sd: numberValue("alpha_sd"),
    alpha_L: numberValue("alpha_L"),
    alpha_U: numberValue("alpha_U"),
    D0_mu: numberValue("D0_mu"),
    D0_sd: Math.max(0.2 * numberValue("D0_mu"), 0),
    cover_mu: numberValue("cover_mu"),
    cover_sd: numberValue("cover_sd"),
    Ccrit_mu: numberValue("Ccrit_mu"),
    Ccrit_sd: numberValue("Ccrit_sd"),
    Ccrit_L: numberValue("Ccrit_L"),
    Ccrit_U: numberValue("Ccrit_U"),
    be_mu: numberValue("be_mu"),
    be_sd: numberValue("be_sd"),
    Treal_mu: numberValue("Treal_mu_K"),
    Treal_sd: numberValue("Treal_sd_K"),
    t0: numberValue("t0"),
    Tref: numberValue("Tref_K"),
    C0: numberValue("C0"),
    dx_mode: dxMode,
    dx_mu: numberValue("dx_mu"),
    dx_sd: numberValue("dx_sd"),
    dx_L: numberValue("dx_L"),
    dx_U: numberValue("dx_U"),
  };

  const options = {
    samples: Math.trunc(numberValue("samples")),
    seed: Math.trunc(numberValue("seed")),
    tStart: 0,
    tEnd,
    timePoints: Math.trunc(numberValue("t_points")),
  };

  return { params, options, displayStart: tStart };
}

function setRunning(running) {
  runButton.disabled = running;
  runButton.querySelector("span").textContent = running ? "Calculating…" : "Run simulation";
  progressCard.hidden = !running;

  if (running) {
    emptyState.hidden = true;
    resultContent.hidden = true;
    downloadButton.disabled = true;
    progressBar.style.width = "0%";
    progressLabel.textContent = "Preparing random samples…";
  }
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
  setRunning(false);
  if (!currentData.length) emptyState.hidden = false;
}

function hideError() {
  errorMessage.hidden = true;
  errorMessage.textContent = "";
}

function tickValues(minimum, maximum, step) {
  if (!(step > 0) || !(maximum > minimum)) return [minimum, maximum];
  const values = [];
  const first = Math.ceil(minimum / step) * step;
  for (let value = first; value <= maximum + step * 1e-8; value += step) {
    values.push(Number(value.toFixed(10)));
    if (values.length > 200) break;
  }
  return values;
}

function svgNode(name, attributes = {}, text = null) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  if (text !== null) node.textContent = text;
  return node;
}

function formatAxis(value) {
  const absolute = Math.abs(value);
  if (absolute > 0 && absolute < 0.001) return value.toExponential(1);
  if (absolute >= 1000) return value.toLocaleString();
  return Number(value.toFixed(3)).toString();
}

function createChart({
  container,
  data,
  tEnd,
  yMin,
  yMax,
  yTick,
  xTick,
  series,
  rightAxis = null,
  target = null,
  crossing = null,
  yLabel,
}) {
  container.replaceChildren();
  const width = 1000;
  const height = 440;
  const margin = { top: 26, right: rightAxis ? 72 : 30, bottom: 58, left: 70 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": container.getAttribute("aria-label") || "Simulation chart",
  });

  const xScale = (value) => margin.left + (value / tEnd) * plotWidth;
  const yScale = (value) =>
    margin.top + ((yMax - value) / Math.max(yMax - yMin, Number.EPSILON)) * plotHeight;

  const plotGroup = svgNode("g");
  const gridColor = "#e5ece8";
  const axisColor = "#9eada8";

  for (const tick of tickValues(yMin, yMax, yTick)) {
    const y = yScale(tick);
    plotGroup.append(
      svgNode("line", {
        x1: margin.left,
        y1: y,
        x2: width - margin.right,
        y2: y,
        stroke: gridColor,
        "stroke-width": 1,
      }),
      svgNode(
        "text",
        { x: margin.left - 12, y: y + 4, "text-anchor": "end" },
        formatAxis(tick),
      ),
    );
  }

  for (const tick of tickValues(0, tEnd, xTick)) {
    const x = xScale(tick);
    plotGroup.append(
      svgNode("line", {
        x1: x,
        y1: margin.top,
        x2: x,
        y2: height - margin.bottom,
        stroke: gridColor,
        "stroke-width": 1,
      }),
      svgNode(
        "text",
        { x, y: height - margin.bottom + 24, "text-anchor": "middle" },
        formatAxis(tick),
      ),
    );
  }

  plotGroup.append(
    svgNode("line", {
      x1: margin.left,
      y1: height - margin.bottom,
      x2: width - margin.right,
      y2: height - margin.bottom,
      stroke: axisColor,
    }),
    svgNode("line", {
      x1: margin.left,
      y1: margin.top,
      x2: margin.left,
      y2: height - margin.bottom,
      stroke: axisColor,
    }),
    svgNode(
      "text",
      { x: margin.left + plotWidth / 2, y: height - 12, "text-anchor": "middle" },
      "Time (yr)",
    ),
    svgNode(
      "text",
      {
        x: 18,
        y: margin.top + plotHeight / 2,
        transform: `rotate(-90 18 ${margin.top + plotHeight / 2})`,
        "text-anchor": "middle",
      },
      yLabel,
    ),
  );

  if (target !== null && target >= yMin && target <= yMax) {
    const targetY = yScale(target);
    plotGroup.append(
      svgNode("line", {
        x1: margin.left,
        y1: targetY,
        x2: width - margin.right,
        y2: targetY,
        stroke: "#c84b45",
        "stroke-width": 1.5,
        "stroke-dasharray": "7 5",
      }),
    );

    if (crossing !== null) {
      const crossingX = xScale(crossing);
      plotGroup.append(
        svgNode("line", {
          x1: crossingX,
          y1: margin.top,
          x2: crossingX,
          y2: height - margin.bottom,
          stroke: "#c84b45",
          "stroke-width": 1,
          "stroke-dasharray": "3 5",
          opacity: 0.65,
        }),
        svgNode("circle", {
          cx: crossingX,
          cy: targetY,
          r: 4.5,
          fill: "#fff",
          stroke: "#c84b45",
          "stroke-width": 2,
        }),
      );

      const labelWidth = 130;
      const labelX = Math.min(Math.max(crossingX - labelWidth / 2, margin.left), width - margin.right - labelWidth);
      const labelY = Math.max(targetY - 38, margin.top + 4);
      plotGroup.append(
        svgNode("rect", {
          x: labelX,
          y: labelY,
          width: labelWidth,
          height: 27,
          rx: 7,
          fill: "#fff7f5",
          stroke: "#e6b2ad",
        }),
        svgNode(
          "text",
          {
            x: labelX + labelWidth / 2,
            y: labelY + 18,
            "text-anchor": "middle",
            fill: "#9a3935",
          },
          `Target: ${crossing.toFixed(2)} yr`,
        ),
      );
    }
  }

  for (const item of series) {
    const points = data
      .map((row) => `${xScale(row.t_years).toFixed(2)},${yScale(row[item.key]).toFixed(2)}`)
      .join(" ");
    plotGroup.append(
      svgNode("polyline", {
        points,
        fill: "none",
        stroke: item.color,
        "stroke-width": item.width ?? 2.5,
        "stroke-linejoin": "round",
        "stroke-linecap": "round",
        "stroke-dasharray": item.dash ?? "",
      }),
    );
  }

  if (rightAxis) {
    const rightScale = (value) =>
      margin.top +
      ((rightAxis.max - value) / Math.max(rightAxis.max - rightAxis.min, Number.EPSILON)) *
        plotHeight;
    const points = data
      .map((row) => `${xScale(row.t_years).toFixed(2)},${rightScale(row[rightAxis.key]).toFixed(2)}`)
      .join(" ");

    plotGroup.append(
      svgNode("line", {
        x1: width - margin.right,
        y1: margin.top,
        x2: width - margin.right,
        y2: height - margin.bottom,
        stroke: axisColor,
      }),
      svgNode("polyline", {
        points,
        fill: "none",
        stroke: rightAxis.color,
        "stroke-width": 2,
        "stroke-dasharray": "7 5",
        "stroke-linejoin": "round",
      }),
    );

    for (const tick of tickValues(rightAxis.min, rightAxis.max, rightAxis.tick)) {
      plotGroup.append(
        svgNode(
          "text",
          {
            x: width - margin.right + 12,
            y: rightScale(tick) + 4,
            "text-anchor": "start",
          },
          formatAxis(tick),
        ),
      );
    }

    plotGroup.append(
      svgNode(
        "text",
        {
          x: width - 14,
          y: margin.top + plotHeight / 2,
          transform: `rotate(90 ${width - 14} ${margin.top + plotHeight / 2})`,
          "text-anchor": "middle",
        },
        rightAxis.label,
      ),
    );
  }

  svg.append(plotGroup);
  container.append(svg);
}

function chartConfiguration() {
  const betaMin = numberValue("beta_min");
  const betaMax = numberValue("beta_max");
  const pfMin = numberValue("pf_min");
  const pfMax = numberValue("pf_max");
  if (!(betaMax > betaMin)) throw new Error("β maximum must be greater than β minimum.");
  if (!(pfMax > pfMin)) throw new Error("Pf maximum must be greater than Pf minimum.");

  return {
    tEnd: numberValue("t_end"),
    betaMin,
    betaMax,
    betaTick: numberValue("beta_tick"),
    pfMin,
    pfMax,
    pfTick: numberValue("pf_tick"),
    xTick: numberValue("x_tick"),
    betaTarget: numberValue("beta_target"),
    showTarget: element("show_beta_target").checked,
    pfMode: element("pf_mode").value,
  };
}

function renderResults() {
  if (!currentData.length) return;
  const config = chartConfiguration();
  const displayStart = numberValue("t_start");
  const visibleData = currentData.filter(
    (row) => row.t_years >= displayStart && row.t_years <= config.tEnd,
  );
  if (!visibleData.length) throw new Error("No result points fall inside the display window.");

  const last = visibleData.at(-1);
  const crossing = config.showTarget
    ? findTargetCrossing(visibleData, config.betaTarget)
    : null;

  element("metric-beta").textContent = last.beta.toFixed(3);
  element("metric-pf").textContent =
    last.Pf < 0.001 ? last.Pf.toExponential(2) : last.Pf.toFixed(4);
  element("metric-crossing").textContent =
    crossing === null ? "Not reached" : `${crossing.toFixed(2)} yr`;
  element("metric-time").textContent =
    latestElapsedMs < 1000
      ? `${latestElapsedMs.toFixed(0)} ms`
      : `${(latestElapsedMs / 1000).toFixed(2)} s`;

  createChart({
    container: element("beta-chart"),
    data: visibleData,
    tEnd: config.tEnd,
    yMin: config.betaMin,
    yMax: config.betaMax,
    yTick: config.betaTick,
    xTick: config.xTick,
    series: [{ key: "beta", color: "#0b756d", width: 3 }],
    rightAxis:
      config.pfMode === "overlay"
        ? {
            key: "Pf",
            min: config.pfMin,
            max: config.pfMax,
            tick: config.pfTick,
            color: "#d9982f",
            label: "Failure probability Pf(t)",
          }
        : null,
    target: config.showTarget ? config.betaTarget : null,
    crossing,
    yLabel: "Reliability index β(-)",
  });

  const separatePf = config.pfMode === "separate";
  element("pf-card").hidden = !separatePf;
  element("pf-legend").hidden = separatePf;
  element("target-legend").hidden = !config.showTarget;

  if (separatePf) {
    createChart({
      container: element("pf-chart"),
      data: visibleData,
      tEnd: config.tEnd,
      yMin: config.pfMin,
      yMax: config.pfMax,
      yTick: config.pfTick,
      xTick: config.xTick,
      series: [{ key: "Pf", color: "#d9982f", width: 3 }],
      yLabel: "Failure probability Pf(t)",
    });
  }

  emptyState.hidden = true;
  progressCard.hidden = true;
  resultContent.hidden = false;
  downloadButton.disabled = false;
}

function startSimulation(model) {
  if (worker) worker.terminate();
  worker = new Worker(new URL("./simulation.worker.js", import.meta.url), {
    type: "module",
  });

  setRunning(true);
  worker.addEventListener("message", (event) => {
    const message = event.data;
    if (message.type === "progress") {
      const percent = Math.min(Math.round(message.progress * 100), 100);
      progressBar.style.width = `${percent}%`;
      progressLabel.textContent = `${percent}% complete`;
      return;
    }

    if (message.type === "complete") {
      currentData = message.data;
      latestElapsedMs = message.elapsedMs;
      setRunning(false);
      renderResults();
      worker.terminate();
      worker = null;
      element("results").scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }

    if (message.type === "error") {
      showError(message.message);
      worker.terminate();
      worker = null;
    }
  });

  worker.addEventListener("error", (event) => {
    showError(event.message || "The simulation worker failed.");
    worker?.terminate();
    worker = null;
  });

  worker.postMessage({ params: model.params, options: model.options });
}

element("alpha_preset").addEventListener("change", updateAlphaPreset);
element("dx_mode").addEventListener("change", updateDxPreset);
element("t0_preset").addEventListener("change", (event) => {
  if (event.target.value) setNumber("t0", event.target.value);
});
element("D0_mu").addEventListener("input", () => {
  const mean = Number(element("D0_mu").value);
  if (Number.isFinite(mean)) setNumber("D0_sd", Math.max(0.2 * mean, 0).toFixed(4));
});

syncTemperature("Treal_mu_C", "Treal_mu_K", 273.15);
syncTemperature("Treal_mu_K", "Treal_mu_C", -273.15);
syncTemperature("Tref_C", "Tref_K", 273.15);
syncTemperature("Tref_K", "Tref_C", -273.15);
syncTemperature("Treal_sd_C", "Treal_sd_K", 0);
syncTemperature("Treal_sd_K", "Treal_sd_C", 0);

form.addEventListener("submit", (event) => {
  event.preventDefault();
  hideError();
  try {
    startSimulation(collectModel());
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
  }
});

downloadButton.addEventListener("click", () => {
  if (!currentData.length) return;
  const displayStart = numberValue("t_start");
  const tEnd = numberValue("t_end");
  const visibleData = currentData.filter(
    (row) => row.t_years >= displayStart && row.t_years <= tEnd,
  );
  const blob = new Blob([toCsv(visibleData)], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "fib_output_window.csv";
  anchor.click();
  URL.revokeObjectURL(url);
});

[
  "pf_mode",
  "show_beta_target",
  "beta_target",
  "x_tick",
  "beta_min",
  "beta_max",
  "beta_tick",
  "pf_min",
  "pf_max",
  "pf_tick",
  "t_start",
].forEach((id) => {
  element(id).addEventListener("change", () => {
    if (!currentData.length) return;
    hideError();
    try {
      renderResults();
    } catch (error) {
      showError(error instanceof Error ? error.message : String(error));
    }
  });
});
