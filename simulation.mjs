const SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60;
const SQRT_TWO_PI = Math.sqrt(2 * Math.PI);

export class SeededRandom {
  constructor(seed = 42) {
    this.state = Number(seed) >>> 0;
    this.spareNormal = null;
  }

  uniform() {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let value = this.state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  }

  normal() {
    if (this.spareNormal !== null) {
      const value = this.spareNormal;
      this.spareNormal = null;
      return value;
    }

    let u = 0;
    let v = 0;
    while (u === 0) u = this.uniform();
    while (v === 0) v = this.uniform();

    const magnitude = Math.sqrt(-2 * Math.log(u));
    const angle = 2 * Math.PI * v;
    this.spareNormal = magnitude * Math.sin(angle);
    return magnitude * Math.cos(angle);
  }

  gamma(shape) {
    if (!(shape > 0)) {
      throw new Error("Gamma shape must be positive.");
    }

    if (shape < 1) {
      const u = Math.max(this.uniform(), Number.MIN_VALUE);
      return this.gamma(shape + 1) * Math.pow(u, 1 / shape);
    }

    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);

    while (true) {
      const x = this.normal();
      let v = 1 + c * x;
      if (v <= 0) continue;
      v *= v * v;

      const u = this.uniform();
      if (u < 1 - 0.0331 * x ** 4) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
        return d * v;
      }
    }
  }

  beta(alpha, beta) {
    const x = this.gamma(alpha);
    const y = this.gamma(beta);
    const total = x + y;
    return total > 0 && Number.isFinite(total) ? x / total : 0.5;
  }
}

export function betaShapesFromMeanSd(mean, sd) {
  const mu = Math.max(Math.min(mean, 1 - 1e-9), 1e-9);
  const variance = Math.max(sd ** 2, 1e-12);
  const scale = (mu * (1 - mu)) / variance - 1;
  return {
    alpha: Math.max(mu * scale, 1e-6),
    beta: Math.max((1 - mu) * scale, 1e-6),
  };
}

export function normalQuantile(probability) {
  if (probability <= 0) return -Infinity;
  if (probability >= 1) return Infinity;

  const a = [
    -3.969683028665376e1,
    2.209460984245205e2,
    -2.759285104469687e2,
    1.38357751867269e2,
    -3.066479806614716e1,
    2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1,
    1.615858368580409e2,
    -1.556989798598866e2,
    6.680131188771972e1,
    -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3,
    -3.223964580411365e-1,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3,
    3.224671290700398e-1,
    2.445134137142996,
    3.754408661907416,
  ];

  const low = 0.02425;
  const high = 1 - low;

  if (probability < low) {
    const q = Math.sqrt(-2 * Math.log(probability));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }

  if (probability > high) {
    const q = Math.sqrt(-2 * Math.log(1 - probability));
    return -(
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }

  const q = probability - 0.5;
  const r = q * q;
  return (
    (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
    q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  );
}

export function erfc(value) {
  const z = Math.abs(value);
  const t = 1 / (1 + 0.5 * z);
  const approximation =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t *
                          (0.27886807 +
                            t *
                              (-1.13520398 +
                                t *
                                  (1.48851587 +
                                    t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return value >= 0 ? approximation : 2 - approximation;
}

function sampleLognormal(random, mean, sd) {
  const sigmaSquared = Math.log(1 + sd ** 2 / mean ** 2);
  const logMean = Math.log(mean) - 0.5 * sigmaSquared;
  return Math.exp(logMean + Math.sqrt(sigmaSquared) * random.normal());
}

function sampleBetaInterval(random, mean, sd, lower, upper) {
  if (!(upper > lower)) {
    throw new Error("Upper bound must be greater than lower bound.");
  }

  const boundedMean = Math.max(Math.min(mean, upper - 1e-12), lower + 1e-12);
  const mean01 = (boundedMean - lower) / (upper - lower);
  const sd01 = Math.max(sd, 1e-12) / (upper - lower);
  const shapes = betaShapesFromMeanSd(mean01, sd01);
  return lower + (upper - lower) * random.beta(shapes.alpha, shapes.beta);
}

function linspace(start, end, count) {
  if (count === 1) return [start];
  const step = (end - start) / (count - 1);
  return Array.from({ length: count }, (_, index) => start + step * index);
}

function validateParams(params, options) {
  const positive = [
    ["Surface chloride mean", params.Cs_mu],
    ["DRCM0 mean", params.D0_mu],
    ["Cover mean", params.cover_mu],
    ["Critical chloride mean", params.Ccrit_mu],
    ["Temperature coefficient mean", params.be_mu],
    ["Actual temperature", params.Treal_mu],
    ["Reference temperature", params.Tref],
    ["Reference age", params.t0],
  ];

  for (const [label, value] of positive) {
    if (!(Number(value) > 0)) throw new Error(`${label} must be positive.`);
  }
  if (!(options.tEnd > 0)) throw new Error("Plot end time must be positive.");
  if (!(options.timePoints >= 10)) throw new Error("Use at least 10 time points.");
  if (!(options.samples >= 1000)) throw new Error("Use at least 1,000 samples.");
}

export function runFibChloride(params, options = {}, onProgress = null) {
  const samples = Math.trunc(options.samples ?? 100000);
  const seed = Math.trunc(options.seed ?? 42);
  const tStart = Number(options.tStart ?? 0);
  const tEnd = Number(options.tEnd ?? 50);
  const timePoints = Math.trunc(options.timePoints ?? 200);
  validateParams(params, { samples, tEnd, timePoints });

  const random = new SeededRandom(seed);
  const cs = new Float64Array(samples);
  const alpha = new Float64Array(samples);
  const ccrit = new Float64Array(samples);
  const d0 = new Float64Array(samples);
  const cover = new Float64Array(samples);
  const be = new Float64Array(samples);
  const temperature = new Float64Array(samples);
  const temperatureFactor = new Float64Array(samples);
  const dx = new Float64Array(samples);

  for (let index = 0; index < samples; index += 1) {
    cs[index] = sampleLognormal(random, params.Cs_mu, params.Cs_sd);
    alpha[index] = sampleBetaInterval(
      random,
      params.alpha_mu,
      params.alpha_sd,
      params.alpha_L,
      params.alpha_U,
    );
    ccrit[index] = sampleBetaInterval(
      random,
      params.Ccrit_mu,
      params.Ccrit_sd,
      params.Ccrit_L,
      params.Ccrit_U,
    );
    d0[index] = Math.max(params.D0_mu + params.D0_sd * random.normal(), 1e-3) * 1e-12;
    cover[index] = Math.max(params.cover_mu + params.cover_sd * random.normal(), 1) / 1000;
    be[index] = Math.max(params.be_mu + params.be_sd * random.normal(), 1);
    temperature[index] = Math.max(
      params.Treal_mu + params.Treal_sd * random.normal(),
      250,
    );
    temperatureFactor[index] = Math.exp(
      be[index] * (1 / params.Tref - 1 / temperature[index]),
    );

    if (params.dx_mode === "zero") {
      dx[index] = 0;
    } else if (
      params.dx_mode === "beta_submerged" ||
      params.dx_mode === "beta_tidal"
    ) {
      dx[index] = sampleBetaInterval(
        random,
        params.dx_mu,
        params.dx_sd,
        params.dx_L,
        params.dx_U,
      );
    } else {
      throw new Error("Choose a convection-zone mode.");
    }
  }

  const times = linspace(tStart, tEnd, timePoints);
  const result = new Array(timePoints);
  const t0Seconds = params.t0 * SECONDS_PER_YEAR;
  const progressInterval = Math.max(1, Math.floor(timePoints / 20));

  for (let timeIndex = 0; timeIndex < timePoints; timeIndex += 1) {
    const years = times[timeIndex];
    const seconds = Math.max(years * SECONDS_PER_YEAR, 1);
    let failures = 0;

    for (let sampleIndex = 0; sampleIndex < samples; sampleIndex += 1) {
      const apparentDiffusion =
        temperatureFactor[sampleIndex] *
        d0[sampleIndex] *
        Math.pow(t0Seconds / seconds, alpha[sampleIndex]);
      const argument =
        (cover[sampleIndex] - dx[sampleIndex] / 1000) /
        (2 * Math.sqrt(apparentDiffusion * seconds));
      const chlorideAtDepth = params.C0 + (cs[sampleIndex] - params.C0) * erfc(argument);
      if (chlorideAtDepth >= ccrit[sampleIndex]) failures += 1;
    }

    const probability = Math.min(Math.max(failures / samples, 1e-12), 1 - 1e-12);
    result[timeIndex] = {
      t_years: years,
      Pf: probability,
      beta: -normalQuantile(probability),
    };

    if (
      onProgress &&
      (timeIndex % progressInterval === 0 || timeIndex === timePoints - 1)
    ) {
      onProgress((timeIndex + 1) / timePoints);
    }
  }

  return result;
}

export function findTargetCrossing(data, target) {
  for (let index = 0; index < data.length - 1; index += 1) {
    const first = data[index];
    const second = data[index + 1];
    if ((first.beta - target) * (second.beta - target) <= 0) {
      const delta = second.beta - first.beta;
      if (delta === 0) return first.t_years;
      return (
        first.t_years +
        ((target - first.beta) * (second.t_years - first.t_years)) / delta
      );
    }
  }
  return null;
}

export function toCsv(data) {
  const rows = ["t_years,Pf,beta"];
  for (const row of data) {
    rows.push(
      `${row.t_years.toFixed(8)},${row.Pf.toPrecision(10)},${row.beta.toPrecision(10)}`,
    );
  }
  return `${rows.join("\n")}\n`;
}
