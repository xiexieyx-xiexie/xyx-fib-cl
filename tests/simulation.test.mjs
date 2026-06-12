import assert from "node:assert/strict";
import test from "node:test";

import {
  erfc,
  findTargetCrossing,
  normalQuantile,
  runFibChloride,
  SeededRandom,
  toCsv,
} from "../simulation.mjs";

test("seeded random generator is reproducible", () => {
  const first = new SeededRandom(42);
  const second = new SeededRandom(42);
  const valuesA = Array.from({ length: 10 }, () => first.uniform());
  const valuesB = Array.from({ length: 10 }, () => second.uniform());
  assert.deepEqual(valuesA, valuesB);
});

test("normal quantile and erfc match known reference values", () => {
  assert.ok(Math.abs(normalQuantile(0.5)) < 1e-12);
  assert.ok(Math.abs(normalQuantile(0.975) - 1.959964) < 1e-5);
  assert.ok(Math.abs(erfc(0) - 1) < 1e-7);
  assert.ok(Math.abs(erfc(1) - 0.157299) < 1e-5);
});

test("simulation returns finite probabilities and reliability values", () => {
  const params = {
    Cs_mu: 1.8,
    Cs_sd: 0.3,
    alpha_mu: 0.3,
    alpha_sd: 0.12,
    alpha_L: 0,
    alpha_U: 1,
    D0_mu: 10,
    D0_sd: 2,
    cover_mu: 50,
    cover_sd: 7,
    Ccrit_mu: 0.6,
    Ccrit_sd: 0.15,
    Ccrit_L: 0.2,
    Ccrit_U: 2,
    be_mu: 4800,
    be_sd: 700,
    Treal_mu: 296.15,
    Treal_sd: 3,
    t0: 0.0767,
    Tref: 296.15,
    C0: 0,
    dx_mode: "zero",
    dx_mu: 0,
    dx_sd: 0,
    dx_L: 0,
    dx_U: 0,
  };

  const data = runFibChloride(params, {
    samples: 2000,
    seed: 42,
    tStart: 0,
    tEnd: 50,
    timePoints: 20,
  });

  assert.equal(data.length, 20);
  assert.ok(data.every((row) => row.Pf > 0 && row.Pf < 1));
  assert.ok(data.every((row) => Number.isFinite(row.beta)));
  assert.ok(data.at(-1).Pf >= data[0].Pf);
});

test("target crossing and CSV export work", () => {
  const data = [
    { t_years: 0, Pf: 0.01, beta: 4.2 },
    { t_years: 10, Pf: 0.03, beta: 3.4 },
  ];
  assert.ok(Math.abs(findTargetCrossing(data, 3.8) - 5) < 1e-10);
  assert.match(toCsv(data), /^t_years,Pf,beta\n/);
});
