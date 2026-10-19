#!/usr/bin/env node
/* Generate evidence charts for the 27 Sep–02 Oct 2026 RA-QMIX update. */

const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const ROOT = path.resolve(__dirname, "..", "..");
const OUT = __dirname;
const C = {
  cream: "#F1EFE6", ink: "#2B2C30", red: "#D76A5B", teal: "#4F8C85",
  blue: "#587AA5", gold: "#C99B45", gray: "#A8A69E", grid: "#D9D6CB",
  greenZone: "#DCE9DF",
};

const esc = (s) => String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const txt = (x, y, s, size = 30, weight = 400, anchor = "start", color = C.ink) =>
  `<text x="${x}" y="${y}" font-family="Arial, Helvetica, sans-serif" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}" fill="${color}">${esc(s)}</text>`;
const line = (x1, y1, x2, y2, stroke = C.ink, width = 2, dash = "") =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${width}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
const rect = (x, y, w, h, fill, rx = 0, stroke = "none") =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}"/>`;

function summary(run) {
  return JSON.parse(fs.readFileSync(path.join(ROOT, "lab_results", run, "summary.json"), "utf8"));
}

async function writePng(name, body) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">${rect(0, 0, 1600, 900, C.cream)}${body}</svg>`;
  fs.writeFileSync(path.join(OUT, name.replace(/\.png$/, ".svg")), svg);
  await sharp(Buffer.from(svg)).png().toFile(path.join(OUT, name));
}

async function ablation() {
  const data = summary("ra_qmix_leave_one_out_5seed_full_cuda_20260927T095633Z");
  const scenarios = ["in_distribution", "early_failure", "slow_service", "combined_pressure"];
  const contrasts = [
    ["edge_ablation_minus_full", "Remove edge", C.red],
    ["queue_ablation_minus_full", "Remove queue mixer", C.gold],
    ["cf_ablation_minus_full", "Remove CF loss", C.teal],
    ["qmix_minus_full", "Standard QMIX", C.blue],
  ];
  const rows = new Map();
  for (const r of data.paired_differences) {
    if (r.row_type === "across_training_seeds" && r.budget === 50000) rows.set(`${r.contrast}|${r.scenario}`, r.delta_mean);
  }
  let b = txt(110, 95, "Full RA-QMIX beats QMIX on average, but component effects change sign", 40, 700);
  b += txt(110, 138, "Positive values favor Full RA-QMIX. Five training seeds, 50k episodes.", 23, 400);
  const left = 145, top = 220, width = 1370, height = 500, ymin = -20, ymax = 20;
  const y = (v) => top + (ymax - v) / (ymax - ymin) * height;
  for (const v of [-20, -10, 0, 10, 20]) {
    b += line(left, y(v), left + width, y(v), v === 0 ? C.ink : C.grid, v === 0 ? 2 : 1);
    b += txt(left - 18, y(v) + 8, v, 20, 400, "end");
  }
  const groupW = width / scenarios.length, barW = 60;
  const labels = ["Nominal", "Early failure", "Slow service", "Combined pressure"];
  scenarios.forEach((scenario, si) => {
    const cx = left + groupW * (si + 0.5);
    contrasts.forEach(([key, , color], ci) => {
      const value = rows.get(`${key}|${scenario}`);
      const x = cx + (ci - 1.5) * (barW + 8) - barW / 2;
      const y0 = y(0), yv = y(value);
      b += rect(x, Math.min(y0, yv), barW, Math.max(2, Math.abs(y0 - yv)), color, 3);
      if (Math.abs(value) >= 3) b += txt(x + barW / 2, value >= 0 ? yv - 10 : yv + 27, value.toFixed(1), 18, 700, "middle");
    });
    b += txt(cx, 765, labels[si], 23, 600, "middle");
  });
  contrasts.forEach(([, label, color], i) => {
    const x = 200 + (i % 2) * 420, yy = 820 + Math.floor(i / 2) * 40;
    b += rect(x, yy - 18, 26, 18, color, 2) + txt(x + 40, yy, label, 20, 500);
  });
  await writePng("01_ablation_deltas.png", b);
}

async function tradeoff() {
  const points = [
    ["Domain randomization", 33.54, 0.008, C.red],
    ["Fixed 87.5% nominal", 10.44, 0.542, C.gold],
    ["Curriculum + replay", -1.66, 0.607, C.teal],
    ["Curriculum + KL", -5.68, 0.550, C.blue],
  ];
  let b = txt(110, 95, "Curriculum training resolves the stress–nominal trade-off", 40, 700);
  b += txt(110, 138, "The green region satisfies the pre-specified mean nominal gate of 5% or less.", 23, 400);
  const left = 170, top = 210, width = 1290, height = 520, xmin = -10, xmax = 38, ymin = -0.05, ymax = 0.7;
  const x = (v) => left + (v - xmin) / (xmax - xmin) * width;
  const y = (v) => top + (ymax - v) / (ymax - ymin) * height;
  b += rect(x(-10), top, x(5) - x(-10), height, C.greenZone);
  for (const v of [0, 0.2, 0.4, 0.6]) {
    b += line(left, y(v), left + width, y(v), C.grid, 1);
    b += txt(left - 18, y(v) + 8, v.toFixed(1), 20, 400, "end");
  }
  for (const v of [-10, 0, 10, 20, 30]) b += txt(x(v), 770, `${v}%`, 20, 400, "middle");
  b += line(x(5), top, x(5), top + height, C.gray, 2, "10 8");
  b += txt(815, 835, "Nominal degradation (%)  |  lower is better", 24, 600, "middle");
  b += `<text x="45" y="500" transform="rotate(-90 45 500)" font-family="Arial" font-size="24" font-weight="600" text-anchor="middle" fill="${C.ink}">Stress max-regret improvement  |  higher is better</text>`;
  points.forEach(([label, nominal, stress, color]) => {
    b += `<circle cx="${x(nominal)}" cy="${y(stress)}" r="17" fill="${color}" stroke="${C.ink}" stroke-width="2"/>`;
    const dy = label === "Curriculum + KL" ? 36 : -24;
    b += txt(x(nominal) + 18, y(stress) + dy, label, 23, 700);
  });
  await writePng("02_training_tradeoff.png", b);
}

async function curriculum() {
  const data = summary("ra_qmix_anchor_preservation_full_cpu_20260930T075415Z");
  const across = {};
  for (const r of data.primary_results) if (r.row_type === "across_training_seeds") across[r.candidate_regime] = r;
  const keys = ["fixed_87_5", "curriculum_replay", "curriculum_kl_anchor"];
  const labels = ["Fixed mix", "Curriculum + replay", "Curriculum + KL"];
  let b = txt(110, 95, "Only the curriculum arms pass both preservation gates", 40, 700);
  b += txt(110, 138, "Development panel of five training seeds. Confirmation uses ten fresh seeds.", 23, 400);
  const left = 170, top = 220, width = 1280, height = 500;
  const y = (v) => top + (5.6 - v) / 5.6 * height;
  for (let v = 0; v <= 5; v++) {
    b += line(left, y(v), left + width, y(v), v === 4 ? C.ink : C.grid, v === 4 ? 2 : 1, v === 4 ? "10 8" : "");
    b += txt(left - 18, y(v) + 8, v, 20, 400, "end");
  }
  b += txt(left + width - 5, y(4) - 12, "Gate: 4/5", 20, 700, "end");
  const groupW = width / 3, barW = 135;
  keys.forEach((key, i) => {
    const cx = left + groupW * (i + 0.5);
    const vals = [across[key].stress_better_count, across[key].nominal_within_10pct_count];
    [C.teal, C.gold].forEach((color, j) => {
      const xx = cx + (j === 0 ? -barW - 12 : 12), yy = y(vals[j]);
      b += rect(xx, yy, barW, y(0) - yy, color, 5);
      b += txt(xx + barW / 2, yy - 15, `${vals[j]}/5`, 25, 700, "middle");
    });
    b += txt(cx, 770, labels[i], 24, 600, "middle");
  });
  b += rect(390, 825, 28, 20, C.teal, 2) + txt(435, 844, "Stress better than control", 22, 500);
  b += rect(850, 825, 28, 20, C.gold, 2) + txt(895, 844, "Nominal within 10%", 22, 500);
  await writePng("03_curriculum_seed_gates.png", b);
}

Promise.all([ablation(), tradeoff(), curriculum()]).catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
