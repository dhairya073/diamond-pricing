/**
 * Browser model inference.
 *
 * The trained scikit-learn artifacts are 787 MB, so they cannot be deployed.
 * Instead export_web.py writes depth-capped forests as flat arrays and this
 * module walks them in JavaScript. A random forest prediction is just tree
 * traversal, so no server is involved.
 *
 * Accuracy of the exported model, measured on the same held-out split:
 *   regressor  R2 0.9808  MAE $292   (full model: 0.9824 / $262)
 *   classifier 91.35% accuracy        (full model: 93.34%)
 *
 * The reference dataset is only fetched when the walkthrough needs it, so the
 * predictor alone costs about 7 MB instead of 9.4 MB.
 */

import type { DiamondInput, Segment, Trace } from "./types";
import summaryJson from "@/data/summary.json";

export const CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"];
export const COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"];
export const CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"];
export const SEGMENTS: Segment[] = ["Budget", "Mid-range", "Premium", "Luxury"];

/** Numeric features the regressor expects, in order. */
const FEATS = [
  "carat", "depth", "table", "x", "y", "z",
  "volume", "carat_per_volume", "log_carat",
  "cut", "color", "clarity",
] as const;

type Tree = {
  f: number[]; // split feature per node, -2 at a leaf
  t: number[]; // split threshold per node
  l: number[]; // left child, -1 at a leaf
  r: number[]; // right child, -1 at a leaf
  v: number[] | number[][]; // leaf value(s)
};

type Forest = {
  trees: Tree[];
  n_features: number;
  classes?: number[];
  segments?: string[];
};

type KMeans = {
  features: string[];
  mean: number[];
  scale: number[];
  centroids: number[][];
};

type Reference = {
  fields: string[];
  cut_order: string[];
  color_order: string[];
  clarity_order: string[];
  rows: number[][];
};

type Core = { reg: Forest; clf: Forest; kmeans: KMeans };

let corePromise: Promise<Core> | null = null;
let refPromise: Promise<Reference> | null = null;

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url} (${res.status})`);
  return res.json() as Promise<T>;
}

/** Regressor, classifier and clusterer. Fetched once, then cached. */
export function loadCore(): Promise<Core> {
  corePromise ??= Promise.all([
    getJson<Forest>("/model/reg.json"),
    getJson<Forest>("/model/clf.json"),
    getJson<KMeans>("/model/kmeans.json"),
  ]).then(([reg, clf, kmeans]) => ({ reg, clf, kmeans }));
  return corePromise;
}

/** The 53,772 cleaned stones, for the walkthrough's real comparisons. */
export function loadReference(): Promise<Reference> {
  refPromise ??= getJson<Reference>("/model/reference.json");
  return refPromise;
}

/* ------------------------------------------------------------------ maths */

function mean(a: number[]): number {
  let s = 0;
  for (const v of a) s += v;
  return s / a.length;
}

function median(a: number[]): number {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/* --------------------------------------------------------------- inference */

/** Walk one tree to its leaf and return that leaf's mean price. */
function leafValue(tree: Tree, x: number[]): number {
  let node = 0;
  while (tree.l[node] !== -1) {
    node = x[tree.f[node]] <= tree.t[node] ? tree.l[node] : tree.r[node];
  }
  return tree.v[node] as number;
}

/** Which leaf each tree chooses. Used to explain the averaging. */
function leafPath(tree: Tree, x: number[]): number {
  let node = 0;
  while (tree.l[node] !== -1) {
    node = x[tree.f[node]] <= tree.t[node] ? tree.l[node] : tree.r[node];
  }
  return node;
}

/** sklearn's predict_proba: the mean of the per-tree class distributions. */
function treeClassProbs(tree: Tree, x: number[]): number[] {
  const node = leafPath(tree, x);
  const counts = tree.v[node] as number[];
  let total = 0;
  for (const c of counts) total += c;
  return counts.map((c) => (total > 0 ? c / total : 0));
}

function forestClassProbs(forest: Forest, x: number[]): number[] {
  const nClasses = forest.segments?.length ?? 4;
  const acc = new Array<number>(nClasses).fill(0);
  for (const tree of forest.trees) {
    const p = treeClassProbs(tree, x);
    for (let i = 0; i < nClasses; i++) acc[i] += p[i] ?? 0;
  }
  return acc.map((v) => v / forest.trees.length);
}

function kmeansCluster(km: KMeans, input: DiamondInput): number {
  const raw: Record<string, number> = {
    carat: input.carat,
    volume: input.x * input.y * input.z,
    depth: input.depth,
    table: input.table,
  };
  const z = km.features.map(
    (f, i) => (raw[f] - km.mean[i]) / (km.scale[i] || 1),
  );
  let best = 0;
  let bestDist = Infinity;
  km.centroids.forEach((c, i) => {
    let d = 0;
    for (let j = 0; j < z.length; j++) d += (z[j] - c[j]) ** 2;
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  });
  return best;
}

/** Build the ordered numeric row the forests expect. */
export function featurize(input: DiamondInput): number[] {
  const volume = input.x * input.y * input.z;
  const row: Record<string, number> = {
    carat: input.carat,
    depth: input.depth,
    table: input.table,
    x: input.x,
    y: input.y,
    z: input.z,
    volume,
    carat_per_volume: volume ? input.carat / volume : 0,
    log_carat: Math.log1p(input.carat),
    cut: CUT_ORDER.indexOf(input.cut),
    color: COLOR_ORDER.indexOf(input.color),
    clarity: CLARITY_ORDER.indexOf(input.clarity),
  };
  return FEATS.map((f) => row[f]);
}

export type CorePrediction = {
  predicted_price: number;
  price_per_carat: number;
  predicted_segment: Segment;
  segment_probabilities: Record<Segment, number>;
  cluster: number;
  /** Every tree's own answer, so the average can be shown rather than claimed. */
  tree_answers: number[];
};

/** Price, tier and cluster, entirely in the browser. */
export function predictCore(core: Core, input: DiamondInput): CorePrediction {
  const x = featurize(input);
  const treeAnswers = core.reg.trees.map((t) => leafValue(t, x));
  const price = mean(treeAnswers);
  const probs = forestClassProbs(core.clf, x);
  const bestIdx = probs.indexOf(Math.max(...probs));
  const segment_probabilities = Object.fromEntries(
    SEGMENTS.map((s, i) => [s, Number((probs[i] ?? 0).toFixed(4))]),
  ) as Record<Segment, number>;

  return {
    predicted_price: Number(price.toFixed(2)),
    price_per_carat: Number((price / input.carat).toFixed(2)),
    predicted_segment: SEGMENTS[bestIdx],
    segment_probabilities,
    cluster: kmeansCluster(core.kmeans, input),
    tree_answers: treeAnswers,
  };
}

/* --------------------------------------------------------------- the trace */

export type TraceOptions = { carat: number; cut: string; color: string; clarity: string };

/** Nearest real stone matching a description, preferring an exact grade match. */
function pickSource(ref: Reference, o: TraceOptions): { row: number[]; group: number[][] } {
  const band = ref.rows.filter(
    (r) => r[0] >= o.carat * 0.9 && r[0] <= o.carat * 1.1,
  );
  const pool = band.length
    ? band
    : [...ref.rows].sort((a, b) => Math.abs(a[0] - o.carat) - Math.abs(b[0] - o.carat)).slice(0, 200);

  const cut = CUT_ORDER.indexOf(o.cut);
  const color = COLOR_ORDER.indexOf(o.color);
  const clarity = CLARITY_ORDER.indexOf(o.clarity);
  const exact = pool.filter((r) => r[1] === cut && r[2] === color && r[3] === clarity);
  const pick = exact.length >= 5 ? exact : pool;

  // The median-priced stone of the group, so the walkthrough is not cherry-picked.
  const med = median(pick.map((r) => r[4]));
  let best = pick[0];
  let bestD = Infinity;
  for (const r of pick) {
    const d = Math.abs(r[4] - med);
    if (d < bestD) {
      bestD = d;
      best = r;
    }
  }
  return { row: best, group: pick };
}

/**
 * An honest comparison group.
 *
 * Two rules, both learned the hard way. Match the grade, or the gap measures
 * the grade instead of accuracy. Match the weight tightly, or the gap
 * measures the weight instead: a 0.63-0.77ct window around a 0.7ct stone
 * holds 5,995 stones and its median describes something far smaller. That
 * too-wide window once produced a +97% "error" that was pure artefact.
 */
function comparisonGroup(ref: Reference, o: TraceOptions) {
  const cut = CUT_ORDER.indexOf(o.cut);
  const color = COLOR_ORDER.indexOf(o.color);
  const clarity = CLARITY_ORDER.indexOf(o.clarity);

  for (const tol of [0.01, 0.02, 0.05, 0.1, 0.2]) {
    const s = ref.rows.filter(
      (r) => Math.abs(r[0] - o.carat) <= tol && r[1] === cut && r[2] === color && r[3] === clarity,
    );
    if (s.length >= 8)
      return { rows: s, tol, label: `${o.cut} / ${o.color} / ${o.clarity}` };
  }
  for (const tol of [0.05, 0.1, 0.2]) {
    const s = ref.rows.filter(
      (r) => Math.abs(r[0] - o.carat) <= tol && r[1] === cut && r[2] === color,
    );
    if (s.length >= 8)
      return { rows: s, tol, label: `${o.cut} / ${o.color}, any clarity` };
  }
  const s = ref.rows.filter((r) => Math.abs(r[0] - o.carat) <= 0.1);
  if (s.length >= 8) return { rows: s, tol: 0.1, label: "any grade at this weight" };
  return null;
}

function quantile(a: number[], q: number): number {
  const s = [...a].sort((x, y) => x - y);
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

/** Every stage of the pipeline for one stone, with measured intermediates. */
export function buildTraceInBrowser(
  core: Core,
  ref: Reference,
  o: TraceOptions,
): Trace {
  const { row: src } = pickSource(ref, o);
  const [, , , , srcPrice, depth, table, x, y, z] = src;

  // Stage 1: the real stone whose measurements stand in for the description.
  const input: DiamondInput = {
    carat: o.carat, cut: o.cut, color: o.color, clarity: o.clarity,
    depth, table, x, y, z,
  };

  // Stage 2: engineered features.
  const volume = Number((x * y * z).toFixed(2));
  const cpv = Number((o.carat / volume).toFixed(6));
  const logCarat = Number(Math.log1p(o.carat).toFixed(4));
  const derived = {
    volume: { value: volume, formula: "x * y * z", note: `${x} x ${y} x ${z} mm` },
    carat_per_volume: {
      value: cpv, formula: "carat / volume",
      note: "compactness: how tightly the weight is packed",
    },
    log_carat: {
      value: logCarat, formula: "ln(1 + carat)",
      note: "tames the long right tail of weight",
    },
  };

  // Stage 3: ordinal encoding.
  const encoding = [
    {
      feature: "cut",
      value: `${o.cut} -> ${CUT_ORDER.indexOf(o.cut)}`,
      why: "grades are ordered, so the order is kept instead of 5 separate columns",
    },
    {
      feature: "color",
      value: `${o.color} -> ${COLOR_ORDER.indexOf(o.color)}`,
      why: `J is 0 through D being ${COLOR_ORDER.length - 1}`,
    },
    {
      feature: "clarity",
      value: `${o.clarity} -> ${CLARITY_ORDER.indexOf(o.clarity)}`,
      why: `I1 is 0 through IF being ${CLARITY_ORDER.length - 1}`,
    },
  ];

  const prediction = predictCore(core, input);
  const answers = prediction.tree_answers;

  // Stage 4: what the forest actually does.
  //
  // Each tree sends the stone to one leaf and answers with that leaf's mean
  // price. The forest averages those answers, and that average IS the
  // prediction, so the reader can check it rather than take it on trust.
  //
  // (An earlier version claimed "400 real stones landed in this same leaf".
  // That was false: reg.apply() returns leaf ids, which are not row
  // positions, so the claim described an arbitrary slice of the dataset.)
  //
  // The feature ranking is the pipeline's grouped permutation importance, not
  // a count of how often each column was split on. Split counts are not
  // importance: a 7-level column like colour gets more chances to be used,
  // and with five columns correlating above 0.99 the trees pick among them
  // almost arbitrarily. Counting splits here ranked colour first, flatly
  // contradicting the permutation result on the same page.
  const grouped = (
    summaryJson as unknown as {
      regression: { grouped: Record<string, number> };
    }
  ).regression.grouped;
  const topFeatures = Object.entries(grouped)
    .map(([feature, importance]) => ({ feature, importance }))
    .sort((a, b) => b.importance - a.importance);

  // Stage 5: reality check against a genuinely comparable group.
  const group = comparisonGroup(ref, o);
  const marketMedian = group ? Math.round(median(group.rows.map((r) => r[4]))) : null;
  const prices = group ? group.rows.map((r) => r[4]) : [];
  const errorPct =
    marketMedian !== null
      ? Number(((prediction.predicted_price / marketMedian - 1) * 100).toFixed(1))
      : null;

  // Weight alone, all three grades flattened to worst.
  const flatPrice = mean(
    core.reg.trees.map((t) =>
      leafValue(t, featurize({ ...input, cut: "Fair", color: "J", clarity: "I1" })),
    ),
  );

  return {
    input,
    derived,
    encoding,
    tree: {
      // How many trees voted, and how widely their answers spread.
      leaf_size: answers.length,
      leaf_median_price: Math.round(median(answers)),
      tree_answers: {
        count: answers.length,
        mean: Number(mean(answers).toFixed(2)),
        median: Math.round(median(answers)),
        min: Math.round(Math.min(...answers)),
        max: Math.round(Math.max(...answers)),
        std: Number(
          Math.sqrt(mean(answers.map((a) => (a - mean(answers)) ** 2))).toFixed(1),
        ),
      },
      top_features: topFeatures,
      neighbours: [],
    },
    result: {
      predicted_price: prediction.predicted_price,
      price_per_carat: prediction.price_per_carat,
      lowest_quality_price: Number(flatPrice.toFixed(2)),
      quality_premium: Number((prediction.predicted_price - flatPrice).toFixed(2)),
      predicted_segment: prediction.predicted_segment,
      segment_probabilities: prediction.segment_probabilities,
      cluster: prediction.cluster,
    },
    reality_check: {
      source_stone_price: Math.round(srcPrice),
      market_median_price: marketMedian,
      comparable_stones: group ? group.rows.length : 0,
      comparison_group: group ? group.label : "no comparable group",
      carat_tolerance: group ? group.tol : null,
      error_pct: errorPct,
      price_spread:
        group && group.rows.length
          ? ([Math.round(quantile(prices, 0.1)), Math.round(quantile(prices, 0.9))] as [
              number,
              number,
            ])
          : null,
    },
  };
}
