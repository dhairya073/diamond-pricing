import type { DiamondInput, PredictResult, Summary, Trace } from "./types";
import { buildTraceInBrowser, loadCore, loadReference, predictCore } from "./model";
import summaryJson from "@/data/summary.json";
import samplesJson from "@/data/samples.json";

/**
 * Everything here runs without a server.
 *
 * The trained models total 787 MB, so they cannot be hosted next to the site.
 * Instead export_web.py writes depth-capped forests as JSON and lib/model.ts
 * walks them in the browser. The analysis payload and the charts ship as
 * static files too, which removes a whole failure mode: there is no backend
 * that can be unreachable.
 *
 * api.py still exists and still serves the full-precision models locally for
 * anything that wants them. The site simply no longer depends on it.
 */

type Sample = DiamondInput & { actual_price: number };

const samples = samplesJson as Sample[];

/** The analysis payload is a bundled import, so server render needs no fetch. */
export function fetchSummary(): Promise<Summary> {
  return Promise.resolve(summaryJson as unknown as Summary);
}

/** A real stone, drawn from a fixed committed set rather than the full 2.3 MB. */
export function fetchSample(): Promise<Sample> {
  return Promise.resolve(samples[Math.floor(Math.random() * samples.length)]);
}

export async function predictDiamond(input: DiamondInput): Promise<PredictResult> {
  const core = await loadCore();
  const p = predictCore(core, input);
  return {
    input,
    predicted_price: p.predicted_price,
    price_per_carat: p.price_per_carat,
    predicted_segment: p.predicted_segment,
    segment_probabilities: p.segment_probabilities,
    cluster: p.cluster,
  } as unknown as PredictResult;
}

/**
 * The per-stone walkthrough. Needs the reference dataset as well as the
 * model, because it borrows a real stone's measurements and builds its
 * comparison group out of stones that are genuinely like the one asked about.
 */
export async function fetchTrace(input: {
  carat: number;
  cut: string;
  color: string;
  clarity: string;
}): Promise<Trace> {
  const [core, ref] = await Promise.all([loadCore(), loadReference()]);
  return buildTraceInBrowser(core, ref, input);
}

/** Charts are pre-rendered PNGs served straight from the static folder. */
export function chartUrl(name: string): string {
  return `/charts/${name}`;
}
