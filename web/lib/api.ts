import type { DiamondInput, PredictResult, Summary, Trace } from "./types";

/**
 * The FastAPI server runs locally during development and is reachable at
 * PYTHON_API_URL in production. In the browser the calls go to Next route
 * handlers under /api so the browser never needs CORS access to Python.
 *
 * Server Components run on the server, where a relative fetch has no base URL
 * to resolve against. There we talk to the Python server directly, which also
 * saves a network hop.
 */

const PYTHON_API = process.env.PYTHON_API_URL ?? "http://127.0.0.1:8000";
const TIMEOUT_MS = 20000;

const isServer = typeof window === "undefined";

function endpoint(path: string): string {
  return isServer ? `${PYTHON_API}${path}` : path;
}

async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail[0]?.msg) return String(detail[0].msg);
    return `Request failed (${res.status})`;
  } catch {
    return `Request failed (${res.status})`;
  }
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(endpoint(path), {
    signal: AbortSignal.timeout(TIMEOUT_MS),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json() as Promise<T>;
}

export function fetchSummary(): Promise<Summary> {
  return getJson<Summary>("/summary");
}

export function fetchSample(): Promise<DiamondInput & {
  actual_price: number;
  actual_segment: string;
}> {
  return getJson("/sample");
}

/**
 * The per-stone walkthrough. Always fetched from the browser, never during
 * server render, because it depends on what the reader has dialled in.
 */
export async function fetchTrace(input: {
  carat: number;
  cut: string;
  color: string;
  clarity: string;
}): Promise<Trace> {
  const qs = new URLSearchParams({
    carat: String(input.carat),
    cut: input.cut,
    color: input.color,
    clarity: input.clarity,
  });
  const res = await fetch(`/api/trace?${qs}`, {
    signal: AbortSignal.timeout(TIMEOUT_MS),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function predictDiamond(input: DiamondInput): Promise<PredictResult> {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

/** Chart PNGs are served by Python. Proxied so the browser stays same-origin. */
export function chartUrl(name: string): string {
  return `/api/charts/${name}`;
}
