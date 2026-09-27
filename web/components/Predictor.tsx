"use client";

/**
 * The live predictor. Full control over every input the models take, so a
 * reader can check the model's claim directly: change one grade, watch the
 * price move, then read why in the trace above.
 */

import { useState } from "react";
import { predictDiamond, fetchSample } from "@/lib/api";
import type { DiamondInput, PredictResult, Segment } from "@/lib/types";
import type { ReactNode } from "react";

const CUTS = ["Fair", "Good", "Very Good", "Premium", "Ideal"];
const COLORS = ["J", "I", "H", "G", "F", "E", "D"];
const CLARITIES = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"];
const SEGMENTS: Segment[] = ["Budget", "Mid-range", "Premium", "Luxury"];

const DEFAULTS: DiamondInput = {
  carat: 1.2,
  cut: "Very Good",
  color: "G",
  clarity: "VS2",
  depth: 62.0,
  table: 57.0,
  x: 6.7,
  y: 6.7,
  z: 4.2,
};

function money(v: number) {
  return `$${Math.round(v).toLocaleString("en-US")}`;
}

export function Predictor() {
  const [input, setInput] = useState<DiamondInput>(DEFAULTS);
  const [out, setOut] = useState<PredictResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setOut(await predictDiamond(input));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed.");
      setOut(null);
    } finally {
      setBusy(false);
    }
  }

  async function loadReal() {
    setBusy(true);
    setError(null);
    try {
      const s = await fetchSample();
      setInput({
        carat: s.carat,
        cut: s.cut,
        color: s.color,
        clarity: s.clarity,
        depth: s.depth,
        table: s.table,
        x: s.x,
        y: s.y,
        z: s.z,
      });
      setOut(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load a sample.");
    } finally {
      setBusy(false);
    }
  }

  const set = <K extends keyof DiamondInput>(k: K, v: DiamondInput[K]) =>
    setInput((i) => ({ ...i, [k]: v }));

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
      {/* --- inputs ---------------------------------------------------- */}
      <div className="panel p-6 md:p-8">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <p className="eyebrow">Every input the models take</p>
          <button
            type="button"
            onClick={loadReal}
            disabled={busy}
            className="btn btn-ghost"
          >
            Load a real stone
          </button>
        </div>

        {/* weight */}
        <div className="mt-7">
          <div className="flex items-baseline justify-between">
            <label htmlFor="p-carat" className="eyebrow">
              Carat
            </label>
            <span className="num text-sm" style={{ color: "var(--accent)" }}>
              {input.carat.toFixed(2)} ct
            </span>
          </div>
          <input
            id="p-carat"
            type="range"
            min={0.2}
            max={5}
            step={0.05}
            value={input.carat}
            onChange={(e) => set("carat", Number(e.target.value))}
            className="mt-2"
          />
        </div>

        {/* grades */}
        <div className="mt-7 grid grid-cols-3 gap-3">
          <Select
            id="p-cut"
            label="Cut"
            opts={CUTS}
            value={input.cut}
            onChange={(v) => set("cut", v)}
          />
          <Select
            id="p-color"
            label="Colour"
            opts={COLORS}
            value={input.color}
            onChange={(v) => set("color", v)}
          />
          <Select
            id="p-clarity"
            label="Clarity"
            opts={CLARITIES}
            value={input.clarity}
            onChange={(v) => set("clarity", v)}
          />
        </div>

        {/* measurements */}
        <p className="eyebrow mt-8">Measurements in millimetres</p>
        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
          {(
            [
              ["x", "Length", 2, 20],
              ["y", "Width", 2, 20],
              ["z", "Depth", 1, 15],
            ] as const
          ).map(([k, label, min, max]) => (
            <NumField
              key={k}
              id={`p-${k}`}
              label={label}
              value={input[k]}
              min={min}
              max={max}
              step={0.01}
              onChange={(v) => set(k, v)}
            />
          ))}
          {(
            [
              ["depth", "Depth %", 30, 90],
              ["table", "Table %", 40, 80],
            ] as const
          ).map(([k, label, min, max]) => (
            <NumField
              key={k}
              id={`p-${k}`}
              label={label}
              value={input[k]}
              min={min}
              max={max}
              step={0.1}
              onChange={(v) => set(k, v)}
            />
          ))}
        </div>

        <div className="mt-8 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={run}
            disabled={busy}
            className="btn btn-primary"
          >
            {busy ? "Predicting" : "Predict price"}
          </button>
          <button
            type="button"
            onClick={() => {
              setInput(DEFAULTS);
              setOut(null);
              setError(null);
            }}
            disabled={busy}
            className="btn btn-ghost"
          >
            Reset
          </button>
        </div>
        {error ? (
          <p className="prose-note mt-4" style={{ color: "var(--accent)" }}>
            {error}
          </p>
        ) : null}
      </div>

      {/* --- output ----------------------------------------------------- */}
      <div className="lg:sticky lg:top-24 lg:self-start">
        {!out ? (
          <div className="panel flex h-full min-h-[22rem] flex-col items-center justify-center p-8 text-center">
            <p className="eyebrow">No prediction yet</p>
            <p className="prose-note mt-4 max-w-[34ch]">
              Set the stone and press predict. The model returns a price, a
              price per carat, a value tier with its confidence, and which
              size cluster it falls into.
            </p>
          </div>
        ) : (
          <div className="panel p-6 md:p-8">
            <p className="eyebrow">Predicted price</p>
            <p
              className="num mt-3 text-5xl leading-none md:text-6xl"
              style={{ color: "var(--accent)" }}
            >
              {money(out.predicted_price)}
            </p>
            <p className="prose-note mt-3">
              {money(out.price_per_carat)} per carat
            </p>

            <div className="mt-7">
              <p className="index mb-3">Value tier</p>
              <p className="display display-sm">{out.predicted_segment}</p>
              <ul className="mt-4 space-y-2">
                {SEGMENTS.map((s) => {
                  const p = out.segment_probabilities[s] ?? 0;
                  const on = s === out.predicted_segment;
                  return (
                    <li key={s}>
                      <div className="flex justify-between text-xs">
                        <span
                          className="num"
                          style={{ color: on ? "var(--accent)" : "var(--ink-faint)" }}
                        >
                          {s}
                        </span>
                        <span
                          className="num"
                          style={{ color: on ? "var(--accent)" : "var(--ink-faint)" }}
                        >
                          {(p * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div
                        className="mt-1 h-1.5 rounded-full"
                        style={{ background: "var(--line-dark)" }}
                      >
                        <div
                          className="h-1.5 rounded-full"
                          style={{
                            width: `${Math.max(p * 100, 0.5)}%`,
                            background: on
                              ? "var(--accent)"
                              : "var(--ink-faint)",
                          }}
                        />
                      </div>
                    </li>
                  );
                })}
              </ul>
            </div>

            {out.cluster_profile ? (
              <div className="mt-7">
                <p className="index mb-3">
                  Size cluster {out.cluster_profile.cluster}
                </p>
                <p className="prose-note">
                  {out.cluster_profile.n.toLocaleString("en-US")} stones,
                  median {out.cluster_profile.median_carat} ct, median{" "}
                  {money(out.cluster_profile.median_price_per_carat)} per carat.
                  The typical stone here is {out.cluster_profile.modal_cut},{" "}
                  {out.cluster_profile.modal_color},{" "}
                  {out.cluster_profile.modal_clarity}.
                </p>
                <p className="caption mt-3">
                  This cluster is about size, not value. It is not a price tier
                  and does not claim to be.
                </p>
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}

function Select({
  id,
  label,
  opts,
  value,
  onChange,
}: {
  id: string;
  label: string;
  opts: readonly string[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div>
      <label htmlFor={id} className="eyebrow mb-2 block">
        {label}
      </label>
      <select
        id={id}
        className="field"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {opts.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </div>
  );
}

function NumField({
  id,
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <label htmlFor={id} className="eyebrow mb-2 block">
        {label}
      </label>
      <input
        id={id}
        type="number"
        className="field num"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (Number.isFinite(n)) onChange(n);
        }}
      />
    </div>
  );
}
