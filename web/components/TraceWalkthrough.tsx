"use client";

/**
 * The step-through walkthrough.
 *
 * Five steps, each showing what the pipeline actually computed for the stone
 * the reader has dialled in. Every number comes from GET /trace, which runs
 * the real models. Nothing here is narrated or approximated.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { fetchTrace } from "@/lib/api";
import type { Trace } from "@/lib/types";

type Choice = { carat: number; cut: string; color: string; clarity: string };

const STEPS = [
  { id: "input", label: "Describe" },
  { id: "engineer", label: "Engineer" },
  { id: "encode", label: "Encode" },
  { id: "reason", label: "Reason" },
  { id: "check", label: "Verify" },
] as const;

type StepId = (typeof STEPS)[number]["id"];

const CUTS = ["Fair", "Good", "Very Good", "Premium", "Ideal"];
const COLORS = ["J", "I", "H", "G", "F", "E", "D"];
const CLARITIES = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"];

function money(v: number) {
  return `$${Math.round(v).toLocaleString("en-US")}`;
}

export function TraceWalkthrough() {
  const [choice, setChoice] = useState<Choice>({
    carat: 1.2,
    cut: "Very Good",
    color: "G",
    clarity: "VS2",
  });
  const [step, setStep] = useState<StepId>("input");
  const [data, setData] = useState<Trace | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const reqId = useRef(0);

  const run = useCallback(async (c: Choice) => {
    const id = ++reqId.current;
    setBusy(true);
    setError(null);
    try {
      const t = await fetchTrace(c);
      // A slow earlier request must not overwrite a newer one.
      if (id === reqId.current) setData(t);
    } catch (e) {
      if (id === reqId.current) {
        setError(e instanceof Error ? e.message : "Could not run the trace.");
        setData(null);
      }
    } finally {
      if (id === reqId.current) setBusy(false);
    }
  }, []);

  // Debounced: dragging the slider should not fire 40 requests.
  useEffect(() => {
    const t = setTimeout(() => void run(choice), 220);
    return () => clearTimeout(t);
  }, [choice, run]);

  const idx = STEPS.findIndex((s) => s.id === step);

  return (
    <div>
      {/* --- the stone the reader is describing ------------------------- */}
      <div className="panel p-6 md:p-8">
        <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_auto]">
          <div>
            <label
              htmlFor="trace-carat"
              className="eyebrow mb-3 block"
            >
              Weight in carats
            </label>
            <div className="flex items-baseline gap-3">
              <span className="num text-4xl" style={{ color: "var(--accent)" }}>
                {choice.carat.toFixed(2)}
              </span>
              <span className="text-sm" style={{ color: "var(--ink-faint)" }}>
                ct
              </span>
            </div>
            <input
              id="trace-carat"
              type="range"
              min={0.2}
              max={3.5}
              step={0.05}
              value={choice.carat}
              onChange={(e) =>
                setChoice((c) => ({ ...c, carat: Number(e.target.value) }))
              }
              className="mt-3"
            />
            <div
              className="index mt-1 flex justify-between"
              aria-hidden="true"
            >
              <span>0.20</span>
              <span>3.50</span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 lg:w-[26rem]">
            {(
              [
                ["cut", "Cut", CUTS],
                ["color", "Colour", COLORS],
                ["clarity", "Clarity", CLARITIES],
              ] as const
            ).map(([key, label, opts]) => (
              <div key={key}>
                <label htmlFor={`trace-${key}`} className="eyebrow mb-2 block">
                  {label}
                </label>
                <select
                  id={`trace-${key}`}
                  className="field"
                  value={choice[key]}
                  onChange={(e) =>
                    setChoice((c) => ({ ...c, [key]: e.target.value }))
                  }
                >
                  {opts.map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* --- the stepper ------------------------------------------------ */}
      <div className="mt-8 flex flex-wrap items-center gap-x-2 gap-y-3">
        {STEPS.map((s, i) => (
          <div key={s.id} className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setStep(s.id)}
              aria-current={s.id === step ? "step" : undefined}
              className="group flex items-center gap-2"
            >
              <span
                className="step-dot"
                data-on={i <= idx}
                aria-hidden="true"
              />
              <span
                className="num text-xs transition-colors"
                style={{
                  color:
                    s.id === step ? "var(--accent)" : "var(--ink-faint)",
                }}
              >
                {String(i + 1).padStart(2, "0")} {s.label}
              </span>
            </button>
            {i < STEPS.length - 1 ? (
              <span
                aria-hidden="true"
                className="inline-block h-px w-5"
                style={{ background: "var(--line-dark)" }}
              />
            ) : null}
          </div>
        ))}
      </div>

      {/* --- the active step -------------------------------------------- */}
      <div className="panel mt-5 p-6 md:p-8">
        {error ? (
          <p className="prose-note" style={{ color: "var(--accent)" }}>
            {error}
          </p>
        ) : !data ? (
          <p className="prose-note">Running the models on this stone.</p>
        ) : (
          <div className={busy ? "opacity-60 transition-opacity" : "transition-opacity"}>
            {step === "input" ? <StepInput data={data} /> : null}
            {step === "engineer" ? <StepEngineer data={data} /> : null}
            {step === "encode" ? <StepEncode data={data} /> : null}
            {step === "reason" ? <StepReason data={data} /> : null}
            {step === "check" ? <StepCheck data={data} /> : null}
          </div>
        )}
      </div>

      {/* --- the answer, always visible so the walkthrough has a target -- */}
      {data && !error ? (
        <div className="panel-raised mt-5 p-6 md:p-8">
          <p className="eyebrow">The number it produces</p>
          <p
            className="num mt-3 text-5xl leading-none md:text-6xl"
            style={{ color: "var(--accent)" }}
          >
            {money(data.result.predicted_price)}
          </p>
          <p className="prose-note mt-4 max-w-[58ch]">
            {money(data.result.price_per_carat)} per carat, placed in the{" "}
            <strong style={{ color: "var(--cream)" }}>
              {data.result.predicted_segment}
            </strong>{" "}
            tier with{" "}
            {Math.round(
              (data.result.segment_probabilities[data.result.predicted_segment] ?? 0) *
                100,
            )}
            % confidence.
          </p>
        </div>
      ) : null}
    </div>
  );
}

/* --- step 1: the stone as a person states it ---------------------------- */
function StepInput({ data }: { data: Trace }) {
  const i = data.input;
  return (
    <div>
      <h3 className="display display-md">
        What the reader asked for, and what the model is handed
      </h3>
      <p className="prose-note mt-4 max-w-[62ch]">
        You describe a stone by weight and three grades. The model also wants
        its measurements, so the pipeline finds a real stone from the dataset
        matching this description and borrows its dimensions. That is more
        honest than inventing measurements from textbook diamond physics: in
        this dataset volume tracks carat at 163.5 cubic millimetres per carat,
        not the 35.9 that crystal density would predict, because{" "}
        <code className="inline">x</code>,{" "}
        <code className="inline">y</code> and{" "}
        <code className="inline">z</code> are girdle measurements rather than
        the stone&apos;s bounding box.
      </p>
      <dl className="mt-7 grid gap-x-10 sm:grid-cols-2">
        <Row k="Carat" v={`${i.carat} ct`} />
        <Row k="Cut" v={i.cut} />
        <Row k="Colour" v={i.color} />
        <Row k="Clarity" v={i.clarity} />
        <Row k="Length x" v={`${i.x} mm`} />
        <Row k="Width y" v={`${i.y} mm`} />
        <Row k="Depth z" v={`${i.z} mm`} />
        <Row k="Depth %" v={`${i.depth}%`} />
        <Row k="Table %" v={`${i.table}%`} />
        <Row
          k="That stone sold for"
          v={money(data.reality_check.source_stone_price)}
          accent
        />
      </dl>
    </div>
  );
}

/* --- step 2: engineered features ---------------------------------------- */
function StepEngineer({ data }: { data: Trace }) {
  const d = data.derived;
  return (
    <div>
      <h3 className="display display-md">Three numbers the model adds</h3>
      <p className="prose-note mt-4 max-w-[62ch]">
        Nine raw columns become twelve. Three are computed from what you
        entered, and each one answers a question the raw columns cannot.
      </p>
      <div className="mt-7 grid gap-4 md:grid-cols-3">
        {[
          ["volume", d.volume],
          ["carat_per_volume", d.carat_per_volume],
          ["log_carat", d.log_carat],
        ].map(([name, v]) => {
          const val = v as Trace["derived"]["volume"];
          return (
            <div key={name as string} className="panel-raised p-5">
              <p className="num text-xs" style={{ color: "var(--accent)" }}>
                {val.formula}
              </p>
              <p
                className="num mt-3 text-2xl"
                style={{ wordBreak: "break-word" }}
              >
                {val.value}
              </p>
              <p className="index mt-3">{name as string}</p>
              <p className="caption mt-2">{val.note}</p>
            </div>
          );
        })}
      </div>
      <p className="prose-note mt-6 max-w-[62ch]">
        Weight is heavily right-skewed: a handful of stones weigh more than
        twenty small ones combined. The tree model can split on raw carat
        perfectly well, so the log is not required for accuracy, but it gives
        the splits a cleaner ordering to work with near zero.
      </p>
    </div>
  );
}

/* --- step 3: ordinal encoding ------------------------------------------ */
function StepEncode({ data }: { data: Trace }) {
  return (
    <div>
      <h3 className="display display-md">Grades become ordered numbers</h3>
      <p className="prose-note mt-4 max-w-[62ch]">
        Cut, colour and clarity are not arbitrary labels. They are ranked, so
        they get a number that keeps the ranking rather than a separate
        indicator column per level. One column says &ldquo;clarity 5&rdquo;,
        which the model can compare, instead of seven columns where only one
        is ever switched on.
      </p>
      <div className="mt-7 grid gap-4 md:grid-cols-3">
        {data.encoding.map((e) => (
          <div key={e.feature} className="panel-raised p-5">
            <p className="index">{e.feature}</p>
            <p
              className="num mt-3 text-xl"
              style={{ color: "var(--accent)" }}
            >
              {e.value}
            </p>
            <p className="caption mt-3">{e.why}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

/* --- step 4: what the forest actually does ----------------------------- */
function StepReason({ data }: { data: Trace }) {
  const t = data.tree;
  const r = data.result;
  const maxImp = Math.max(...t.top_features.map((f) => f.importance), 0.0001);
  return (
    <div>
      <h3 className="display display-md">The forest averages 300 trees</h3>
      <p className="prose-note mt-4 max-w-[62ch]">
        A random forest is 300 decision trees, each asking &ldquo;is this
        heavier than 1.4 carats?&rdquo; and averaging the answers. Each tree
        routes this stone into one leaf, and the price is the average of the
        training stones that landed in the same leaves. For this stone that is{" "}
        <strong style={{ color: "var(--cream)" }}>
          {t.leaf_size} real diamonds
        </strong>{" "}
        whose median price is{" "}
        <strong style={{ color: "var(--cream)" }}>
          {money(t.leaf_median_price)}
        </strong>
        .
      </p>

      <div className="mt-8">
        <p className="eyebrow mb-4">
          Which features the trees split on most
        </p>
        <div className="space-y-2.5">
          {t.top_features.map((f) => (
            <div key={f.feature} className="flex items-center gap-4">
              <span className="num w-32 shrink-0 text-xs">{f.feature}</span>
              <span
                className="h-2.5 rounded-full"
                style={{
                  width: `${Math.max((f.importance / maxImp) * 100, 2)}%`,
                  background: "var(--accent)",
                }}
              />
              <span className="num w-16 shrink-0 text-xs" style={{ color: "var(--ink-faint)" }}>
                {f.importance.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
        <p className="caption mt-4">
          Volume, carat and the two girdle measurements correlate above 0.99,
          so the trees split the same signal across all four. Read this as one
          strong weight signal, not four separate findings.
        </p>
      </div>

      <div className="mt-8 grid gap-4 md:grid-cols-3">
        <div className="panel-raised p-5">
          <p className="index">Weight alone</p>
          <p className="num mt-2 text-xl">{money(r.lowest_quality_price)}</p>
          <p className="caption mt-2">
            Same stone, all three grades dropped to worst.
          </p>
        </div>
        <div className="panel-raised p-5">
          <p className="index">Quality adds</p>
          <p
            className="num mt-2 text-xl"
            style={{ color: "var(--accent)" }}
          >
            {money(r.quality_premium)}
          </p>
          <p className="caption mt-2">
            What {r.predicted_segment.toLowerCase()} grades are worth on top
            of the weight.
          </p>
        </div>
        <div className="panel-raised p-5">
          <p className="index">Tier probabilities</p>
          <ul className="mt-2 space-y-1">
            {Object.entries(r.segment_probabilities).map(([k, v]) => (
              <li key={k} className="num flex justify-between text-xs">
                <span style={{ color: "var(--ink-faint)" }}>{k}</span>
                <span
                  style={{
                    color:
                      k === r.predicted_segment ? "var(--accent)" : "var(--cream)",
                  }}
                >
                  {(v * 100).toFixed(1)}%
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

/* --- step 5: is it right? --------------------------------------------- */
function StepCheck({ data }: { data: Trace }) {
  const rc = data.reality_check;
  return (
    <div>
      <h3 className="display display-md">Checking it against real stones</h3>
      <p className="prose-note mt-4 max-w-[62ch]">
        A prediction is only meaningful next to something it can be compared
        to. The comparison has to hold two things constant at once: the grade
        and the weight. Match the grade and the gap measures the grade. Match
        the weight only loosely and the gap measures the weight.
      </p>

      {rc.error_pct === null ? (
        <div className="panel-raised mt-7 p-6">
          <p className="index">No honest comparison available</p>
          <p className="prose-note mt-3 max-w-[58ch]">
            There are no comparable stones at this weight and grade in the
            dataset, so no accuracy figure is shown. Printing one anyway
            would be inventing a comparison.
          </p>
        </div>
      ) : (
        <>
          <div className="mt-7 grid gap-4 md:grid-cols-3">
            <div className="panel-raised p-5">
              <p className="index">This stone</p>
              <p className="num mt-2 text-xl">
                {money(data.result.predicted_price)}
              </p>
              <p className="caption mt-2">
                The real stone it was matched to sold for{" "}
                {money(rc.source_stone_price)}.
              </p>
            </div>
            <div className="panel-raised p-5">
              <p className="index">Comparable market</p>
              <p className="num mt-2 text-xl">
                {money(rc.market_median_price ?? 0)}
              </p>
              <p className="caption mt-2">
                Median of {rc.comparable_stones} real stones: {rc.comparison_group}
                , within{" "}
                {rc.carat_tolerance?.toFixed(2)} carats.
              </p>
            </div>
            <div className="panel-raised p-5">
              <p className="index">Difference</p>
              <p
                className="num mt-2 text-xl"
                style={{
                  color:
                    Math.abs(rc.error_pct) < 10
                      ? "var(--accent)"
                      : "var(--cream)",
                }}
              >
                {rc.error_pct > 0 ? "+" : ""}
                {rc.error_pct}%
              </p>
              <p className="caption mt-2">
                {Math.abs(rc.error_pct) < 10
                  ? "Within the normal spread of what comparable stones fetch."
                  : "Wide. This combination is rare, so few stones anchor it."}
              </p>
            </div>
          </div>

          {rc.price_spread ? (
            <p className="prose-note mt-6 max-w-[62ch]">
              Those comparable stones ranged from{" "}
              {money(rc.price_spread[0])} to {money(rc.price_spread[1])}, so
              even a correct prediction is only one draw from a spread of{" "}
              {money(rc.price_spread[1] - rc.price_spread[0])}. Real pricing
              is a range, not a number.
            </p>
          ) : null}
        </>
      )}
    </div>
  );
}

function Row({
  k,
  v,
  accent = false,
}: {
  k: string;
  v: string;
  accent?: boolean;
}) {
  return (
    <div className="kv">
      <dt>{k}</dt>
      <dd style={accent ? { color: "var(--accent)" } : undefined}>{v}</dd>
    </div>
  );
}
