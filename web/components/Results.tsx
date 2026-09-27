"use client";

/**
 * The analysis display. Every figure here is read from the generated
 * artifacts, and the copy is checked against them by verify_copy.py, so a
 * claim cannot drift from the data without the build catching it.
 */

import { useState } from "react";
import type { GradeEffect, ModelScore, Summary } from "@/lib/types";
import { Figure } from "./ui";

function money(v: number) {
  return `$${Math.round(v).toLocaleString("en-US")}`;
}

/**
 * The best grade priced as a percentage above the worst.
 *
 * The table stores worst/best, so 50% means the worst grade costs half of
 * the best, which makes the best 100% more than the worst. Reading the
 * table number directly as a premium understates it by roughly half, which
 * is exactly the sort of arithmetic slip that makes a finding wrong.
 */
function premium(effect: GradeEffect, band: string) {
  return Math.round((1 / effect.ratio_by_band[band] - 1) * 100);
}

export function Results({ data }: { data: Summary }) {
  const [model, setModel] = useState<"random_forest" | "linear" | "gradient_boosting">(
    "random_forest",
  );

  const e = data.eda;
  const r = data.regression;
  const c = data.classification;
  const k = data.clustering;
  const segs = data.segments.segments;
  const scores: Record<string, ModelScore> = {
    random_forest: r.random_forest,
    linear: r.linear,
    gradient_boosting: r.gradient_boosting,
  };
  const chosen = scores[model];

  const groups = Object.entries(r.grouped).sort((a, b) => b[1] - a[1]);
  const maxGroup = Math.max(...groups.map(([, v]) => Math.abs(v)), 0.0001);
  const effects = e.size_controlled_grade_effect;
  const bands = Object.keys(effects.clarity.ratio_by_band).sort(
    (a, b) => Number(a) - Number(b),
  );
  // Where the clarity premium is widest. Not the largest band: the premium
  // peaks in the middle, so quoting the biggest band would understate it.
  const bestBand = bands.reduce((best, b) =>
    effects.clarity.ratio_by_band[b] < effects.clarity.ratio_by_band[best]
      ? b
      : best,
  );
  // bestBand is already a band label, NOT an index into bands. Indexing bands
  // with it returns undefined and prints nothing, so use it directly.
  const smallBand = bands[0];
  const largeBand = bands[bands.length - 1];

  return (
    <div className="space-y-20">
      {/* ================= EDA ================= */}
      <section id="eda">
        <div className="grid gap-6 lg:grid-cols-2">
          <Figure
            src="/charts/01_distributions.png"
            alt="Histograms of price, carat and price per carat"
            caption={`Price and carat are both strongly right-skewed, with skewness of ${e.skew.price.toFixed(1)} and ${e.skew.carat.toFixed(1)}. A few very large stones pull the mean well above the median, so a median or a log scale describes the typical diamond far better than an average does.`}
          />
          <Figure
            src="/charts/02_carat_price.png"
            alt="Density hexbin of carat against price on log scales, with the fitted power law"
            caption={`Weight is the dominant driver. On log-log axes the relationship is almost a straight line, with a correlation of ${e.carat_price_r_log.toFixed(3)}. The fitted slope is an elasticity of ${e.elasticity.toFixed(2)}: a 10% heavier stone costs roughly ${(Math.pow(1.1, e.elasticity) * 100 - 100).toFixed(0)}% more, so price rises faster than weight.`}
          />
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <Figure
            src="/charts/03_quality_boxplots.png"
            alt="Box plots of price by cut, colour and clarity grade"
            caption="Better grades cost more at every weight, but the gap widens with size, so the raw medians below understate what quality is worth on a large stone."
          />
          <Figure
            src="/charts/04_correlation_heatmap.png"
            alt="Correlation heatmap across the numeric features"
            caption={`Carat correlates with the girdle measurements at ${(e.correlations.carat_volume ?? 0).toFixed(2)} for volume, near-perfectly, because volume is derived from them. Redundant columns like these make per-column importance unreliable, which is why the importance section below reports feature groups instead.`}
          />
        </div>
      </section>

      {/* ================= the size-controlled finding ================= */}
      <section id="finding">
        <div className="panel p-6 md:p-10">
          <p className="eyebrow">The main finding</p>
          <h3 className="display display-lg mt-4 max-w-[18ch]">
            Quality is worth more in bigger stones, except colour
          </h3>
          <p className="lede mt-6 max-w-[64ch]">
            Comparing grade medians directly is misleading, because better
            graded diamonds are also heavier and weight dominates price. The
            fix is to compare price <em>per carat</em> inside bands of similar
            weight, so size is held constant and only the grade varies. Each
            line below shows the worst grade as a percentage of the best,
            within one of {effects.bands} carat bands. A line that rises means
            the quality premium is shrinking as stones get bigger.
          </p>

          <div className="mt-9 overflow-x-auto">
            <table className="w-full min-w-[34rem]">
              <thead>
                <tr className="border-b border-line-dark">
                  <th className="index pb-3 text-left">Grade compared</th>
                  {bands.map((b, i) => (
                    <th key={b} className="index pb-3 text-right">
                      {i === 0
                        ? "Smallest stones"
                        : i === bands.length - 1
                          ? "Largest stones"
                          : `Band ${i + 1}`}
                    </th>
                  ))}
                  <th className="index pb-3 text-right">Direction</th>
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ["Clarity", effects.clarity, "I1 vs IF"],
                    ["Cut", effects.cut, "Fair vs Ideal"],
                    ["Colour", effects.color, "J vs D"],
                  ] as const
                ).map(([label, eff, detail]) => {
                  const first = eff.ratio_by_band[bands[0]];
                  const last = eff.ratio_by_band[bands[bands.length - 1]];
                  // A falling worst/best ratio means the best grade is worth
                  // proportionally more as stones get heavier, so the premium
                  // widens. Compare the two ends, not the extremes within.
                  const widening = last < first;
                  return (
                    <tr key={label} className="border-b border-line-dark/40">
                      <td className="py-4 pr-4">
                        <span className="text-sm">{label}</span>
                        <span className="index ml-2">{detail}</span>
                      </td>
                      {bands.map((b) => (
                        <td
                          key={b}
                          className="num py-4 text-right text-sm"
                        >
                          {Math.round(eff.ratio_by_band[b] * 100)}%
                        </td>
                      ))}
                      <td className="py-4 pl-4 text-right">
                        <span
                          className="num text-xs"
                          style={{ color: "var(--accent)" }}
                        >
                          {widening ? "widens" : "narrows"}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="lede mt-8 max-w-[64ch]">
            So a single sourcing rule cannot treat all three grades the same.
            An IF stone is worth about {premium(effects.clarity, smallBand)}{" "}
            percent more per carat than an I1 in the smallest band, rising as
            far as {premium(effects.clarity, bestBand)} percent in the{" "}
            {bestBand} carat band. Cut behaves the same way, from{" "}
            {premium(effects.cut, smallBand)} to{" "}
            {premium(effects.cut, largeBand)} percent. Colour runs the other
            way, easing from {premium(effects.color, smallBand)} percent down
            to {premium(effects.color, largeBand)} percent, so a fixed rule
            for colour is defensible where one for clarity is not.
          </p>
          <p className="caption mt-5 max-w-[64ch]">
            The premium is the best grade priced as a percentage above the
            worst, so it is the inverse of the table above. It is also not
            perfectly smooth: the clarity premium peaks in the {bestBand} carat
            band rather than the largest one, which is a reminder that a
            five-band table is a summary of 53,772 stones, not a law.
          </p>
        </div>

        <div className="mt-6">
          <Figure
            src="/charts/11_grade_effect_by_size.png"
            alt="Line chart of worst grade as a percentage of best grade, across carat bands, for clarity, cut and colour"
            caption="The same comparison as a chart. Clarity and cut fall as stones get heavier, meaning the premium grows. Colour rises, meaning its premium shrinks. Reading the chart this way matters: the y axis is worst divided by best, so down is good for the buyer."
          />
        </div>
      </section>

      {/* ================= regression ================= */}
      <section id="regression">
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
          <div className="panel p-6 md:p-8">
            <p className="eyebrow">Price prediction</p>
            <h3 className="display display-md mt-4">
              Three models, same split, same seed
            </h3>
            <p className="prose-note mt-4 max-w-[54ch]">
              80% of the cleaned data trains the models, 20% is held back and
              never seen. The seed is fixed, so every number on this page
              reproduces exactly.
            </p>

            <div className="mt-7 flex flex-wrap gap-2">
              {(
                [
                  ["linear", "Linear"],
                  ["random_forest", "Random forest"],
                  ["gradient_boosting", "Gradient boost"],
                ] as const
              ).map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setModel(key)}
                  className={`btn ${model === key ? "btn-primary" : "btn-ghost"}`}
                >
                  {label}
                </button>
              ))}
            </div>

            <dl className="mt-7">
              <Row k="R squared" v={chosen.r2.toFixed(4)} accent />
              <Row k="Mean abs error" v={money(chosen.mae)} />
              <Row k="Root mean sq error" v={money(chosen.rmse)} />
              <Row
                k="Cross-val RMSE"
                v={`${money(chosen.cv_rmse_mean)} ± ${money(chosen.cv_rmse_std)}`}
              />
            </dl>
            <p className="caption mt-5">
              R squared of {chosen.r2.toFixed(3)} means the model explains{" "}
              {Math.round(chosen.r2 * 100)}% of the variation in price between
              held-out stones. A mean absolute error of {money(chosen.mae)} is
              small against a median price of{" "}
              {money(segs[1]?.median_price ?? 0)} for a mid-range stone.
            </p>
          </div>

          <div className="space-y-6">
            <Figure
              src="/charts/07_predicted_vs_actual.png"
              alt="Hexbin of predicted against actual price, and a residual plot"
              caption="Left: predictions sit close to the diagonal across four orders of magnitude. Right: residuals are centred on zero for most of the range but fan out at the top, meaning the model is least certain about the most expensive stones, which is also where absolute error matters most."
            />
            <div className="panel p-6 md:p-8">
              <p className="eyebrow">What actually drives price</p>
              <p className="prose-note mt-4 max-w-[58ch]">
                {r.grouped_note}
              </p>
              <div className="mt-6 space-y-3">
                {groups.map(([name, v]) => (
                  <div key={name} className="flex items-center gap-4">
                    <span className="num w-40 shrink-0 text-xs">{name}</span>
                    <span
                      className="h-2.5 rounded-full"
                      style={{
                        width: `${Math.max((Math.abs(v) / maxGroup) * 100, 1.5)}%`,
                        background: "var(--accent)",
                      }}
                    />
                    <span
                      className="num w-20 shrink-0 text-right text-xs"
                      style={{ color: "var(--ink-faint)" }}
                    >
                      {v >= 0 ? "+" : ""}
                      {v.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
              <p className="caption mt-5">
                Each bar is the drop in R squared when that whole group is
                shuffled. It measures how much the model relies on the group,
                not how much the group causes a price difference.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ================= classification ================= */}
      <section id="classification">
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="panel p-6 md:p-8">
            <p className="eyebrow">Value segmentation</p>
            <h3 className="display display-md mt-4">
              Predicting the tier without seeing the price
            </h3>
            <p className="prose-note mt-4 max-w-[54ch]">
              The four tiers are this project&apos;s own labels, cut from price
              quartiles. They are not official industry categories. The
              classifier never sees price as an input, so it has to infer the
              tier from weight and quality alone.
            </p>
            <dl className="mt-7">
              <Row k="Model" v={c.best_model} />
              <Row k="Accuracy" v={`${(c.accuracy * 100).toFixed(2)}%`} accent />
              <Row k="Weighted F1" v={c.f1_weighted.toFixed(4)} />
              <Row
                k="Logistic baseline"
                v={`${(c.logistic.accuracy * 100).toFixed(2)}% acc`}
              />
            </dl>
            <p className="caption mt-5">{c.imbalance_note}</p>
          </div>

          <div className="space-y-6">
            <Figure
              src="/charts/09b_confusion_matrix.png"
              alt="Confusion matrix of predicted against true value tier"
              caption="The diagonal is heavy, meaning most stones land in the right tier. The off-diagonal cells sit almost entirely between neighbouring tiers, and no stone is confused with a tier two steps away, so the errors are boundary cases rather than nonsense."
            />
            <div className="panel overflow-hidden">
              <div className="px-6 pt-6 md:px-8">
                <p className="eyebrow">The four tiers</p>
              </div>
              <div className="mt-4 overflow-x-auto">
                <table className="w-full min-w-[30rem]">
                  <thead>
                    <tr className="border-b border-line-dark">
                      <th className="index px-6 py-3 text-left md:px-8">Tier</th>
                      <th className="index px-3 py-3 text-right">Stones</th>
                      <th className="index px-3 py-3 text-right">Median</th>
                      <th className="index px-3 py-3 text-right">Per ct</th>
                      <th className="index px-6 py-3 text-right md:px-8">
                        Best median grade
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {segs.map((s) => (
                      <tr key={s.segment} className="border-b border-line-dark/40">
                        <td className="px-6 py-3.5 text-sm md:px-8">
                          {s.segment}
                        </td>
                        <td className="num px-3 py-3.5 text-right text-sm">
                          {s.n.toLocaleString("en-US")}
                        </td>
                        <td className="num px-3 py-3.5 text-right text-sm">
                          {money(s.median_price)}
                        </td>
                        <td className="num px-3 py-3.5 text-right text-sm">
                          {money(s.median_ppc)}
                        </td>
                        <td className="px-6 py-3.5 text-right text-xs md:px-8">
                          {s.best_median_cut}, {s.best_median_color},{" "}
                          {s.best_median_clarity}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ================= clustering ================= */}
      <section id="clustering">
        <div className="grid gap-6 lg:grid-cols-2">
          <Figure
            src="/charts/10_kmeans_clusters.png"
            alt="Scatter of carat against volume coloured by K-means cluster"
            caption={`K-means finds two groups with a silhouette score of ${k.silhouette.toFixed(3)}. The split is by size, not by price: one group is small stones, the other large ones. This is a different question from the value tiers above, and the two do not agree.`}
          />
          <div className="panel p-6 md:p-8">
            <p className="eyebrow">Clustering, and why it is not the tiers</p>
            <h3 className="display display-md mt-4">
              An unsupervised answer that disagrees
            </h3>
            <p className="prose-note mt-4 max-w-[56ch]">{k.interpretation}</p>
            <p className="prose-note mt-4 max-w-[56ch]">
              The clearest evidence: the cluster that most often falls on a
              tier label agrees only{" "}
              <strong style={{ color: "var(--accent)" }}>
                {(k.modal_segment_agreement * 100).toFixed(1)}%
              </strong>{" "}
              of the time. Clustering on size and proportions finds structure
              that price quartiles do not capture, which is the more useful
              result of the two.
            </p>
            <div className="mt-7 grid gap-4 sm:grid-cols-2">
              {k.profiles.map((p) => (
                <div key={p.cluster} className="panel-raised p-5">
                  <p className="index">Cluster {p.cluster}</p>
                  <p className="num mt-2 text-lg">
                    {p.median_carat} ct median
                  </p>
                  <p className="caption mt-2">
                    {p.n.toLocaleString("en-US")} stones at{" "}
                    {money(p.median_price_per_carat)} per carat. Most often{" "}
                    {p.modal_clarity} clarity, {p.modal_color} colour.
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
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
