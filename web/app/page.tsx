import { fetchSummary } from "@/lib/api";
import { Eyebrow, SectionHead } from "@/components/ui";
import { Predictor } from "@/components/Predictor";
import { Results } from "@/components/Results";
import { TraceWalkthrough } from "@/components/TraceWalkthrough";
import { Faq } from "@/components/Faq";

const TOTAL = 7;

const NAV = [
  ["#overview", "Overview"],
  ["#eda", "The data"],
  ["#trace", "How it works"],
  ["#predict", "Predictor"],
  ["#finding", "Finding"],
  ["#method", "Method"],
  ["#questions", "Questions"],
] as const;

export default async function Page() {
  let data: Awaited<ReturnType<typeof fetchSummary>> | null = null;
  let failure: string | null = null;
  try {
    data = await fetchSummary();
  } catch (e) {
    failure =
      e instanceof Error
        ? e.message
        : "The model server did not respond.";
  }

  return (
    <>
      {/* ============================ header ============================ */}
      <header className="sticky top-0 z-50 border-b border-line-dark bg-night/85 backdrop-blur-md">
        <div className="shell flex h-16 items-center justify-between gap-6">
          <a href="#top" className="display display-sm shrink-0">
            Diamond<span style={{ color: "var(--accent)" }}>.</span>
          </a>
          <nav className="hidden gap-5 md:flex">
            {NAV.map(([href, label]) => (
              <a
                key={href}
                href={href}
                className="num text-[11px] uppercase tracking-wider transition-colors hover:text-accent"
                style={{ color: "var(--ink-faint)" }}
              >
                {label}
              </a>
            ))}
          </nav>
          <a href="#predict" className="btn btn-primary hidden sm:block">
            Try it
          </a>
        </div>
      </header>

      <main id="top">
        {/* ============================ hero ============================= */}
        <section className="section">
          <div className="shell">
            <Eyebrow>A data science project, explained end to end</Eyebrow>
            <h1 className="display display-xl mt-7 max-w-[13ch]">
              What actually makes a diamond cost what it costs?
            </h1>
            <p className="lede mt-8 max-w-[58ch]">
              {data
                ? `${data.meta.rows.toLocaleString("en-US")} cleaned diamond records, put through exploratory analysis, price regression, value-tier classification and clustering. Then a predictor you can drive, and a walkthrough showing how one number is produced.`
                : "Exploratory analysis, price regression, value-tier classification and clustering, with a live predictor and a walkthrough of how one number is produced."}
            </p>

            {data ? (
              <div className="mt-12 grid gap-px overflow-hidden rounded-lg border border-line-dark bg-line-dark sm:grid-cols-2 lg:grid-cols-4">
                {[
                  [
                    "Rows analysed",
                    data.meta.rows.toLocaleString("en-US"),
                    "after removing 168 impossible measurements",
                  ],
                  [
                    "Price R squared",
                    data.regression.best.r2.toFixed(4),
                    "held-out test set, never trained on",
                  ],
                  [
                    "Tier accuracy",
                    `${(data.classification.accuracy * 100).toFixed(1)}%`,
                    "predicting tier without seeing price",
                  ],
                  [
                    "Typical error",
                    `$${Math.round(data.regression.best.mae).toLocaleString("en-US")}`,
                    "mean absolute error per stone",
                  ],
                ].map(([label, value, note]) => (
                  <div key={label} className="bg-night-2 p-6">
                    <p className="index">{label}</p>
                    <p
                      className="num mt-3 text-3xl"
                      style={{ color: "var(--accent)" }}
                    >
                      {value}
                    </p>
                    <p className="caption mt-2">{note}</p>
                  </div>
                ))}
              </div>
            ) : null}

            <div className="mt-10 flex flex-wrap gap-3">
              <a href="#trace" className="btn btn-primary">
                See how a price is made
              </a>
              <a href="#predict" className="btn btn-ghost">
                Price a diamond
              </a>
            </div>

            {failure ? (
              <div className="panel mt-10 p-6">
                <p className="eyebrow">Model server not reachable</p>
                <p className="prose-note mt-3 max-w-[58ch]">
                  {failure} Start it with{" "}
                  <code className="inline">
                    uvicorn api:app --port 8000
                  </code>{" "}
                  and reload. The written sections below still render; the
                  numbers need the server.
                </p>
              </div>
            ) : null}
          </div>
        </section>

        {/* ========================== overview =========================== */}
        <section className="section-tight" id="overview">
          <div className="shell">
            <SectionHead
              n={1}
              total={TOTAL}
              eyebrow="What was done"
              title="Four questions, four models"
              lede="The project was built to answer specific questions rather than to try techniques off a list. Each section below maps to one question."
            />
            <div className="grid gap-px overflow-hidden rounded-lg border border-line-dark bg-line-dark md:grid-cols-2 lg:grid-cols-4">
              {[
                [
                  "01",
                  "What is in the data?",
                  "Distribution, correlation and outlier analysis across 53,772 records, to find out which columns actually carry information.",
                ],
                [
                  "02",
                  "What predicts price?",
                  "Three regressors on one split. Linear for interpretability, forests and boosting for accuracy. Chosen on RMSE, not vibes.",
                ],
                [
                  "03",
                  "Can price be tiered?",
                  "A classifier that predicts which value band a stone falls in using only its weight and quality, never its price.",
                ],
                [
                  "04",
                  "Is there hidden structure?",
                  "K-means on size and proportions, to see what the data groups by when nothing tells it to group by price.",
                ],
              ].map(([n, q, a]) => (
                <div key={n} className="bg-night-2 p-6">
                  <p className="num text-xs" style={{ color: "var(--accent)" }}>
                    {n}
                  </p>
                  <h3 className="display display-sm mt-4">{q}</h3>
                  <p className="prose-note mt-3">{a}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ============================ the data ========================= */}
        {data ? (
          <section className="section" id="eda">
            <div className="shell">
              <SectionHead
                n={2}
                total={TOTAL}
                eyebrow="Exploratory analysis"
                title="What the data looks like before any model touches it"
                lede="Every chart here is generated by the pipeline from the data, not drawn by hand. The captions state what the chart shows and what it means for pricing."
              />
              <Results data={data} />
            </div>
          </section>
        ) : null}

        {/* =========================== the trace ========================= */}
        <section className="section" id="trace">
          <div className="shell">
            <SectionHead
              n={3}
              total={TOTAL}
              eyebrow="How it works"
              title="One stone, five steps, no hand waving"
              lede="Most model explainers stop at &ldquo;the model learned a relationship&rdquo;. This walks a single stone through every stage and shows the actual intermediate values, because the models are running live as you change the inputs."
            />
            <TraceWalkthrough />
          </div>
        </section>

        {/* ========================== the predictor ====================== */}
        <section className="section" id="predict">
          <div className="shell">
            <SectionHead
              n={4}
              total={TOTAL}
              eyebrow="Live model"
              title="Drive the model yourself"
              lede="Every column the models accept, exposed. Hold the stone fixed and change one grade at a time to see what each is worth, then compare against the walkthrough above."
            />
            <Predictor />
          </div>
        </section>

        {/* ====================== business reading ====================== */}
        {data ? (
          <section className="section" id="reading">
            <div className="shell">
              <SectionHead
                n={5}
                total={TOTAL}
                eyebrow="Reading the result"
                title="What a buyer should take from this"
                lede="Model accuracy is the least interesting output. These are the decisions the numbers actually support."
              />
              <div className="grid gap-px overflow-hidden rounded-lg border border-line-dark bg-line-dark md:grid-cols-3">
                {[
                  [
                    "Weight sets the tier",
                    `A ${data.eda.elasticity.toFixed(2)} elasticity means price rises faster than weight. Segments are defined by price, so weight alone sorts most of the market into the right band.`,
                  ],
                  [
                    "Clarity is a large-stone play",
                    "Clarity is worth about twice as much per carat at the top of the weight range as at the bottom. A fixed premium schedule undervalues clarity precisely where inventory budget matters most.",
                  ],
                  [
                    "Colour is a flat play",
                    "Colour holds close to a constant multiple across every size band, so a fixed rule for it is defensible. Clarity and cut are not.",
                  ],
                ].map(([title, body]) => (
                  <div key={title} className="bg-night-2 p-7">
                    <h3 className="display display-sm">{title}</h3>
                    <p className="prose-note mt-4">{body}</p>
                  </div>
                ))}
              </div>

              <div className="panel mt-6 p-7 md:p-9">
                <p className="eyebrow">What this is not</p>
                <ul className="mt-5 space-y-4">
                  {[
                    "These are associations in one public dataset, not causal effects. Nothing here proves that raising clarity raises price, only that in this data the two move together.",
                    "The dataset is a teaching set. It is not a jewellery house inventory and it is not current market data, so it cannot price a real stone.",
                    "The model has no notion of brand, certification, market movement or negotiation, all of which move real prices.",
                    "Feature importance measures how much the model relies on a feature, not how much that feature matters in the market.",
                  ].map((t) => (
                    <li key={t} className="flex gap-4">
                      <span
                        aria-hidden="true"
                        className="num mt-1 shrink-0 text-xs"
                        style={{ color: "var(--accent)" }}
                      >
                        —
                      </span>
                      <span className="prose-note">{t}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </section>
        ) : null}

        {/* ============================ the method ======================= */}
        <section className="section" id="method">
          <div className="shell">
            <SectionHead
              n={6}
              total={TOTAL}
              eyebrow="Method"
              title="How it was built"
              lede="The parts of the process that are easy to skip and hard to undo."
            />
            <div className="grid gap-6 md:grid-cols-2">
              {[
                [
                  "Cleaning",
                  "168 of 53,940 records have a length, width or depth of zero, which is physically impossible. They are dropped rather than repaired, because a stone with no recorded depth tells you nothing about depth. That leaves 53,772 rows.",
                ],
                [
                  "Leakage",
                  "The models were checked for target leakage twice. A log-price column survived the first attempt, and the tier classifier was checked to confirm it never sees price, since the tiers are defined by price.",
                ],
                [
                  "Encoding",
                  "Cut, colour and clarity are ordinal, so they get a number that keeps the ranking. One-hot encoding them would throw away the fact that IF is better than I1.",
                ],
                [
                  "Reproducibility",
                  "One seed, 42, fixed for the split, the models and the sampling. Re-running the pipeline produces byte-identical metrics, which is what makes the automated copy checks meaningful.",
                ],
                [
                  "Honest comparison groups",
                  "Comparing a prediction to a market median is only fair if the group matches on grade and weight. An early version used a loose weight window and produced a spurious 97% error that was purely an artefact of the comparison.",
                ],
                [
                  "Verifying the writing",
                  "A script re-derives every number stated on this page from the generated artifacts and fails the build on a mismatch. It caught five wrong claims, including a trend direction that was backwards.",
                ],
              ].map(([title, body]) => (
                <div key={title} className="panel p-6 md:p-7">
                  <h3 className="display display-sm">{title}</h3>
                  <p className="prose-note mt-4">{body}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* =========================== questions ========================= */}
        <section className="section" id="questions">
          <div className="shell">
            <SectionHead
              n={7}
              total={TOTAL}
              eyebrow="Before you ask"
              title="The questions worth asking"
              lede="Including the ones that undercut the project, because they are the ones that show the limits were understood rather than missed."
            />
            <Faq />
          </div>
        </section>
      </main>

      {/* ============================= footer =========================== */}
      <footer className="border-t border-line-dark">
        <div className="shell py-12">
          <div className="flex flex-wrap items-end justify-between gap-8">
            <div>
              <p className="display display-md">
                Diamond Price Prediction
                <span style={{ color: "var(--accent)" }}>.</span>
              </p>
              <p className="caption mt-3 max-w-[46ch]">
                Built with pandas, scikit-learn, FastAPI and Next.js. Every
                figure and prediction on this page comes from the models
                running behind it, and every claim is checked against the data
                by an automated script.
              </p>
            </div>
            <div className="num index text-right">
              <p>53,772 rows</p>
              <p>seed 42</p>
              <p className="mt-3">public ggplot2 diamonds dataset</p>
            </div>
          </div>
        </div>
      </footer>
    </>
  );
}
