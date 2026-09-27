"use client";

/** The closing questions, answered plainly, including the awkward ones. */

import { useState } from "react";

const FAQS = [
  {
    q: "What is this built on?",
    a: "The public ggplot2 diamonds dataset, which has 53,940 rows. After removing 168 records with impossible measurements it leaves 53,772. This is a teaching dataset, not a jewellery house inventory, so nothing here should be read as live market pricing.",
  },
  {
    q: "Why are the four value tiers my own labels?",
    a: "There is no industry-standard set of price bands, so the tiers are cut from price quartiles of this dataset: Budget, Mid-range, Premium and Luxury. They are even by construction, roughly 13,443 stones each, which is why the classifier scores so well and why that score should not be over-read.",
  },
  {
    q: "Why does the classifier not use price as an input?",
    a: "Because the tiers are defined by price. Feeding price in would let the model read the answer off the question, and it would score almost perfectly while learning nothing. It only sees weight, proportions and the three grades.",
  },
  {
    q: "The scatter charts look blocky. Why?",
    a: "There are over 50,000 points, and drawing them all with transparency produces a solid rectangle with no visible shape. They are drawn as density bins on log axes instead, which shows where the diamonds actually are and lets a fitted power law be drawn through them.",
  },
  {
    q: "Why do the feature importance bars group columns together?",
    a: "Because carat, volume and the girdle measurements correlate above 0.99. A tree can move credit freely between them, so ranking them individually would split one strong signal across four names and look like four findings. They are shuffled as one group, which matches how a buyer actually thinks: weight, clarity, colour, cut, proportions.",
  },
  {
    q: "Can I trust these numbers for a real purchase?",
    a: "Not on their own. The model is accurate within this dataset, on average, but it has never seen a single stone from your supplier, and it ignores brand, certification, market conditions and anything a buyer negotiates. Treat it as a way to check whether a price is in the right neighbourhood, the same way a calculator is not a valuation.",
  },
  {
    q: "What would make it better?",
    a: "Real inventory with actual sale prices rather than list prices, brand and certificate details as features, and a rolling retrain so the model does not drift as the market moves. The pipeline is built so those are additions, not a rewrite.",
  },
];

export function Faq() {
  const [open, setOpen] = useState<number | null>(0);

  return (
    <div className="border-t border-line-dark">
      {FAQS.map((f, i) => {
        const isOpen = open === i;
        return (
          <div key={f.q} className="border-b border-line-dark">
            <h3>
              <button
                type="button"
                onClick={() => setOpen(isOpen ? null : i)}
                aria-expanded={isOpen}
                className="flex w-full items-start justify-between gap-6 py-6 text-left"
              >
                <span
                  className="display display-sm"
                  style={{ color: isOpen ? "var(--accent)" : "var(--cream)" }}
                >
                  {f.q}
                </span>
                <span
                  aria-hidden="true"
                  className="num mt-1 shrink-0 text-sm"
                  style={{ color: "var(--ink-faint)" }}
                >
                  {isOpen ? "−" : "+"}
                </span>
              </button>
            </h3>
            {isOpen ? (
              <p className="prose-note max-w-[64ch] pb-7">{f.a}</p>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
