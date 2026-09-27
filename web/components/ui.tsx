/* Shared building blocks. The design language lives in globals.css; this
   file only composes the few shapes that repeat across sections. */

import type { ReactNode } from "react";

/** The small monospace label that opens every section. */
export function Eyebrow({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <p className={`eyebrow ${className}`}>{children}</p>;
}

/** Section index in the top-right corner, e.g. 03 / 07. */
export function Index({ n, total }: { n: number; total: number }) {
  return (
    <p className="index shrink-0 tabular-nums">
      {String(n).padStart(2, "0")} / {String(total).padStart(2, "0")}
    </p>
  );
}

/**
 * One section shell: eyebrow, index marker, heading, lede. The heading and
 * lede widths are capped so a long sentence never runs the full 1180px,
 * which is well past a comfortable reading measure.
 */
export function SectionHead({
  n,
  total,
  eyebrow,
  title,
  lede,
}: {
  n: number;
  total: number;
  eyebrow: string;
  title: ReactNode;
  lede?: ReactNode;
}) {
  return (
    <header className="mb-12 md:mb-16">
      <div className="flex items-start justify-between gap-6">
        <Eyebrow>{eyebrow}</Eyebrow>
        <Index n={n} total={total} />
      </div>
      <h2 className="display display-lg mt-5 max-w-[16ch]">{title}</h2>
      {lede ? <p className="lede mt-6 max-w-[62ch]">{lede}</p> : null}
    </header>
  );
}

/** A key/value row, the spec-sheet motif used for model metrics. */
export function KV({
  k,
  v,
  highlight = false,
}: {
  k: string;
  v: ReactNode;
  highlight?: boolean;
}) {
  return (
    <div className="kv">
      <dt>{k}</dt>
      <dd style={highlight ? { color: "var(--accent)" } : undefined}>{v}</dd>
    </div>
  );
}

/**
 * A chart on its cream plate. The PNGs are drawn on paper, so the plate is
 * deliberate in both colour schemes rather than an accidental white block.
 */
export function Figure({
  src,
  alt,
  caption,
}: {
  src: string;
  alt: string;
  caption: string;
}) {
  return (
    <figure className="panel overflow-hidden">
      <div className="plate">
        {/* Plain img: assets are proxied same-origin through /api/charts, so
            Next Image optimization would add nothing.
            loading="lazy" is deliberately NOT used. A full-page screenshot
            catches the charts before the lazy observer fires, and a chart
            plate that renders as a blank cream block looks like a broken
            build. These are small PNGs, so eager loading is the right
            trade here. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={alt} width={1400} height={900} />
      </div>
      <figcaption className="border-t border-line-dark px-5 py-4">
        <p className="caption">{caption}</p>
      </figcaption>
    </figure>
  );
}

/** Inline code chip. */
export function Code({ children }: { children: ReactNode }) {
  return <code className="inline">{children}</code>;
}

/** Money, formatted the same way everywhere. */
export function Money({ v }: { v: number }) {
  return <>${Math.round(v).toLocaleString("en-US")}</>;
}
