import { NextResponse } from "next/server";

const PYTHON_API =
  process.env.PYTHON_API_URL ?? "http://127.0.0.1:8000";

const ALLOWED = new Set([
  "01_distributions.png",
  "02_carat_price.png",
  "03_quality_boxplots.png",
  "04_correlation_heatmap.png",
  "05_price_per_carat.png",
  "06_segment_sizes.png",
  "07_predicted_vs_actual.png",
  "08_residual_bias.png",
  "09_feature_importance.png",
  "09b_confusion_matrix.png",
  "10_kmeans_clusters.png",
  "11_grade_effect_by_size.png",
]);

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ name: string }> },
) {
  const { name } = await params;
  if (!ALLOWED.has(name)) {
    return NextResponse.json({ detail: "Unknown chart" }, { status: 404 });
  }
  try {
    const res = await fetch(`${PYTHON_API}/charts/${name}`, {
      cache: "no-store",
      signal: AbortSignal.timeout(20000),
    });
    if (!res.ok) {
      return NextResponse.json({ detail: `Chart not available (${res.status})` }, { status: 502 });
    }
    const buf = await res.arrayBuffer();
    return new NextResponse(buf, {
      headers: {
        "Content-Type": "image/png",
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return NextResponse.json({ detail: "Could not reach the model server" }, { status: 503 });
  }
}
