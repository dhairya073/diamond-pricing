import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const PYTHON_API =
  process.env.PYTHON_API_URL ?? "http://127.0.0.1:8000";

/**
 * Proxies the per-stone walkthrough. The browser calls this same-origin
 * handler rather than Python directly, so no CORS rule is needed.
 */
export async function GET(req: NextRequest) {
  const q = req.nextUrl.searchParams;
  const params = new URLSearchParams({
    carat: q.get("carat") ?? "1.2",
    cut: q.get("cut") ?? "Very Good",
    color: q.get("color") ?? "G",
    clarity: q.get("clarity") ?? "VS2",
  });

  try {
    const res = await fetch(`${PYTHON_API}/trace?${params}`, {
      cache: "no-store",
      signal: AbortSignal.timeout(20000),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      return NextResponse.json(
        { detail: body?.detail ?? `Model server returned ${res.status}.` },
        { status: res.status === 422 ? 422 : 502 },
      );
    }
    return NextResponse.json(await res.json());
  } catch {
    return NextResponse.json(
      {
        detail:
          "Could not reach the model server. Start it with: uvicorn api:app --port 8000",
      },
      { status: 503 },
    );
  }
}
