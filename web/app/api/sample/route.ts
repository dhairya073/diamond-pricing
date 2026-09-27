import { NextResponse } from "next/server";

const PYTHON_API =
  process.env.PYTHON_API_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const res = await fetch(`${PYTHON_API}/sample`, {
      cache: "no-store",
      signal: AbortSignal.timeout(20000),
    });
    if (!res.ok) {
      return NextResponse.json({ detail: `Model server returned ${res.status}` }, { status: 502 });
    }
    return NextResponse.json(await res.json());
  } catch {
    return NextResponse.json({ detail: "Could not reach the model server" }, { status: 503 });
  }
}
