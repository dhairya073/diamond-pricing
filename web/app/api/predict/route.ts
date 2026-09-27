import { NextResponse } from "next/server";

const PYTHON_API =
  process.env.PYTHON_API_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request) {
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ detail: "Body was not valid JSON" }, { status: 400 });
  }

  try {
    const res = await fetch(`${PYTHON_API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
      signal: AbortSignal.timeout(20000),
    });
    const body = await res.json();
    return NextResponse.json(body, { status: res.status });
  } catch {
    return NextResponse.json(
      { detail: "Could not reach the model server. Is uvicorn running on port 8000?" },
      { status: 503 },
    );
  }
}
