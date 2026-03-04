import axios from "axios";
import type { GenerateRequest, GenerateResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const client = axios.create({
  baseURL: API_URL,
  timeout: 60_000, // 60s — model inference can be slow on CPU
});

export async function generateQuestions(
  request: GenerateRequest
): Promise<GenerateResponse> {
  const { data } = await client.post<GenerateResponse>(
    "/api/generate",
    request
  );
  return data;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const { data } = await client.get("/api/health");
    return data.model_loaded === true;
  } catch {
    return false;
  }
}
