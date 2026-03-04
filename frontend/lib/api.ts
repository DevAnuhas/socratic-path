import axios, { AxiosError } from "axios";
import type { GenerateRequest, GenerateResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const client = axios.create({
	baseURL: API_URL,
	timeout: 60_000, // 60s — model inference can be slow on CPU
});

function extractErrorMessage(err: unknown): string {
	if (err instanceof AxiosError) {
		// FastAPI error response
		const detail = err.response?.data?.detail;
		if (typeof detail === "string") return detail;

		// Network / timeout errors
		if (err.code === "ECONNABORTED")
			return "Request timed out. The model may still be loading — please wait and try again.";
		if (err.code === "ERR_NETWORK")
			return "Cannot connect to the backend. Is the server running?";

		return err.message;
	}
	if (err instanceof Error) return err.message;
	return "An unexpected error occurred";
}

export async function generateQuestions(
	request: GenerateRequest,
): Promise<GenerateResponse> {
	try {
		const { data } = await client.post<GenerateResponse>(
			"/api/generate",
			request,
		);
		return data;
	} catch (err) {
		throw new Error(extractErrorMessage(err));
	}
}

export async function checkHealth(): Promise<boolean> {
	try {
		const { data } = await client.get("/api/health");
		return data.model_loaded === true;
	} catch {
		return false;
	}
}
