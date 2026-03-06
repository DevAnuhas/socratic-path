import axios, { AxiosError } from "axios";
import { createClient } from "./supabase";
import type {
	ExploreRequest,
	ExploreResponse,
	ExplorationSummary,
	ExplorationDetail,
	SaveExplorationPayload,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const client = axios.create({
	baseURL: API_URL,
	timeout: 60_000, // 60s — model inference can be slow on CPU
});

// Attach Supabase JWT to every request
client.interceptors.request.use(async (config) => {
	const supabase = createClient();
	const {
		data: { session },
	} = await supabase.auth.getSession();

	if (session?.access_token) {
		config.headers.Authorization = `Bearer ${session.access_token}`;
	}
	return config;
});

function extractErrorMessage(err: unknown): string {
	if (err instanceof AxiosError) {
		// FastAPI error response
		const detail = err.response?.data?.detail;
		if (typeof detail === "string") return detail;

		// Auth errors
		if (err.response?.status === 401)
			return "Your session has expired. Please sign in again.";

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

export async function exploreQuestion(
	request: ExploreRequest,
): Promise<ExploreResponse> {
	try {
		const { data } = await client.post<ExploreResponse>(
			"/api/explore",
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

// ── Exploration CRUD ───────────────────────────────────────

export async function listExplorations(): Promise<ExplorationSummary[]> {
	try {
		const { data } =
			await client.get<ExplorationSummary[]>("/api/explorations");
		return data;
	} catch (err) {
		throw new Error(extractErrorMessage(err));
	}
}

export async function getExploration(id: string): Promise<ExplorationDetail> {
	try {
		const { data } = await client.get<ExplorationDetail>(
			`/api/explorations/${id}`,
		);
		return data;
	} catch (err) {
		throw new Error(extractErrorMessage(err));
	}
}

export async function saveExploration(
	payload: SaveExplorationPayload,
): Promise<ExplorationSummary> {
	try {
		const { data } = await client.post<ExplorationSummary>(
			"/api/explorations",
			payload,
		);
		return data;
	} catch (err) {
		throw new Error(extractErrorMessage(err));
	}
}

export async function deleteExploration(id: string): Promise<void> {
	try {
		await client.delete(`/api/explorations/${id}`);
	} catch (err) {
		throw new Error(extractErrorMessage(err));
	}
}
