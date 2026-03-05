"use client";

import { useEffect, useState } from "react";
import { X, Trash2, Clock, GitBranch, Loader2 } from "lucide-react";
import { listExplorations, getExploration, deleteExploration } from "@/lib/api";
import { useAppStore } from "@/lib/store";
import type { ExplorationSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

interface ExplorationsListProps {
	open: boolean;
	onClose: () => void;
}

export function ExplorationsList({ open, onClose }: ExplorationsListProps) {
	const [explorations, setExplorations] = useState<ExplorationSummary[]>([]);
	const [isLoading, setIsLoading] = useState(false);
	const [loadingId, setLoadingId] = useState<string | null>(null);
	const [deletingId, setDeletingId] = useState<string | null>(null);
	const [error, setError] = useState<string | null>(null);

	const loadExploration = useAppStore((s) => s.loadExploration);

	useEffect(() => {
		if (!open) return;
		setIsLoading(true);
		setError(null);
		listExplorations()
			.then(setExplorations)
			.catch((err) => setError(err.message))
			.finally(() => setIsLoading(false));
	}, [open]);

	async function handleLoad(id: string) {
		setLoadingId(id);
		try {
			const detail = await getExploration(id);
			loadExploration(detail);
			onClose();
		} catch (err) {
			setError(
				err instanceof Error ? err.message : "Failed to load exploration",
			);
		} finally {
			setLoadingId(null);
		}
	}

	async function handleDelete(id: string) {
		if (!confirm("Delete this exploration? This cannot be undone.")) return;
		setDeletingId(id);
		try {
			await deleteExploration(id);
			setExplorations((prev) => prev.filter((e) => e.id !== id));
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to delete");
		} finally {
			setDeletingId(null);
		}
	}

	if (!open) return null;

	return (
		<div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
			<div className="mx-4 w-full max-w-lg rounded-xl border border-border bg-white shadow-xl">
				{/* Header */}
				<div className="flex items-center justify-between border-b px-5 py-4">
					<h2 className="text-base font-semibold text-foreground">
						My Explorations
					</h2>
					<button
						onClick={onClose}
						className="rounded p-1 text-muted-foreground transition-colors hover:text-foreground cursor-pointer"
					>
						<X className="h-4 w-4" />
					</button>
				</div>

				{/* Content */}
				<div className="max-h-96 overflow-y-auto p-5">
					{isLoading && (
						<div className="flex items-center justify-center py-8">
							<Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
						</div>
					)}

					{error && (
						<div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
							{error}
						</div>
					)}

					{!isLoading && !error && explorations.length === 0 && (
						<p className="py-8 text-center text-sm text-muted-foreground">
							No saved explorations yet. Start exploring a topic and it will be
							saved automatically.
						</p>
					)}

					{!isLoading && explorations.length > 0 && (
						<ul className="space-y-2">
							{explorations.map((exp) => (
								<li
									key={exp.id}
									className={cn(
										"group flex items-center justify-between rounded-lg border px-4 py-3",
										"transition-colors hover:border-foreground/20 hover:bg-secondary/40",
										loadingId === exp.id && "opacity-60",
									)}
								>
									<button
										onClick={() => handleLoad(exp.id)}
										disabled={loadingId !== null}
										className="flex-1 text-left cursor-pointer"
									>
										<p className="text-sm font-medium text-foreground line-clamp-1">
											{exp.title}
										</p>
										<div className="mt-1 flex items-center gap-3 text-xs text-muted-foreground">
											<span className="inline-flex items-center gap-1">
												<GitBranch className="h-3 w-3" />
												{exp.node_count} nodes
											</span>
											<span className="inline-flex items-center gap-1">
												<Clock className="h-3 w-3" />
												{new Date(exp.updated_at).toLocaleDateString()}
											</span>
										</div>
									</button>

									<button
										onClick={(e) => {
											e.stopPropagation();
											handleDelete(exp.id);
										}}
										disabled={deletingId === exp.id}
										className={cn(
											"ml-3 rounded p-1.5 text-muted-foreground/50 transition-colors cursor-pointer",
											"opacity-0 group-hover:opacity-100",
											"hover:bg-red-50 hover:text-red-600",
											deletingId === exp.id && "opacity-100",
										)}
									>
										{deletingId === exp.id ?
											<Loader2 className="h-3.5 w-3.5 animate-spin" />
										:	<Trash2 className="h-3.5 w-3.5" />}
									</button>
								</li>
							))}
						</ul>
					)}
				</div>
			</div>
		</div>
	);
}
