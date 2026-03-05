"use client";

import { Sparkles, RotateCcw, LoaderCircle } from "lucide-react";
import { useAppStore } from "@/lib/store";
import {
	ALL_QUESTION_TYPES,
	QUESTION_TYPE_CONFIG,
	type QuestionType,
} from "@/lib/types";
import { cn } from "@/lib/utils";

export function InputPanel() {
	const {
		topic,
		setTopic,
		selectedTypes,
		toggleType,
		generate,
		reset,
		isLoading,
		hasGenerated,
	} = useAppStore();

	const handleSubmit = (e: React.FormEvent) => {
		e.preventDefault();
		generate();
	};

	const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
		if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
			e.preventDefault();
			generate();
		}
	};

	return (
		<form onSubmit={handleSubmit} className="space-y-4">
			{/* Topic input */}
			<div className="relative">
				<textarea
					value={topic}
					onChange={(e) => setTopic(e.target.value)}
					onKeyDown={handleKeyDown}
					placeholder="Enter a topic or statement to generate Socratic questions..."
					rows={3}
					className={cn(
						"w-full resize-none rounded-lg border bg-white px-4 py-3",
						"text-[15px] leading-relaxed placeholder:text-muted-foreground/50",
						"transition-colors duration-150",
						"focus:border-ring focus:outline-none focus:ring-2 focus:ring-ring/20",
					)}
				/>
				<span className="pointer-events-none absolute right-3 bottom-2.5 font-mono text-[11px] text-muted-foreground/40">
					{topic.length > 0 ? `${topic.length} chars` : "Cmd+Enter to submit"}
				</span>
			</div>

			{/* Question type chips */}
			<div className="flex flex-wrap items-center gap-2">
				<span className="mr-1 text-xs font-medium tracking-wide text-muted-foreground uppercase">
					Types
				</span>
				{ALL_QUESTION_TYPES.map((type) => {
					const config = QUESTION_TYPE_CONFIG[type];
					const isSelected = selectedTypes.includes(type);
					return (
						<button
							key={type}
							type="button"
							onClick={() => toggleType(type)}
							className={cn(
								"rounded-full border px-3 py-1 text-xs font-medium",
								"transition-all duration-150",
								isSelected ?
									`${config.bg} ${config.border} ${config.color}`
								:	"border-border bg-transparent text-muted-foreground/50 hover:border-muted-foreground/30",
							)}
						>
							{config.label}
						</button>
					);
				})}
			</div>

			{/* Actions */}
			<div className="flex items-center gap-3">
				<button
					type="submit"
					disabled={isLoading || !topic.trim()}
					className={cn(
						"inline-flex items-center gap-2 rounded-lg px-5 py-2.5",
						"bg-foreground text-background text-sm font-medium",
						"transition-all duration-150",
						"hover:bg-foreground/90",
						"disabled:cursor-not-allowed disabled:opacity-40",
					)}
				>
					{isLoading ?
						<>
							<LoaderCircle className="h-4 w-4 animate-spin" />
							Generating...
						</>
					:	<>
							<Sparkles className="h-4 w-4" />
							Generate Questions
						</>
					}
				</button>

				{hasGenerated && (
					<button
						type="button"
						onClick={reset}
						className="inline-flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
					>
						<RotateCcw className="h-3.5 w-3.5" />
						Reset
					</button>
				)}
			</div>
		</form>
	);
}
