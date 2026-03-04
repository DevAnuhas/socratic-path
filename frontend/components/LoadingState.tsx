"use client";

import { cn } from "@/lib/utils";

export function LoadingState() {
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_280px]">
      {/* Question cards skeleton */}
      <div className="space-y-3">
        <div className="skeleton-shimmer h-4 w-36 rounded" />
        {[1, 2, 3, 4, 5].map((i) => (
          <div
            key={i}
            className={cn(
              "rounded-lg border border-border/60 bg-white p-4 space-y-2.5",
              `stagger-${i}`
            )}
            style={{ animationFillMode: "backwards" }}
          >
            <div className="skeleton-shimmer h-5 w-20 rounded-full" />
            <div className="skeleton-shimmer h-4 w-full rounded" />
            <div className="skeleton-shimmer h-4 w-3/4 rounded" />
          </div>
        ))}
      </div>

      {/* Sidebar skeleton */}
      <div className="space-y-4">
        <div className="skeleton-shimmer h-4 w-24 rounded" />
        <div className="flex flex-wrap gap-1.5">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="skeleton-shimmer h-6 w-16 rounded-md" />
          ))}
        </div>
        <div className="skeleton-shimmer mt-2 h-4 w-20 rounded" />
        {[1, 2].map((i) => (
          <div
            key={i}
            className="space-y-2 rounded-lg border border-border/60 bg-white p-3"
          >
            <div className="skeleton-shimmer h-3 w-24 rounded" />
            <div className="skeleton-shimmer h-3 w-full rounded" />
            <div className="skeleton-shimmer h-3 w-2/3 rounded" />
          </div>
        ))}
      </div>
    </div>
  );
}
