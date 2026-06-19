import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

// Auto-cleanup React Testing Library between tests.
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// jsdom doesn't implement window.confirm / window.alert — stub them.
globalThis.confirm = vi.fn(() => true);
globalThis.alert = vi.fn();

// Stub scrollTo (used by some animations / focus handling)
globalThis.scrollTo = vi.fn();

// Provide a fresh fetch mock per-test via this helper.
export function installFetchMock(handler) {
  globalThis.fetch = vi.fn(handler);
  return globalThis.fetch;
}
