import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "../src/App.jsx";

// ---- Fetch mock router -------------------------------------------------
// A small handler that routes each request URL to a configurable response,
// so individual tests can override behavior without re-wiring.
let routes = {};

function fetchMockImpl(url, opts = {}) {
  const method = (opts.method || "GET").toUpperCase();
  // Normalize: drop query string and the API_BASE prefix
  const cleanUrl = url.replace("http://127.0.0.1:4000", "").split("?")[0];
  const key = `${method} ${cleanUrl}`;
  const route = routes[key] ?? routes[cleanUrl];

  if (!route) {
    // Default: 401, mimics not-authenticated state
    return Promise.resolve(makeRes({ error: "Not authenticated" }, 401));
  }
  if (typeof route === "function") {
    return Promise.resolve(route(url, opts));
  }
  return Promise.resolve(makeRes(route.body, route.status ?? 200));
}

function makeRes(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Map(),
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

function setRoute(key, response) {
  routes[key] = response;
}

beforeEach(() => {
  routes = {};
  globalThis.fetch = vi.fn(fetchMockImpl);
});

// ============================================================
// LOGGED-OUT LANDING PAGE
// ============================================================
describe("Landing page (logged-out)", () => {
  it("renders the title, tagline, and Spotify login button", async () => {
    // /api/me returns 401 → loggedIn stays false → landing renders
    render(<App />);

    expect(await screen.findByText("Playlist Splitter")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /log in with spotify/i })).toBeInTheDocument();
  });

  it("explains what the app does in the description", async () => {
    render(<App />);
    // Just check we have non-trivial copy on the landing page
    const desc = await screen.findByText(/Drop in any Spotify playlist/i);
    expect(desc).toBeInTheDocument();
  });

  it("renders the feature highlights (vibes / steer / cached)", async () => {
    render(<App />);
    // Each feature card has a heading
    expect(await screen.findByText(/Vibes, not genres/i)).toBeInTheDocument();
    expect(screen.getByText(/You're in the loop/i)).toBeInTheDocument();
    expect(screen.getByText(/Free to explore/i)).toBeInTheDocument();
  });

  it("does NOT show any playlist sidebar or main UI when logged out", async () => {
    render(<App />);
    await screen.findByText("Playlist Splitter");
    expect(screen.queryByText(/your playlists/i)).not.toBeInTheDocument();
  });

  it("clicking 'Log in with Spotify' redirects to the auth endpoint", async () => {
    // jsdom doesn't navigate; intercept window.location.href setter
    const origLocation = window.location;
    let assignedUrl = null;
    delete window.location;
    window.location = { href: "", get assign() { return assignedUrl; } };
    Object.defineProperty(window.location, "href", {
      set(v) { assignedUrl = v; },
      get() { return assignedUrl; },
    });

    render(<App />);
    const btn = await screen.findByRole("button", { name: /log in with spotify/i });
    await userEvent.click(btn);
    expect(assignedUrl).toContain("/auth/login");

    window.location = origLocation;
  });
});

// ============================================================
// LOGGED-IN APP SHELL
// ============================================================
describe("Logged-in app", () => {
  function loggedInRoutes(overrides = {}) {
    routes = {
      "GET /api/me": { body: { id: "spotify-user-1", display_name: "Test User" } },
      "GET /api/playlists": { body: { playlists: [
        { id: "p1", name: "My Mix", images: [{ url: "x" }], tracks: { total: 100 } },
        { id: "p2", name: "Workout", images: [{ url: "y" }], tracks: { total: 200 } },
      ] } },
      ...overrides,
    };
  }

  it("renders the playlist sidebar after login", async () => {
    loggedInRoutes();
    render(<App />);

    expect(await screen.findByText("My Mix")).toBeInTheDocument();
    expect(screen.getByText("Workout")).toBeInTheDocument();
    expect(screen.queryByText(/Drop in any Spotify playlist/i)).not.toBeInTheDocument();
  });

  it("renders the URL-paste field for loading playlists by link", async () => {
    loggedInRoutes();
    render(<App />);
    expect(await screen.findByPlaceholderText(/spotify playlist url/i)).toBeInTheDocument();
  });

  it("clicking a playlist triggers the tracks fetch", async () => {
    loggedInRoutes({
      "GET /api/playlists/p1/tracks": { body: { tracks: [], playlistName: "My Mix" } },
    });
    render(<App />);
    const pl = await screen.findByText("My Mix");
    await userEvent.click(pl);

    await waitFor(() => {
      const calls = globalThis.fetch.mock.calls.map((c) => c[0]);
      expect(calls.some((url) => url.includes("/api/playlists/p1/tracks"))).toBe(true);
    });
  });

  it("does not store Spotify tokens in localStorage or sessionStorage", async () => {
    loggedInRoutes();
    render(<App />);
    await screen.findByText("My Mix");

    // Sanity scan of all storage values
    const allLocal = Object.entries(localStorage).map(([k, v]) => `${k}=${v}`).join("|");
    const allSession = Object.entries(sessionStorage).map(([k, v]) => `${k}=${v}`).join("|");
    for (const blob of [allLocal, allSession]) {
      expect(blob).not.toContain("access_token");
      expect(blob).not.toContain("BQA"); // common Spotify token prefix
      expect(blob).not.toMatch(/refresh.token/i);
    }
  });
});

// ============================================================
// VIBE ANALYSIS FLOW
// ============================================================
describe("Vibe analysis flow", () => {
  function makeTrack(i) {
    return {
      id: `track-${i}`,
      name: `Track ${i}`,
      artists: ["Artist"],
      year: 2024,
      album: "Album",
      popularity: 50,
      imageUrl: "img",
      spotifyUrl: "https://...",
    };
  }

  function setupLoggedInWithPlaylist({ tracks = Array.from({ length: 10 }, (_, i) => makeTrack(i)) } = {}) {
    routes = {
      "GET /api/me": { body: { id: "spotify-user-1", display_name: "Test User" } },
      "GET /api/playlists": { body: { playlists: [{ id: "p1", name: "Mix", images: [] }] } },
      "GET /api/playlists/p1/tracks": { body: { tracks, playlistName: "Mix" } },
      "POST /api/spotify/artist-genres": { body: { byTrackId: {} } },
    };
  }

  it("fires POST /api/vibes/analyze after artist-genres returns", async () => {
    setupLoggedInWithPlaylist();
    let vibeCalled = false;
    routes["POST /api/vibes/analyze"] = (url, opts) => {
      vibeCalled = true;
      const body = JSON.parse(opts.body);
      expect(Array.isArray(body.tracks)).toBe(true);
      // Each track in the payload should have id (server uses this for mapping)
      expect(body.tracks[0].id).toBeDefined();
      return makeRes({
        cached: false,
        groupings: [
          {
            name: "Late Night",
            description: "calm hours",
            track_ids: ["track-0", "track-1", "track-2", "track-3"],
          },
        ],
      });
    };

    render(<App />);
    const pl = await screen.findByText("Mix");
    await userEvent.click(pl);

    await waitFor(() => expect(vibeCalled).toBe(true), { timeout: 3000 });
  });

  it("renders vibe card with name and description when groupings arrive", async () => {
    setupLoggedInWithPlaylist();
    routes["POST /api/vibes/analyze"] = {
      body: {
        cached: false,
        groupings: [
          {
            name: "2am Drive",
            description: "Subdued tracks for the empty highway.",
            track_ids: ["track-0", "track-1", "track-2", "track-3"],
          },
          {
            name: "Cookout Sunday",
            description: "Warm communal energy.",
            track_ids: ["track-4", "track-5", "track-6", "track-7"],
          },
        ],
      },
    };

    render(<App />);
    await userEvent.click(await screen.findByText("Mix"));

    expect(await screen.findByText("2am Drive")).toBeInTheDocument();
    // Description text shows up in both the .suggestion-sub paragraph and the
    // .rule-text "Rule:" line. Use getAllByText so this isn't brittle.
    expect(screen.getAllByText(/empty highway/i).length).toBeGreaterThan(0);
    expect(screen.getByText("Cookout Sunday")).toBeInTheDocument();
  });

  it("shows the loading banner while the AI is working", async () => {
    setupLoggedInWithPlaylist();
    // Make vibes take a moment so we can catch the loading state
    let resolveVibe;
    routes["POST /api/vibes/analyze"] = () =>
      new Promise((resolve) => {
        resolveVibe = () => resolve(makeRes({ cached: false, groupings: [] }));
      });

    render(<App />);
    await userEvent.click(await screen.findByText("Mix"));

    expect(await screen.findByText(/analyzing vibes/i)).toBeInTheDocument();

    resolveVibe();
  });

  it("clicking 'Refresh vibes' calls the API with force=true", async () => {
    setupLoggedInWithPlaylist();
    const calls = [];
    routes["POST /api/vibes/analyze"] = (url, opts) => {
      calls.push({ url, body: JSON.parse(opts.body) });
      return makeRes({
        cached: false,
        groupings: [
          {
            name: "First Vibe",
            description: "d",
            track_ids: ["track-0", "track-1", "track-2", "track-3"],
          },
        ],
      });
    };

    render(<App />);
    await userEvent.click(await screen.findByText("Mix"));

    // Wait for the initial vibe call + render
    await screen.findByText("First Vibe");

    // Click Refresh
    const refreshBtn = await screen.findByRole("button", { name: /refresh vibes/i });
    await userEvent.click(refreshBtn);

    await waitFor(() => {
      expect(calls.some((c) => c.url.includes("force=true"))).toBe(true);
    });
  });

  it("typing a steer prompt and refreshing sends the steer in the body", async () => {
    setupLoggedInWithPlaylist();
    const calls = [];
    routes["POST /api/vibes/analyze"] = (url, opts) => {
      calls.push({ url, body: JSON.parse(opts.body) });
      return makeRes({
        cached: false,
        groupings: [
          { name: "First", description: "d", track_ids: ["track-0", "track-1", "track-2", "track-3"] },
        ],
      });
    };

    render(<App />);
    await userEvent.click(await screen.findByText("Mix"));
    await screen.findByText("First");

    const steerInput = await screen.findByPlaceholderText(/steer the vibes/i);
    await userEvent.type(steerInput, "lean late-night");
    await userEvent.click(screen.getByRole("button", { name: /refresh vibes/i }));

    await waitFor(() => {
      const last = calls[calls.length - 1];
      expect(last.body.steer).toBe("lean late-night");
    });
  });

  it("does not crash when the vibe groupings have unusual or missing fields", async () => {
    setupLoggedInWithPlaylist();
    routes["POST /api/vibes/analyze"] = {
      body: {
        cached: false,
        groupings: [
          { name: "No Desc", description: "", track_ids: ["track-0", "track-1"] },
          { name: "🎵🎶 With Emoji", description: "weird", track_ids: ["track-2", "track-3"] },
        ],
      },
    };

    render(<App />);
    await userEvent.click(await screen.findByText("Mix"));
    expect(await screen.findByText("No Desc")).toBeInTheDocument();
    expect(screen.getByText(/🎵🎶 With Emoji/)).toBeInTheDocument();
  });
});

// ============================================================
// TRACK SELECTION INSIDE A VIBE CARD
// ============================================================
describe("Track keep/drop selection", () => {
  function setupWithExpandableVibe() {
    const tracks = Array.from({ length: 6 }, (_, i) => ({
      id: `t-${i}`,
      name: `Song ${i}`,
      artists: ["Artist"],
      year: 2024,
      album: "Album",
      popularity: 50,
      imageUrl: "i",
    }));
    routes = {
      "GET /api/me": { body: { id: "u1" } },
      "GET /api/playlists": { body: { playlists: [{ id: "p1", name: "Mix", images: [] }] } },
      "GET /api/playlists/p1/tracks": { body: { tracks, playlistName: "Mix" } },
      "POST /api/spotify/artist-genres": { body: { byTrackId: {} } },
      "POST /api/vibes/analyze": {
        body: {
          cached: false,
          groupings: [
            {
              name: "Vibe One",
              description: "d",
              track_ids: ["t-0", "t-1", "t-2", "t-3"],
            },
          ],
        },
      },
    };
  }

  it("expanding a card shows the tracks and Select all / Deselect all buttons", async () => {
    setupWithExpandableVibe();
    render(<App />);
    await userEvent.click(await screen.findByText("Mix"));
    await screen.findByText("Vibe One");

    await userEvent.click(screen.getByRole("button", { name: /view tracks/i }));

    expect(screen.getByText("Song 0")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^select all$/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^deselect all$/i })).toBeInTheDocument();
  });

  // KNOWN BEHAVIOR (arguably a UX bug): the helper
  // `getSelectedTrackIdsForSuggestion` treats an empty selection set as "no
  // explicit selection" and defaults to ALL tracks. So clicking Deselect All
  // and then Create currently saves every track. This test pins that behavior
  // so we know if it changes. Flag for product decision.
  it("[current behavior] Deselect All + Create still saves all tracks (empty set defaults to all)", async () => {
    setupWithExpandableVibe();
    const createCalls = [];
    routes["POST /api/playlists"] = (url, opts) => {
      createCalls.push(JSON.parse(opts.body));
      return makeRes({ id: "new-p", name: "x", external_urls: { spotify: "https://..." } });
    };

    render(<App />);
    await userEvent.click(await screen.findByText("Mix"));
    await screen.findByText("Vibe One");
    await userEvent.click(screen.getByRole("button", { name: /view tracks/i }));
    await userEvent.click(screen.getByRole("button", { name: /^deselect all$/i }));
    await userEvent.click(screen.getByRole("button", { name: /create playlist on spotify/i }));

    await waitFor(() => expect(createCalls.length).toBe(1));
    // Empty selection currently behaves as "select everything" — see
    // getSelectedTrackIdsForSuggestion in App.jsx.
    expect(createCalls[0].trackIds).toEqual(["t-0", "t-1", "t-2", "t-3"]);
  });

  it("Select all then save sends every track", async () => {
    setupWithExpandableVibe();
    const createCalls = [];
    routes["POST /api/playlists"] = (url, opts) => {
      createCalls.push(JSON.parse(opts.body));
      return makeRes({ id: "new-p", name: "x", external_urls: { spotify: "https://..." } });
    };

    render(<App />);
    await userEvent.click(await screen.findByText("Mix"));
    await screen.findByText("Vibe One");
    await userEvent.click(screen.getByRole("button", { name: /view tracks/i }));
    await userEvent.click(screen.getByRole("button", { name: /^select all$/i }));
    await userEvent.click(screen.getByRole("button", { name: /create playlist on spotify/i }));

    await waitFor(() => expect(createCalls.length).toBe(1));
    expect(createCalls[0].trackIds).toEqual(["t-0", "t-1", "t-2", "t-3"]);
  });
});

// ============================================================
// CREATE PLAYLIST FLOW
// ============================================================
describe("Create playlist", () => {
  function setupWithGroupings(groupings) {
    routes = {
      "GET /api/me": { body: { id: "u1" } },
      "GET /api/playlists": { body: { playlists: [{ id: "p1", name: "Mix", images: [] }] } },
      "GET /api/playlists/p1/tracks": { body: {
        tracks: Array.from({ length: 8 }, (_, i) => ({
          id: `t-${i}`, name: `S${i}`, artists: ["A"], year: 2024, popularity: 50, album: "", imageUrl: "",
        })),
        playlistName: "Mix",
      } },
      "POST /api/spotify/artist-genres": { body: { byTrackId: {} } },
      "POST /api/vibes/analyze": { body: { cached: false, groupings } },
    };
  }

  it("clicking 'Create playlist on Spotify' calls POST /api/playlists with name + trackIds", async () => {
    setupWithGroupings([
      { name: "Drive Night", description: "d", track_ids: ["t-0", "t-1", "t-2", "t-3"] },
    ]);
    const calls = [];
    routes["POST /api/playlists"] = (url, opts) => {
      calls.push(JSON.parse(opts.body));
      return makeRes({ id: "new-pid", external_urls: { spotify: "https://..." } });
    };

    render(<App />);
    await userEvent.click(await screen.findByText("Mix"));
    await screen.findByText("Drive Night");
    await userEvent.click(screen.getByRole("button", { name: /create playlist on spotify/i }));

    await waitFor(() => expect(calls.length).toBe(1));
    expect(calls[0].name).toBeTruthy();
    expect(calls[0].trackIds).toEqual(["t-0", "t-1", "t-2", "t-3"]);
  });

  it("only sends selected tracks when user has manually deselected some", async () => {
    setupWithGroupings([
      { name: "Drive Night", description: "d", track_ids: ["t-0", "t-1", "t-2", "t-3"] },
    ]);
    const calls = [];
    routes["POST /api/playlists"] = (url, opts) => {
      calls.push(JSON.parse(opts.body));
      return makeRes({ id: "new-pid", external_urls: { spotify: "https://..." } });
    };

    render(<App />);
    await userEvent.click(await screen.findByText("Mix"));
    await screen.findByText("Drive Night");
    await userEvent.click(screen.getByRole("button", { name: /view tracks/i }));

    // Find checkboxes — there should be 4 (one per track)
    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes.length).toBeGreaterThanOrEqual(4);
    // Untick the second one (t-1)
    await userEvent.click(checkboxes[1]);

    await userEvent.click(screen.getByRole("button", { name: /create playlist on spotify/i }));

    await waitFor(() => expect(calls.length).toBe(1));
    expect(calls[0].trackIds).not.toContain("t-1");
    expect(calls[0].trackIds).toContain("t-0");
    expect(calls[0].trackIds).toContain("t-2");
  });

  it("renders the failure alert when /api/playlists returns an error", async () => {
    setupWithGroupings([
      { name: "Group A", description: "d", track_ids: ["t-0", "t-1", "t-2", "t-3"] },
    ]);
    routes["POST /api/playlists"] = { status: 500, body: { error: "boom" } };

    render(<App />);
    await userEvent.click(await screen.findByText("Mix"));
    await screen.findByText("Group A");
    await userEvent.click(screen.getByRole("button", { name: /create playlist on spotify/i }));

    await waitFor(() => {
      expect(globalThis.alert).toHaveBeenCalled();
    });
  });
});

// ============================================================
// EDGE / RESILIENCE
// ============================================================
describe("Edge cases", () => {
  it("renders gracefully when the playlists endpoint returns an empty list", async () => {
    routes = {
      "GET /api/me": { body: { id: "u1" } },
      "GET /api/playlists": { body: { playlists: [] } },
    };
    render(<App />);
    // Sidebar should still render without throwing
    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining("/api/playlists"),
        expect.any(Object)
      );
    });
  });

  it("renders long playlist names without breaking layout", async () => {
    routes = {
      "GET /api/me": { body: { id: "u1" } },
      "GET /api/playlists": { body: { playlists: [
        { id: "p1", name: "A".repeat(300), images: [], tracks: { total: 1 } },
      ] } },
    };
    render(<App />);
    expect(await screen.findByText("A".repeat(300))).toBeInTheDocument();
  });
});
