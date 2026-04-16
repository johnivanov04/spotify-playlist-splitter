// client/src/ml/kmeansInfer.js
// Inference pipeline for v1_9 representation:
//   buildFeatureDict → vectorize → embed (scale + PCA) → per-playlist KMeans

// ---- Token normalization (mirrors Python featurize_song_records.py) ----

function normalizeToken(s) {
  s = (s || "").trim().toLowerCase();
  s = s.replace(/&/g, " and ");
  s = s.replace(/[^a-z0-9]+/g, "_");
  s = s.replace(/_+/g, "_").replace(/^_+|_+$/g, "");
  return s;
}

const TAG_ALIAS = {
  rhythm_and_blues: "r_and_b",
  west_coast_rap: "west_coast_hip_hop",
  westcoast_rap: "west_coast_hip_hop",
};

const SKIP_TAG_SUBSTRINGS = ["billboard", "hot_100", "chart", "charts", "offizielle", "wochen", "ph_"];
const SKIP_TAG_EXACT = new Set([
  "american", "american_rock", "english", "tempo_change", "vocal",
  "rap_hip_hop", "hip_hop_rap", "hip_hop_underground_hip_hop",
  "contemporary_rap_gangsta_rap_hardcore_rap_rap_west_coast_rap",
]);

function keepTag(tok) {
  if (!tok) return false;
  const canonical = TAG_ALIAS[tok] || tok;
  if (SKIP_TAG_EXACT.has(canonical)) return false;
  for (const bad of SKIP_TAG_SUBSTRINGS) if (canonical.includes(bad)) return false;
  return true;
}

const SKIP_ACOUSTIC_SUBSTRINGS = [
  "__all__not_", "__gender__", "__genre_dortmund__", "__genre_rosamerica__",
  "__genre_tzanetakis__", "__genre_electronic__", "__ismir04_rhythm__",
  "__moods_mirex__", "__tonal_atonal__",
];

function keepAcoustic(name) {
  if (!name) return false;
  if (name.endsWith("__probability")) return false;
  for (const bad of SKIP_ACOUSTIC_SUBSTRINGS) if (name.includes(bad)) return false;
  return true;
}

// ---- Feature extraction ----

export function buildFeatureDict(track, model) {
  const f = {};
  const W = model.weights;
  const yearMean = model.year_mean;
  const yearStd = model.year_std;

  // Year + popularity metadata
  const year = typeof track.year === "number" && track.year > 0 ? track.year : null;
  const pop = typeof track.popularity === "number" ? track.popularity : null;

  if (year !== null) {
    f["meta__has_year"] = W.binary;
    f["meta__year_z"] = ((year - yearMean) / yearStd) * W.year;
  }
  if (pop !== null) {
    f["meta__has_popularity"] = W.binary;
    f["meta__popularity_scaled"] = (pop / 100.0) * W.popularity;
  }

  // MusicBrainz tags
  const tags = track?.brainz?.tags || [];
  if (tags.length > 0) {
    f["meta__has_tags"] = W.binary;
    for (const { name, count } of tags) {
      const tok = normalizeToken(name);
      if (!keepTag(tok)) continue;
      const canonical = TAG_ALIAS[tok] || tok;
      const c = typeof count === "number" ? count : 1;
      const key = "tag__" + canonical;
      f[key] = (f[key] || 0) + Math.log1p(Math.max(c, 0)) * W.tag;
    }
  }

  // MusicBrainz genres + Spotify artist genres
  const mbGenres = track?.brainz?.genres || [];
  const spGenres = track?.spotifyArtistGenres || [];
  const allGenres = [...mbGenres, ...spGenres];
  if (allGenres.length > 0) {
    f["meta__has_genres"] = W.binary;
    for (const { name, count } of allGenres) {
      const tok = normalizeToken(name);
      if (!tok) continue;
      const c = typeof count === "number" ? count : 1;
      const key = "genre__" + tok;
      f[key] = (f[key] || 0) + Math.log1p(Math.max(c, 0)) * W.genre;
    }
  }

  // AcousticBrainz high-level
  const abData = track?.brainz?.acousticHighLevel;
  const hl = abData?.highlevel ?? (abData && typeof abData === "object" ? abData : null);
  if (hl && typeof hl === "object") {
    f["meta__has_acoustic_high_level"] = W.binary;
    for (const [clf, clfObj] of Object.entries(hl)) {
      if (!clfObj || typeof clfObj !== "object") continue;
      const allProbs = clfObj.all;
      if (!allProbs || typeof allProbs !== "object") continue;
      for (const [cls, prob] of Object.entries(allProbs)) {
        if (typeof prob !== "number" || !isFinite(prob)) continue;
        const featName = `acoustic__highlevel__${clf}__all__${cls}`;
        if (!keepAcoustic(featName)) continue;
        f[featName] = Math.max(Math.min(prob, 1.0), -1.0) * W.acoustic;
      }
    }
  }

  return f;
}

// ---- Vectorize + embed ----

export function vectorize(featureNames, featureDict) {
  const x = new Float32Array(featureNames.length);
  for (let i = 0; i < featureNames.length; i++) {
    const v = featureDict[featureNames[i]];
    x[i] = typeof v === "number" && isFinite(v) ? v : 0.0;
  }
  return x;
}

export function embed(x, model) {
  const n = x.length;
  const sm = model.scaler_mean;
  const ss = model.scaler_scale;
  const pm = model.pca_mean;
  const pc = model.pca_components;
  const nComp = pc.length;

  // Scale: (x - mean) / scale
  const xs = new Float32Array(n);
  for (let i = 0; i < n; i++) xs[i] = (x[i] - sm[i]) / (ss[i] || 1.0);

  // PCA: (xs - pca_mean) @ pca_components.T
  const e = new Float32Array(nComp);
  for (let k = 0; k < nComp; k++) {
    const comp = pc[k];
    let dot = 0.0;
    for (let i = 0; i < n; i++) dot += (xs[i] - pm[i]) * comp[i];
    e[k] = dot;
  }
  return e;
}

// ---- KMeans (k-means++ init, per-playlist) ----

function euclidSq(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) { const diff = a[i] - b[i]; d += diff * diff; }
  return d;
}

function kmeansOnce(points, k, maxIter = 100) {
  const n = points.length;
  const dim = points[0].length;

  // k-means++ init
  const centroids = [Array.from(points[Math.floor(Math.random() * n)])];
  for (let ki = 1; ki < k; ki++) {
    const dists = points.map(p => Math.min(...centroids.map(c => euclidSq(p, c))));
    const total = dists.reduce((s, d) => s + d, 0);
    let r = Math.random() * total;
    let chosen = n - 1;
    for (let i = 0; i < n; i++) { r -= dists[i]; if (r <= 0) { chosen = i; break; } }
    centroids.push(Array.from(points[chosen]));
  }

  const labels = new Int32Array(n);

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;
    for (let i = 0; i < n; i++) {
      let best = 0, bestD = Infinity;
      for (let ki = 0; ki < k; ki++) {
        const d = euclidSq(points[i], centroids[ki]);
        if (d < bestD) { bestD = d; best = ki; }
      }
      if (labels[i] !== best) { labels[i] = best; changed = true; }
    }
    if (!changed) break;

    // update centroids
    const sums = Array.from({ length: k }, () => new Float64Array(dim));
    const counts = new Int32Array(k);
    for (let i = 0; i < n; i++) {
      counts[labels[i]]++;
      for (let j = 0; j < dim; j++) sums[labels[i]][j] += points[i][j];
    }
    for (let ki = 0; ki < k; ki++) {
      if (counts[ki] > 0) for (let j = 0; j < dim; j++) centroids[ki][j] = sums[ki][j] / counts[ki];
    }
  }

  let inertia = 0;
  for (let i = 0; i < n; i++) inertia += euclidSq(points[i], centroids[labels[i]]);

  return { labels, centroids, inertia };
}

export function clusterPlaylist(model, tracks, k, nInit = 3) {
  const embeddings = tracks.map(t => {
    const fd = buildFeatureDict(t, model);
    const x = vectorize(model.feature_names, fd);
    return Array.from(embed(x, model));
  });

  const effectiveK = Math.min(k, Math.max(2, Math.floor(tracks.length / 6)));

  let best = null;
  for (let i = 0; i < nInit; i++) {
    const result = kmeansOnce(embeddings, effectiveK);
    if (!best || result.inertia < best.inertia) best = result;
  }
  return best.labels;
}
