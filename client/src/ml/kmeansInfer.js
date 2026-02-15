// client/src/ml/kmeansInfer.js

export function buildFeatureDict(track) {
  const f = {};

  const addNumWithHas = (numName, hasName, v) => {
    const num = typeof v === "number" && Number.isFinite(v) ? v : null;
    f[numName] = num === null ? 0.0 : num;
    f[hasName] = num === null ? 0.0 : 1.0; // IMPORTANT: model expects sp__has_*
  };

  // ---- Spotify numeric features + presence flags (MUST match training names) ----
  addNumWithHas("sp__danceability", "sp__has_danceability", track.danceability);
  addNumWithHas("sp__energy", "sp__has_energy", track.energy);
  addNumWithHas("sp__valence", "sp__has_valence", track.valence);
  addNumWithHas("sp__tempo", "sp__has_tempo", track.tempo);
  addNumWithHas("sp__popularity", "sp__has_popularity", track.popularity);
  addNumWithHas("sp__year", "sp__has_year", track.year);

  // ---- Optional: AcousticBrainz highlevel (ignored unless model.feature_names includes them) ----
  const ab = track?.brainz?.acousticHighLevel;
  const high =
    ab?.highlevel && typeof ab.highlevel === "object" ? ab.highlevel : ab;

  if (high && typeof high === "object") {
    for (const [clfName, clfObj] of Object.entries(high)) {
      if (!clfObj || typeof clfObj !== "object") continue;

      const probs = clfObj.all || clfObj.probabilities;
      if (probs && typeof probs === "object") {
        for (const [clsName, p] of Object.entries(probs)) {
          if (typeof p === "number" && Number.isFinite(p)) {
            f[`ab__${clfName}__${clsName}`] = p;
          }
        }
      } else {
        const val = clfObj.value;
        const p = clfObj.probability;
        if (val != null && typeof p === "number" && Number.isFinite(p)) {
          f[`ab__${clfName}__${val}`] = p;
        }
      }
    }
    f["ab__has_acousticbrainz"] = 1.0;
  } else {
    f["ab__has_acousticbrainz"] = 0.0;
  }

  return f;
}

export function vectorize(featureNames, featureDict) {
  // Dense vector in the exact order used at training time
  const x = new Array(featureNames.length).fill(0.0);
  for (let i = 0; i < featureNames.length; i++) {
    const name = featureNames[i];
    const v = featureDict[name];
    x[i] = typeof v === "number" && Number.isFinite(v) ? v : 0.0;
  }
  return x;
}

export function scaleVector(x, scale) {
  // Training used StandardScaler(with_mean=False): x_scaled = x / scale
  const xs = new Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const s = scale[i] || 1.0;
    xs[i] = s === 0 ? x[i] : x[i] / s;
  }
  return xs;
}

export function predictCluster(model, track) {
  const fd = buildFeatureDict(track);
  const x = vectorize(model.feature_names, fd);
  const xs = scaleVector(x, model.scale);

  // nearest centroid (euclidean)
  let bestK = 0;
  let bestD = Infinity;

  for (let k = 0; k < model.k; k++) {
    const c = model.centroids[k];
    let d = 0.0;
    for (let i = 0; i < xs.length; i++) {
      const diff = xs[i] - c[i];
      d += diff * diff;
    }
    if (d < bestD) {
      bestD = d;
      bestK = k;
    }
  }

  return bestK;
}
