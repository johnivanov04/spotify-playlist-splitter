// client/src/ml/kmeansInfer.js

export function buildFeatureDict(track) {
  const f = {};

  const addNum = (name, v) => {
    const num = typeof v === "number" && Number.isFinite(v) ? v : null;
    f[name] = num === null ? 0.0 : num;
    f[`${name}__missing`] = num === null ? 1.0 : 0.0;
  };

  // Spotify numeric features (whatever you have available)
  addNum("sp__energy", track.energy);
  addNum("sp__valence", track.valence);
  addNum("sp__danceability", track.danceability);
  addNum("sp__tempo", track.tempo);
  addNum("sp__popularity", track.popularity);
  addNum("sp__year", track.year);

  // AcousticBrainz highlevel (robust to shape variations)
  const ab = track?.brainz?.acousticHighLevel;
  const high = ab?.highlevel && typeof ab.highlevel === "object" ? ab.highlevel : ab;

  if (high && typeof high === "object") {
    for (const [clfName, clfObj] of Object.entries(high)) {
      if (!clfObj || typeof clfObj !== "object") continue;

      const probs = clfObj.all || clfObj.probabilities;
      if (probs && typeof probs === "object") {
        for (const [clsName, p] of Object.entries(probs)) {
          const num = typeof p === "number" && Number.isFinite(p) ? p : null;
          if (num === null) continue;
          f[`ab__${clfName}__${clsName}`] = num;
        }
      } else {
        // fallback: value/probability
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
  // training used StandardScaler(with_mean=False): x_scaled = x / scale
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
