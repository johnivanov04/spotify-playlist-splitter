/**
 * Returns true if a user's Spotify access token is within 5 minutes of expiry
 * (or already past). Pure function over the user row's token timestamps.
 */
function isTokenNearExpiry(user) {
  if (!user?.tokenObtainedAt || !user?.expiresInSeconds) return true;
  const obtainedMs = new Date(user.tokenObtainedAt).getTime();
  if (!Number.isFinite(obtainedMs)) return true;
  const ageMs = Date.now() - obtainedMs;
  const expiresInMs = user.expiresInSeconds * 1000;
  // refresh 5 minutes before actual expiry to avoid mid-request failures
  return ageMs > expiresInMs - 5 * 60 * 1000;
}

module.exports = { isTokenNearExpiry };
