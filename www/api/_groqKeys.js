// Rotates across multiple free Groq API keys so the app keeps working even
// after one free account hits its daily rate limit (429). Add more keys as
// GROQ_API_KEY_2, GROQ_API_KEY_3, ... in Vercel env vars — no code change needed.
function getGroqKeys() {
  const keys = [];
  if (process.env.GROQ_API_KEY) keys.push(process.env.GROQ_API_KEY);
  for (let i = 2; i <= 9; i++) {
    const k = process.env['GROQ_API_KEY_' + i];
    if (k) keys.push(k);
  }
  return keys;
}

// Tries each key in order for a given fetch call. `buildRequest(key)` must
// return { url, options }. Moves to the next key only on 429 (rate limit) or
// 401/403 (bad/exhausted key); any other error/response is returned as-is.
async function fetchWithGroqFallback(keys, buildRequest) {
  let lastRes = null;
  let lastErr = null;
  for (let i = 0; i < keys.length; i++) {
    try {
      const { url, options } = buildRequest(keys[i]);
      const res = await fetch(url, options);
      if (res.ok) return { res, keyIndex: i };
      if ((res.status === 429 || res.status === 401 || res.status === 403) && i < keys.length - 1) {
        lastRes = res;
        continue; // try next key
      }
      return { res, keyIndex: i }; // final answer (error or last key)
    } catch (e) {
      lastErr = e;
      if (i === keys.length - 1) throw e;
    }
  }
  if (lastRes) return { res: lastRes, keyIndex: keys.length - 1 };
  throw lastErr || new Error('All Groq keys failed');
}

module.exports = { getGroqKeys, fetchWithGroqFallback };
