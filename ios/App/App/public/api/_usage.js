// Guest-only usage limiter for Tarjiman Live. This app has no login system —
// it's a focused, standalone tool — so we cap usage per anonymous browser
// (client-generated guestId in localStorage) to protect the owner's API keys
// from abuse. Limit counts audio chunks processed (each chunk ~4 seconds of
// speech), reset daily (UTC).
const BLOB_TOKEN = process.env.BLOB_READ_WRITE_TOKEN;
const BLOB_BASE = 'https://blob.vercel-storage.com';
const STORE_ID = process.env.BLOB_STORE_ID || '6tfgxvttzyoiavtu';
const PUBLIC_BASE = 'https://' + STORE_ID + '.public.blob.vercel-storage.com/';

// ~400 chunks/day at ~4s each ≈ 25-30 minutes of live captioning per day, free.
const DAILY_LIMIT = 400;

function isValidGuestId(id) {
  return typeof id === 'string' && /^[a-zA-Z0-9_-]{6,64}$/.test(id);
}

function usagePath(key) {
  return 'omran-caption/db/usage/' + encodeURIComponent(key) + '.json';
}

function todayStr() {
  return new Date().toISOString().slice(0, 10);
}

async function getUsage(key) {
  try {
    const res = await fetch(PUBLIC_BASE + usagePath(key) + '?_=' + Date.now(), { cache: 'no-store' });
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    return null;
  }
}

async function putUsage(key, usage) {
  try {
    await fetch(BLOB_BASE + '/' + usagePath(key), {
      method: 'PUT',
      headers: {
        Authorization: 'Bearer ' + BLOB_TOKEN,
        'x-content-type': 'application/json',
        'x-add-random-suffix': '0',
        'x-cache-control-max-age': '0',
      },
      body: JSON.stringify(usage),
    });
  } catch (e) {
    // best-effort bookkeeping; never block the request over this
  }
}

async function checkAndConsume(guestId) {
  if (!isValidGuestId(guestId)) {
    return { allowed: false, reason: 'auth' };
  }
  const today = todayStr();
  const key = 'guest_' + guestId;
  let usage = await getUsage(key);
  if (!usage || usage.date !== today) {
    usage = { date: today, count: 0 };
  }
  if (usage.count >= DAILY_LIMIT) {
    return { allowed: false, reason: 'limit' };
  }
  usage.count += 1;
  await putUsage(key, usage);
  return { allowed: true, remaining: DAILY_LIMIT - usage.count };
}

module.exports = { checkAndConsume, DAILY_LIMIT };
