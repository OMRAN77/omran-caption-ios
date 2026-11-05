// Vercel Serverless Function: live captioning + translation.
// 1) Transcribes a short audio chunk (one spoken phrase) via Groq Whisper.
// 2) Translates the transcribed text via a Groq chat model, using the last few
//    lines of the conversation as context.
// Runs entirely on the site owner's own server-side GROQ_API_KEY — no user keys.
// Private app: there is no daily usage limit.
const { getGroqKeys, fetchWithGroqFallback } = require('./_groqKeys');

// Groq retires models over time (llama-3.1-8b-instant started returning 404),
// so model names are a preference list, not a hard dependency: the server asks
// Groq which models this account actually has and uses the first available one.
// Groq counts free quota separately per model, so the next one also takes over
// when a model's quota is used up (429).
// Force a model with GROQ_STT_MODEL / GROQ_CHAT_MODEL in Vercel (no code change).
const uniq = (a) => a.filter((x, i) => x && a.indexOf(x) === i);
const STT_MODELS = uniq([process.env.GROQ_STT_MODEL, 'whisper-large-v3-turbo', 'whisper-large-v3']);
const CHAT_MODELS = uniq([process.env.GROQ_CHAT_MODEL,
  'llama-3.1-8b-instant', 'llama-3.3-70b-versatile',
  'meta-llama/llama-4-scout-17b-16e-instruct', 'meta-llama/llama-4-maverick-17b-128e-instruct',
  'moonshotai/kimi-k2-instruct', 'openai/gpt-oss-20b', 'openai/gpt-oss-120b', 'qwen/qwen3-32b']);
// Never used for translation (speech, safety classifiers, tool agents, TTS).
const NOT_CHAT = /whisper|tts|guard|playai|orpheus|compound|distil|safeguard/i;

// Which models this account can use right now (GET /models), cached 1 hour.
let modelIdsCache = { at: 0, ids: null };
async function groqModelIds(keys) {
  const age = Date.now() - modelIdsCache.at;
  if (modelIdsCache.ids && age < 3600 * 1000) return modelIdsCache.ids;
  if (!modelIdsCache.ids && age < 10 * 60 * 1000) return null; // listing failed recently
  for (const key of keys) {
    try {
      const r = await fetch('https://api.groq.com/openai/v1/models', { headers: { 'Authorization': 'Bearer ' + key } });
      if (!r.ok) continue;
      const j = await r.json();
      const ids = (Array.isArray(j.data) ? j.data : []).filter((m) => m && m.id && m.active !== false).map((m) => m.id);
      if (ids.length) { modelIdsCache = { at: Date.now(), ids }; return ids; }
    } catch (e) {}
  }
  modelIdsCache = { at: Date.now(), ids: null };
  return null; // unknown: fall back to trying the preference list
}

// Preference order, limited to what exists; if none of our names exist any
// more, any text model Groq offers is used rather than failing.
async function availableModels(keys, preferred, kind) {
  const ids = await groqModelIds(keys);
  if (!ids) return preferred;
  const have = preferred.filter((m) => ids.includes(m));
  if (kind === 'stt') return have.length ? have : preferred;
  const others = ids.filter((id) => !have.includes(id) && !NOT_CHAT.test(id));
  return have.concat(others).length ? have.concat(others) : preferred;
}

// Reasoning models need room to think and must not print their thoughts.
function chatBody(model, messages) {
  const body = { model, messages, temperature: 0.1, max_tokens: 150 };
  if (/gpt-oss/i.test(model)) { body.reasoning_effort = 'low'; body.max_tokens = 1024; }
  if (/qwen3|deepseek-r1/i.test(model)) { body.reasoning_format = 'hidden'; body.max_tokens = 1024; }
  return body;
}

// OpenAI is used as a *paid* fallback once every free Groq key has hit its
// daily quota (HTTP 429), so live captioning never fully stops.
async function transcribeWithOpenAI(buf, mimeType, ext, lang) {
  const form = new FormData();
  form.append('file', new Blob([buf], { type: mimeType || 'audio/webm' }), 'audio.' + ext);
  form.append('model', 'whisper-1');
  form.append('response_format', 'verbose_json');
  form.append('temperature', '0');
  if (lang) form.append('language', lang);
  return fetch('https://api.openai.com/v1/audio/transcriptions', {
    method: 'POST',
    headers: { 'Authorization': 'Bearer ' + process.env.OPENAI_API_KEY },
    body: form,
  });
}

async function translateWithOpenAI(messages) {
  let r = null;
  for (const model of ['gpt-4o-mini', 'gpt-4.1-mini']) {
    r = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + process.env.OPENAI_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ model, messages, temperature: 0.1, max_completion_tokens: 150 }),
    });
    if (r.status !== 404) return r;
  }
  return r;
}

// Models whose quota ran out on every key, until Groq's retry-after passes.
// Lives as long as the warm serverless instance; saves a wasted call per key.
const cooldownUntil = {};

function retryAfterSec(res) {
  const s = parseFloat(res && res.headers && res.headers.get('retry-after'));
  return Number.isFinite(s) && s > 0 ? s : 60;
}

// Tries each model in turn (each with every Groq key). Moves to the next model
// when that model's quota is used up on all keys (429) or Groq says the model
// itself is unavailable (400/404).
async function groqWithModels(keys, models, build) {
  const ready = models.filter((m) => !(cooldownUntil[m] > Date.now()));
  const list = ready.length ? ready : models; // all cooling down: try anyway
  let res = null;
  for (let i = 0; i < list.length; i++) {
    const model = list[i];
    ({ res } = await fetchWithGroqFallback(keys, (key) => build(key, model)));
    if (res.ok) return res;
    const last = i === list.length - 1;
    if (res.status === 429) {
      cooldownUntil[model] = Date.now() + Math.min(retryAfterSec(res), 3600) * 1000;
      if (last) return res;
      try { await res.body.cancel(); } catch (e) {}
      continue;
    }
    if (last || !(res.status === 400 || res.status === 404)) return res;
    const errText = await res.clone().text().catch(() => '');
    if (!/model/i.test(errText)) return res;
    // Model missing/retired for this account: skip it for a day.
    cooldownUntil[model] = Date.now() + 24 * 3600 * 1000;
  }
  return res;
}

// 429 from Groq is either the per-minute limit (short wait) or the daily free
// quota (all keys, all models). Tell them apart so the message is accurate.
function quotaMessage(res) {
  return retryAfterSec(res) <= 120
    ? 'ضغط مؤقت على Groq (حد الدقيقة)، يرجع خلال ثوانٍ. / Groq is briefly rate-limited, back in a few seconds.'
    : 'خلص رصيد Groq المجاني لليوم في كل المفاتيح والموديلات. أضف مفتاح Groq آخر (GROQ_API_KEY_2) أو OPENAI_API_KEY في Vercel. / Free Groq quota used up for today on all keys and models.';
}

const LANG_NAMES = {
  ar: 'Arabic', en: 'English', fr: 'French', es: 'Spanish', de: 'German',
  it: 'Italian', pt: 'Portuguese', ru: 'Russian', zh: 'Chinese', ja: 'Japanese',
  ko: 'Korean', hi: 'Hindi', ur: 'Urdu', fa: 'Persian', tr: 'Turkish',
  nl: 'Dutch', pl: 'Polish', sv: 'Swedish', el: 'Greek', he: 'Hebrew',
  th: 'Thai', vi: 'Vietnamese', id: 'Indonesian', ms: 'Malay', bn: 'Bengali',
  ta: 'Tamil', ml: 'Malayalam', ku: 'Kurdish', sw: 'Swahili', am: 'Amharic',
  uk: 'Ukrainian', ro: 'Romanian', hu: 'Hungarian', cs: 'Czech', fi: 'Finnish',
  da: 'Danish', no: 'Norwegian', sk: 'Slovak', bg: 'Bulgarian', sr: 'Serbian',
  hr: 'Croatian', lt: 'Lithuanian', lv: 'Latvian', et: 'Estonian', az: 'Azerbaijani',
  ka: 'Georgian', hy: 'Armenian', km: 'Khmer', lo: 'Lao', my: 'Burmese',
  ne: 'Nepali', si: 'Sinhala', ps: 'Pashto', so: 'Somali', ha: 'Hausa',
  yo: 'Yoruba', zu: 'Zulu', af: 'Afrikaans', sq: 'Albanian', mk: 'Macedonian',
  mn: 'Mongolian', tl: 'Filipino',
};
// Whisper's verbose_json returns a full language name (e.g. "english"), not an
// ISO code. Map it back to our short codes (plus Whisper's own spellings).
const NAME_TO_CODE = Object.assign(
  Object.fromEntries(Object.entries(LANG_NAMES).map(([code, name]) => [name.toLowerCase(), code])),
  { tagalog: 'tl', myanmar: 'my', 'burmese': 'my' }
);

// Whisper "hallucinates" these on silence/noise (well-known, esp. in Arabic).
const HALLUCINATION_EXACT = ['thanks for watching', '...', '..', '.', 'يوتيوب', 'موسيقى', '[موسيقى]'];
const HALLUCINATION_PATTERNS = [
  /نانسي\s*قنقر/, /اشتركوا\s*في\s*القناة/, /شكرا[ًا]?\s*(لكم\s*)?(على|لل)\s*المشاهدة/,
  /thanks? (you )?for watching/i, /please subscribe/i, /amara\.org/i,
];

async function transcribe(keys, buf, mimeType, ext, lang) {
  let res = await groqWithModels(keys, await availableModels(keys, STT_MODELS, 'stt'), (key, model) => {
    // A fresh form per attempt (bodies are not reused across fetches).
    const form = new FormData();
    form.append('file', new Blob([buf], { type: mimeType || 'audio/webm' }), 'audio.' + ext);
    form.append('model', model);
    form.append('response_format', 'verbose_json');
    // Encourage the model to only transcribe real speech, not invent text.
    form.append('temperature', '0');
    if (lang) form.append('language', lang);
    return {
      url: 'https://api.groq.com/openai/v1/audio/transcriptions',
      options: { method: 'POST', headers: { 'Authorization': 'Bearer ' + key }, body: form },
    };
  });
  let usedOpenAI = false;
  if (!res.ok && res.status === 429 && process.env.OPENAI_API_KEY) {
    // All free Groq keys are exhausted — fall back to the owner's paid OpenAI key.
    res = await transcribeWithOpenAI(buf, mimeType, ext, lang);
    usedOpenAI = true;
  }
  if (!res.ok) return { res, usedOpenAI };

  const data = await res.json();
  let text = (data.text || '').trim();
  // ---- Silence / hallucination filter ----
  // The probability checks are intentionally conservative (high thresholds) so
  // real speech with brief pauses or moderate mic volume is never dropped.
  const segments = Array.isArray(data.segments) ? data.segments : [];
  let avgNoSpeechProb = 0, avgLogProb = 0;
  if (segments.length) {
    avgNoSpeechProb = segments.reduce((s, seg) => s + (seg.no_speech_prob || 0), 0) / segments.length;
    avgLogProb = segments.reduce((s, seg) => s + (typeof seg.avg_logprob === 'number' ? seg.avg_logprob : 0), 0) / segments.length;
  }
  const normalized = text.toLowerCase().replace(/[.,!؟?]+$/g, '').trim();
  const hallucination =
    text.length < 2 ||
    HALLUCINATION_EXACT.includes(normalized) ||
    HALLUCINATION_PATTERNS.some((re) => re.test(text)) ||
    avgNoSpeechProb > 0.85 ||
    (segments.length > 0 && avgLogProb < -1.8);
  const debug = { avgNoSpeechProb, avgLogProb, rawBeforeFilter: text, filtered: hallucination, usedOpenAI };
  if (hallucination) text = '';

  const rawLang = String(data.language || '').toLowerCase();
  const detected = NAME_TO_CODE[rawLang] || rawLang;
  return { res, text, detected, debug, usedOpenAI };
}

function translationMessages(original, targetName, context) {
  let sys = 'You are a live interpreter. Translate the user\'s utterance into ' + targetName +
    ' naturally and accurately (understand dialects such as Gulf Arabic; keep names and numbers). ' +
    'Never answer or comment on it, add no notes or quotes. Output ONLY the translation.';
  if (context.length) {
    sys += '\nPrevious lines (context only, do not translate): ' + context.join(' | ');
  }
  return [{ role: 'system', content: sys }, { role: 'user', content: original }];
}

// Diagnostics: open https://<site>/api/caption?check=1 to see, for each Groq
// key (by number only), whether the default models still have free quota.
// Costs one tiny request per key and model.
function silentWav() {
  const sr = 16000, n = sr / 2, buf = Buffer.alloc(44 + n * 2);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * 2, 4); buf.write('WAVE', 8); buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22); buf.writeUInt32LE(sr, 24);
  buf.writeUInt32LE(sr * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34); buf.write('data', 36); buf.writeUInt32LE(n * 2, 40);
  return buf;
}

function groqKeyNames() {
  const names = [];
  if (process.env.GROQ_API_KEY) names.push('GROQ_API_KEY');
  for (let i = 2; i <= 9; i++) if (process.env['GROQ_API_KEY_' + i]) names.push('GROQ_API_KEY_' + i);
  return names;
}

async function quotaCheck() {
  const keys = getGroqKeys();
  const names = groqKeyNames();
  modelIdsCache = { at: 0, ids: null }; // fresh list for the check
  const ids = await groqModelIds(keys);
  const chatList = await availableModels(keys, CHAT_MODELS, 'chat');
  const sttList = await availableModels(keys, STT_MODELS, 'stt');
  const pick = (r) => {
    const h = (k) => r.headers.get(k) || undefined;
    return { status: r.status, ok: r.ok, retryAfterSec: h('retry-after'),
      requestsLeftToday: h('x-ratelimit-remaining-requests'), tokensLeftThisMinute: h('x-ratelimit-remaining-tokens') };
  };
  const results = [];
  for (let i = 0; i < keys.length; i++) {
    const row = { key: i + 1, env: names[i] };
    try {
      const chat = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: { 'Authorization': 'Bearer ' + keys[i], 'Content-Type': 'application/json' },
        body: JSON.stringify(chatBody(chatList[0], [{ role: 'user', content: 'hi' }])),
      });
      row[chatList[0]] = pick(chat);
      const form = new FormData();
      form.append('file', new Blob([silentWav()], { type: 'audio/wav' }), 'check.wav');
      form.append('model', sttList[0]);
      const stt = await fetch('https://api.groq.com/openai/v1/audio/transcriptions', {
        method: 'POST', headers: { 'Authorization': 'Bearer ' + keys[i] }, body: form,
      });
      row[sttList[0]] = pick(stt);
    } catch (e) {
      row.error = String(e && e.message || e);
    }
    results.push(row);
  }
  let openai = 'not set';
  if (process.env.OPENAI_API_KEY) {
    try {
      const r = await fetch('https://api.openai.com/v1/models', { headers: { 'Authorization': 'Bearer ' + process.env.OPENAI_API_KEY } });
      openai = r.ok ? 'ok' : 'HTTP ' + r.status;
    } catch (e) { openai = 'error'; }
  }
  return {
    groqKeys: keys.length, openaiFallback: openai,
    translationModels: chatList.slice(0, 5), sttModels: sttList,
    groqTextModelsAvailable: ids ? ids.filter((id) => !NOT_CHAT.test(id)) : 'could not list',
    results,
  };
}

module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return;
  }
  if (req.method === 'GET' && /[?&]check=/.test(req.url || '')) {
    res.status(200).json(await quotaCheck());
    return;
  }
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  try {
    const groqKeys = getGroqKeys();
    if (!groqKeys.length) {
      res.status(500).json({ error: 'Server is missing GROQ_API_KEY' });
      return;
    }

    let body = req.body;
    if (!body || typeof body === 'string') {
      body = JSON.parse(body || '{}');
    }
    const { audioBase64, mimeType } = body;
    if (!audioBase64) {
      res.status(400).json({ error: 'Missing audioBase64' });
      return;
    }
    const sourceLang = String(body.sourceLang || 'auto').toLowerCase();
    const targetLang = String(body.targetLang || 'en').toLowerCase();
    const context = Array.isArray(body.context)
      ? body.context.filter((s) => typeof s === 'string' && s.trim()).map((s) => s.slice(0, 150)).slice(-2)
      : [];
    // Conversation (two-way): both languages chosen. Whoever speaks the target
    // language gets translated back into the source language, so both sides
    // of the conversation can read each other.
    const twoWay = !!body.twoWay && sourceLang !== 'auto' && sourceLang !== targetLang;

    const buf = Buffer.from(audioBase64, 'base64');
    if (buf.length < 400) {
      res.status(200).json({ original: '', translated: '', detectedLang: '' });
      return;
    }
    const ext = (mimeType && mimeType.includes('mp4')) ? 'mp4'
      : (mimeType && mimeType.includes('ogg')) ? 'ogg'
      : (mimeType && mimeType.includes('wav')) ? 'wav'
      : 'webm';

    // One-way with a chosen language: force Whisper to that language (auto-detect
    // can mishear fast/accented speech as the wrong language entirely).
    // Two-way: let Whisper detect which of the two people is speaking.
    let stt = await transcribe(groqKeys, buf, mimeType, ext, (sourceLang !== 'auto' && !twoWay) ? sourceLang : '');
    if (stt.res.ok && twoWay && stt.text && stt.detected !== sourceLang && stt.detected !== targetLang) {
      // Detected neither of the two conversation languages: it misheard the
      // language (common with dialects). Transcribe again in the main language.
      const retry = await transcribe(groqKeys, buf, mimeType, ext, sourceLang);
      if (retry.res.ok) stt = retry;
    }

    if (!stt.res.ok) {
      const errText = await stt.res.text().catch(() => '');
      if (stt.res.status === 429) {
        res.status(429).json({ error: quotaMessage(stt.res) });
        return;
      }
      res.status(stt.res.status).json({ error: 'Transcription error: ' + errText });
      return;
    }

    const original = stt.text;
    const detectedLang = stt.detected || '';
    const toLang = (twoWay && detectedLang === targetLang) ? sourceLang : targetLang;

    if (!original) {
      res.status(200).json({ original: '', translated: '', detectedLang, toLang, debug: stt.debug });
      return;
    }

    // Skip translation if the speech is already in the language we'd translate into.
    if (detectedLang && detectedLang === toLang) {
      res.status(200).json({ original, translated: original, detectedLang, toLang });
      return;
    }

    const targetName = LANG_NAMES[toLang] || toLang || 'English';
    const messages = translationMessages(original, targetName, context);
    let chatRes = await groqWithModels(groqKeys, await availableModels(groqKeys, CHAT_MODELS, 'chat'), (key, model) => ({
      url: 'https://api.groq.com/openai/v1/chat/completions',
      options: {
        method: 'POST',
        headers: { 'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json' },
        body: JSON.stringify(chatBody(model, messages)),
      },
    }));

    if (!chatRes.ok && process.env.OPENAI_API_KEY) {
      // Groq could not translate (quota, retired models, outage): use the
      // owner's OpenAI key so translation never stops.
      const groqRes = chatRes;
      chatRes = await translateWithOpenAI(messages);
      if (!chatRes.ok) chatRes = groqRes; // report Groq's reason if OpenAI fails too
    }

    let translated = original;
    let translationError = '';
    if (chatRes.ok) {
      const chatData = await chatRes.json();
      const raw = (chatData.choices && chatData.choices[0] && chatData.choices[0].message && chatData.choices[0].message.content || '')
        .replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
      // Guard against the model refusing or commenting instead of translating
      // (e.g. "No text was provided for translation.") — fall back to showing
      // the original text rather than a confusing meta-message.
      const looksLikeRefusal = !raw || /no text|nothing to translate|لم يتم توفير|لا يوجد نص|please provide/i.test(raw);
      translated = looksLikeRefusal ? original : raw;
      if (looksLikeRefusal) translationError = 'رفض النموذج الترجمة، تم عرض النص الأصلي';
    } else {
      // v-no-fake-translate: لا نعرض النص الأصلي كأنه ترجمة — نُفرغ الترجمة
      // ونُبلغ السبب صراحةً ليعرضه العميل.
      translated = '';
      const errText = await chatRes.text().catch(() => '');
      translationError = chatRes.status === 429
        ? quotaMessage(chatRes)
        : 'فشلت الترجمة (HTTP ' + chatRes.status + '): ' + errText.slice(0, 200);
    }

    res.status(200).json({ original, translated, detectedLang, toLang, translationError });
  } catch (e) {
    res.status(500).json({ error: 'Proxy error: ' + (e && e.message ? e.message : String(e)) });
  }
};
