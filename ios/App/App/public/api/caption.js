// Vercel Serverless Function: live captioning + translation.
// 1) Transcribes a short audio chunk via Groq Whisper (auto-detects source language).
// 2) Translates the transcribed text into the requested target language via a
//    fast Groq chat model.
// Runs entirely on the site owner's own server-side GROQ_API_KEY — no user keys.
const { checkAndConsume, DAILY_LIMIT } = require('./_usage');
const { getGroqKeys, fetchWithGroqFallback } = require('./_groqKeys');

// OpenAI is used as a *paid* fallback once every free Groq key has hit its
// daily quota (HTTP 429), so live captioning never fully stops.
async function transcribeWithOpenAI(buf, mimeType, ext, sourceLang) {
  const form = new FormData();
  form.append('file', new Blob([buf], { type: mimeType || 'audio/webm' }), 'audio.' + ext);
  form.append('model', 'whisper-1');
  form.append('response_format', 'verbose_json');
  form.append('temperature', '0');
  if (sourceLang && sourceLang !== 'auto') form.append('language', sourceLang);
  const r = await fetch('https://api.openai.com/v1/audio/transcriptions', {
    method: 'POST',
    headers: { 'Authorization': 'Bearer ' + process.env.OPENAI_API_KEY },
    body: form,
  });
  return r;
}

async function translateWithOpenAI(original, targetName) {
  const r = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': 'Bearer ' + process.env.OPENAI_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: 'You are a professional simultaneous-interpretation engine for live speech. ' +
            'Translate the given transcript fragment into ' + targetName + ' with high fidelity: preserve the exact meaning, ' +
            'tone, and register of the source; do not add explanations, notes, disclaimers, or quotation marks; ' +
            'do not ask questions or comment on the input; if the fragment is a sentence fragment, translate it naturally as a fragment. ' +
            'Output ONLY the translated text and nothing else.' },
        { role: 'user', content: original },
      ],
      temperature: 0.1,
      max_tokens: 150,
    }),
  });
  return r;
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

module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(204).end();
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
    const { audioBase64, mimeType, targetLang, guestId, sourceLang } = body;
    if (!audioBase64) {
      res.status(400).json({ error: 'Missing audioBase64' });
      return;
    }

    const usage = await checkAndConsume(guestId);
    if (!usage.allowed) {
      if (usage.reason === 'limit') {
        res.status(402).json({ error: 'وصلت للحد اليومي المجاني (' + DAILY_LIMIT + ' مقطع) لهذا اليوم. حاول غدًا. / Daily free limit reached, try again tomorrow.' });
      } else {
        res.status(401).json({ error: 'Missing guest id' });
      }
      return;
    }

    const buf = Buffer.from(audioBase64, 'base64');
    if (buf.length < 400) {
      res.status(200).json({ original: '', translated: '', detectedLang: '' });
      return;
    }
    const ext = (mimeType && mimeType.includes('mp4')) ? 'mp4'
      : (mimeType && mimeType.includes('ogg')) ? 'ogg'
      : (mimeType && mimeType.includes('wav')) ? 'wav'
      : 'webm';

    const form = new FormData();
    form.append('file', new Blob([buf], { type: mimeType || 'audio/webm' }), 'audio.' + ext);
    form.append('model', 'whisper-large-v3-turbo');
    form.append('response_format', 'verbose_json');
    // Encourage the model to only transcribe real speech, not hallucinate on silence/noise.
    form.append('temperature', '0');
    // If the user told us what language they're speaking, force Whisper to
    // transcribe in that language instead of guessing (auto-detect can
    // mishear fast/accented speech as the wrong language entirely).
    if (sourceLang && sourceLang !== 'auto') form.append('language', sourceLang);

    let { res: upstream } = await fetchWithGroqFallback(groqKeys, (key) => ({
      url: 'https://api.groq.com/openai/v1/audio/transcriptions',
      options: {
        method: 'POST',
        headers: { 'Authorization': 'Bearer ' + key },
        body: form,
      },
    }));

    let usedOpenAIStt = false;
    if (!upstream.ok && upstream.status === 429 && process.env.OPENAI_API_KEY) {
      // All free Groq keys are exhausted for the day — fall back to the
      // owner's paid OpenAI Whisper API so captioning never fully stops.
      upstream = await transcribeWithOpenAI(buf, mimeType, ext, sourceLang);
      usedOpenAIStt = true;
    }

    if (!upstream.ok) {
      const errText = await upstream.text();
      if (upstream.status === 429) {
        res.status(429).json({ error: 'كل حسابات Groq المجانية وصلت حدها اليومي، حاول لاحقًا أو غدًا. / All free Groq accounts hit their daily limit, try again later or tomorrow.' });
        return;
      }
      res.status(upstream.status).json({ error: 'Transcription error: ' + errText });
      return;
    }

    const sttData = await upstream.json();
    let original = (sttData.text || '').trim();

    // ---- Silence / hallucination filter ----
    // Whisper (and similar models) can "hallucinate" short filler phrases on
    // pure silence or background noise (e.g. "you", "thank you.", "...").
    // Use the model's own confidence signals plus a small blacklist to drop
    // these instead of forwarding them to translation.
    const segments = Array.isArray(sttData.segments) ? sttData.segments : [];
    let avgNoSpeechProb = 0, avgLogProb = 0;
    if (segments.length) {
      avgNoSpeechProb = segments.reduce((s, seg) => s + (seg.no_speech_prob || 0), 0) / segments.length;
      avgLogProb = segments.reduce((s, seg) => s + (typeof seg.avg_logprob === 'number' ? seg.avg_logprob : 0), 0) / segments.length;
    }
    // Only the exact-match blacklist entries below are filtered by text; the
    // probability-based checks are intentionally conservative (high thresholds)
    // so real speech with brief pauses or moderate mic volume is never dropped.
    const HALLUCINATION_BLACKLIST = [
      'thanks for watching', 'thanks for watching!', '...', '..', '.',
      'يوتيوب', 'اشتركوا في القناة',
    ];
    const normalized = original.toLowerCase().replace(/[.,!؟?]+$/g, '').trim();
    const looksLikeHallucination =
      original.length < 2 ||
      HALLUCINATION_BLACKLIST.includes(normalized) ||
      avgNoSpeechProb > 0.85 ||
      (segments.length > 0 && avgLogProb < -1.8);

    const debug = { avgNoSpeechProb, avgLogProb, rawBeforeFilter: original, filtered: looksLikeHallucination, usedOpenAIStt };

    if (looksLikeHallucination) {
      original = '';
    }

    // Whisper's verbose_json returns a full language name (e.g. "english"), not
    // an ISO code. Map it back to our short code for a clean UI label and for
    // reliable same-language comparisons below.
    const detectedNameRaw = (sttData.language || '').toLowerCase();
    const NAME_TO_CODE = Object.fromEntries(Object.entries(LANG_NAMES).map(([code, name]) => [name.toLowerCase(), code]));
    const detectedCode = NAME_TO_CODE[detectedNameRaw] || detectedNameRaw;
    const detectedLang = detectedCode;

    if (!original) {
      res.status(200).json({ original: '', translated: '', detectedLang: detectedLang || '', debug });
      return;
    }

    // Skip translation call if source and target are effectively the same language.
    const targetName = LANG_NAMES[targetLang] || targetLang || 'English';
    const targetNorm = (targetLang || '').toLowerCase();
    const sameLang = detectedCode && detectedCode === targetNorm;

    if (sameLang) {
      res.status(200).json({ original, translated: original, detectedLang });
      return;
    }

    let { res: chatRes } = await fetchWithGroqFallback(groqKeys, (key) => ({
      url: 'https://api.groq.com/openai/v1/chat/completions',
      options: {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer ' + key,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'llama-3.1-8b-instant',
          messages: [
            { role: 'system', content: 'You are a professional simultaneous-interpretation engine for live speech. ' +
                'Translate the given transcript fragment into ' + targetName + ' with high fidelity: preserve the exact meaning, ' +
                'tone, and register of the source; do not add explanations, notes, disclaimers, or quotation marks; ' +
                'do not ask questions or comment on the input; if the fragment is a sentence fragment, translate it naturally as a fragment. ' +
                'Output ONLY the translated text and nothing else.' },
            { role: 'user', content: original },
          ],
          temperature: 0.1,
          max_tokens: 150,
        }),
      },
    }));

    if (!chatRes.ok && chatRes.status === 429 && process.env.OPENAI_API_KEY) {
      // Same daily-quota fallback for the translation step.
      chatRes = await translateWithOpenAI(original, targetName);
    }

    let translated = original;
    let translationError = '';
    if (chatRes.ok) {
      const chatData = await chatRes.json();
      const raw = (chatData.choices && chatData.choices[0] && chatData.choices[0].message && chatData.choices[0].message.content || '').trim();
      // Guard against the model refusing or commenting instead of translating
      // (e.g. "No text was provided for translation.") — fall back to showing
      // the original text rather than a confusing meta-message.
      const looksLikeRefusal = !raw || /no text|nothing to translate|لم يتم توفير|لا يوجد نص|please provide/i.test(raw);
      translated = looksLikeRefusal ? original : raw;
      if (looksLikeRefusal) translationError = 'رفض النموذج الترجمة، تم عرض النص الأصلي';
    } else {
      // Do NOT silently show the original text as if it were translated.
      // Surface the real reason so the debug panel (and future UI) can show it.
      const errText = await chatRes.text().catch(() => '');
      translationError = chatRes.status === 429
        ? 'كل حسابات Groq المجانية وصلت حدها اليومي، حاول لاحقًا أو غدًا. / All free Groq accounts hit their daily limit, try again later or tomorrow.'
        : 'فشلت الترجمة (HTTP ' + chatRes.status + '): ' + errText.slice(0, 200);
    }

    res.status(200).json({ original, translated, detectedLang, translationError });
  } catch (e) {
    res.status(500).json({ error: 'Proxy error: ' + (e && e.message ? e.message : String(e)) });
  }
};
