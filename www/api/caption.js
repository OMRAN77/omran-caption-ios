// Vercel Serverless Function: live captioning + translation.
// 1) Transcribes a short audio chunk (one spoken phrase) via Groq Whisper.
// 2) Translates the transcribed text via a Groq chat model, using the last few
//    lines of the conversation as context.
// Runs entirely on the site owner's own server-side GROQ_API_KEY — no user keys.
// Private app: there is no daily usage limit.
const zlib = require('zlib');
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
let modelIdsPending = null; // one listing at a time
async function listGroqModels(keys) {
  for (const key of keys) {
    try {
      const r = await fetch('https://api.groq.com/openai/v1/models', {
        headers: { 'Authorization': 'Bearer ' + key },
        signal: AbortSignal.timeout(3000), // a slow listing must not hold up a caption
      });
      if (!r.ok) continue;
      const j = await r.json();
      const ids = (Array.isArray(j.data) ? j.data : []).filter((m) => m && m.id && m.active !== false).map((m) => m.id);
      if (ids.length) { modelIdsCache = { at: Date.now(), ids }; return ids; }
    } catch (e) {}
  }
  modelIdsCache = { at: Date.now(), ids: null };
  return null; // unknown: fall back to trying the preference list
}
async function groqModelIds(keys) {
  const age = Date.now() - modelIdsCache.at;
  if (modelIdsCache.ids && age < 3600 * 1000) return modelIdsCache.ids;
  if (!modelIdsCache.ids && age < 10 * 60 * 1000) return null; // listing failed recently
  if (!modelIdsPending) modelIdsPending = listGroqModels(keys).finally(() => { modelIdsPending = null; });
  return modelIdsPending;
}

// Speech models without waiting for the model list: on a cold start that
// listing is one more round trip before the first caption. The list is
// fetched in the background (the translation step uses it); Whisper's
// preference order works without it, and a retired model costs one 404 and is
// then skipped for a day (groqWithModels).
function sttModelsNow(keys) {
  groqModelIds(keys).catch(() => {});
  const ids = modelIdsCache.ids;
  const have = ids ? STT_MODELS.filter((m) => ids.includes(m)) : [];
  return have.length ? have : STT_MODELS;
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

// ---- Music / silence / hallucination filter, judged per Whisper segment ----
// The app hears TikTok dramas through the mic: dialogue over non-stop music.
// Whisper then (a) transcribes the music ("♪ ... ♪", "(音乐)", lyrics) and
// (b) invents the lines its training subtitles end with ("Thank you.",
// "谢谢观看", "شكراً", "Sorry") when a clip holds no speech. Each segment is
// judged on its own, so real dialogue next to music or a made-up line is
// kept. Anything Whisper gives no numbers for is kept (never drop real words
// on a guess).
const FILTER = {
  // OpenAI's reference decoder: a segment is silence when no_speech_prob >
  // 0.6, unless Whisper is still confident in the text (avg_logprob > -1.0).
  NO_SPEECH: 0.6, NO_SPEECH_MAX_LOGPROB: -1.0,
  // Kept from the previous whole-text filter, now per segment: near-certain
  // non-speech, or text Whisper itself barely believes (mean token p < 0.17).
  NO_SPEECH_ALONE: 0.85, LOGPROB_FLOOR: -1.8,
  // OpenAI's compression_ratio_threshold: above it the text is a repetition
  // loop. The loop is collapsed to one copy; only if it still compresses this
  // well (or is just a filler/thank-you) is the segment dropped.
  COMPRESSION: 2.4,
  // "谢谢", "Thank you.", "شكراً", "you", "嗯": the classic hallucinations, but
  // also real one-word replies, and over music or from a far speaker those
  // score no_speech_prob 0.1-0.4 and avg_logprob -0.6..-0.9. Dropped only on
  // real no-speech evidence, never on avg_logprob alone.
  WEAK_NO_SPEECH: 0.4, WEAK_BOTH_NO_SPEECH: 0.2, WEAK_BOTH_LOGPROB: -1.0,
  // The same words at the end of a clip after real dialogue: dropped when
  // clearly less sure than the dialogue. Skipped when every segment carries
  // the same numbers (Whisper copies one 30 s window's numbers to each
  // segment in it), since those say nothing about the last segment.
  TRAIL_NO_SPEECH: 0.3, TRAIL_MIN_LOGPROB: -0.6, TRAIL_LOGPROB_GAP: 0.3,
  // Clear speech, for deciding which script the dialogue is in: unsure
  // segments (lyrics, mumbling) do not vote.
  SURE_NO_SPEECH: 0.2, SURE_MIN_LOGPROB: -0.5,
  // A code-switched "Sorry" / "Bye bye" in another script than the dialogue
  // is dropped only when Whisper is not sure of it.
  OTHER_NO_SPEECH: 0.1, OTHER_MIN_LOGPROB: -0.4,
};

// Compare text ignoring case, punctuation, spacing, Arabic diacritics and
// letter variants (أ/إ/آ, ى/ي, ة/ه) and full-width forms.
function normText(s) {
  return String(s || '').normalize('NFKC').toLowerCase()
    .replace(/[\u064B-\u065F\u0670\u0640]/g, '')
    .replace(/[أإآٱ]/g, 'ا').replace(/ى/g, 'ي').replace(/ة/g, 'ه')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ').trim();
}
const compactText = (s) => normText(s).replace(/\s+/g, '');
const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// "Thanks for watching" lines Whisper invents at the end of a clip (its
// training subtitles end that way). Never dialogue in these videos, but Whisper
// may append one to a real line, so only the phrase itself is cut and what is
// left is judged on its own.
// The Chinese family is generated: 谢谢/感谢/多谢 + (大家/各位/您/你/你们) + (的) +
// 观看/收看/收听/观赏. Traditional characters are matched by the regex below.
const OUTRO_ZH = [];
for (const a of ['谢谢', '感谢', '多谢']) for (const b of ['', '大家', '各位', '您', '你', '你们']) {
  for (const c of ['', '的']) for (const d of ['观看', '收看', '收听', '观赏']) OUTRO_ZH.push(a + b + c + d);
}
const STRONG_PHRASES = uniq([
  'thanks for watching', 'thank you for watching', 'thank you so much for watching',
  'thank you very much for watching', 'thanks so much for watching', 'see you in the next video', 'see you in the next one',
  ...OUTRO_ZH, '下期再见', '我们下期再见', '下期见', '我们下期见', '下集再见', '我们下集再见', '本期视频就到这里',
  'ご視聴ありがとうございました', 'ご視聴ありがとうございます', 'ご視聴いただきありがとうございました',
  '最後までご視聴いただきありがとうございました', '最後までご視聴ありがとうございました',
  '시청해주셔서 감사합니다', '시청해주셔서 고맙습니다',
  'ترجمة نانسي قنقر', 'شكرا للمشاهدة', 'شكرا على المشاهدة', 'شكرا لكم على المشاهدة', 'شكرا لكم للمشاهدة',
  'شكرا جزيلا على المشاهدة', 'شكرا جزيلا لكم على المشاهدة', 'شكرا لمشاهدتكم', 'شكرا جزيلا لمشاهدتكم',
  'شكرا على المتابعة', 'شكرا لكم على المتابعة', 'نراكم في الحلقة القادمة', 'نلقاكم في الحلقة القادمة',
  'الى اللقاء في الحلقة القادمة',
]);
// Simplified -> Traditional forms matched in the same place.
const TRAD = { '谢': '謝', '观': '觀', '见': '見', '们': '們', '听': '聽', '赏': '賞', '视': '視', '频': '頻', '这': '這', '里': '裡裏' };
// Matches the phrase in the raw text despite punctuation, spaces, diacritics
// and Arabic letter variants between its letters — only where it ends the
// text or a sentence (also before a space, "~", an emoji, or for English
// "guys/everyone" and "and see you..."), the way Whisper appends it, so a
// real line that merely contains the words ("谢谢观看我的表演", "Thank you for
// watching my show") is never cut.
// Each regex is built and run only when the compact text contains its phrase
// (any match implies that), so the ~170 regexes cost nothing until needed:
// compiling all of them on the first request of a cold start cost ~280 ms.
const TRAD_BACK = {};
for (const [s, t] of Object.entries(TRAD)) for (const c of t) TRAD_BACK[c] = s;
const TRAD_BACK_RE = new RegExp('[' + Object.keys(TRAD_BACK).join('') + ']', 'g');
const foldTrad = (k) => k.replace(TRAD_BACK_RE, (c) => TRAD_BACK[c]);
const strongKey = (s) => foldTrad(compactText(s));
const STRONG = STRONG_PHRASES.map(compactText).sort((a, b) => b.length - a.length).map((p) => {
  const gap = '[\\s\\p{P}\\p{S}\\u064B-\\u065F\\u0670\\u0640]*';
  const cls = (c) => c === 'ا' ? '[اأإآٱ]' : c === 'ي' ? '[يى]' : c === 'ه' ? '[هة]' : TRAD[c] ? '[' + c + TRAD[c] + ']' : escapeRe(c);
  const cjk = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u.test(p);
  const end = cjk ? '(?=$|[\\s\\p{P}\\p{S}\\p{M}])'
    : '(?=[\\s\\p{S}\\p{M}]*(?:$|\\p{P})|\\s+and\\s*(?:$|\\p{P})|\\s+(?:and\\s+)?see\\s+you\\b|' +
      '\\s+(?:guys|everyone|everybody|folks|y\\W?all)(?=[\\s\\p{S}\\p{M}]*(?:$|[!.?,;:…])))';
  // "~", "～", "❤️", "😊" right after the phrase go with it.
  // Built on first use: constructing all of them cost ~85 ms per cold start.
  return { p: foldTrad(p), src: Array.from(p).map(cls).join(gap) + '(?:\\s*[\\p{S}\\p{M}])*' + end, re: null };
});
// Subtitle credits and "subscribe to the channel" lines Whisper prints over
// music or silence: never dialogue, so the whole segment goes.
const CREDIT_RE = new RegExp([
  'amara\\s*org', 'castingwords', 'subtitles?\\s+(by|created by|provided by|made by)', 'captions?\\s+by',
  'transcri(bed|ption)\\s+by', '(please|don\\s*t\\s+forget\\s+to)\\s+(like\\s+(and\\s+)?)?subscribe',
  'like\\s+(and\\s+)?subscribe', 'subscribe\\s+to\\s+(my|our|the|this)\\s+channel',
  '字幕\\s*(by|由|制作|提供|组|組|志愿|志願|翻译|翻譯|校对|校對)', '中文字幕',
  '明镜与点点|明鏡與點點|点点栏目|點點欄目|支持明镜|支持明鏡|明镜需要|明鏡需要', '优优独播|優優獨播|yoyo\\s*television',
  '打赏支持|打賞支持|不吝点赞|不吝點贊', '请订阅|請訂閱|订阅.{0,6}频道|訂閱.{0,6}頻道|点赞.{0,4}订阅|點贊.{0,4}訂閱',
  // "请点赞关注", "点赞、关注、转发", "记得一键三连哦": the line starts with the
  // request (so "你给我点赞关注了吗" stays dialogue).
  '^(大家|各位)?\\s*(请|請|记得|記得|求|欢迎|歡迎|别忘了|別忘了|不要忘记|不要忘記)?\\s*' +
    '((点赞|點贊|点个赞|點個贊)\\s*(和|与|與|加|并|並)?\\s*(关注|關注|转发|轉發|订阅|訂閱|收藏)|一键三连|一鍵三連)',
  '^字幕\\s', '以上言论不代表本台立场|以上言論不代表本台立場',
  '^(作词|作曲|编曲|作詞|編曲|词曲|詞曲|演唱|原唱)\\s', '^词\\s.*\\s曲\\s',
  // "作词李宗盛": credit word + a 2-3 character name and nothing else; not
  // "作词的人是谁" / "作词写得真好".
  '^(作词|作詞|词曲|詞曲)(?![的是了吗呢啊人家很也都就还写寫得真太在不没])\\p{Script=Han}{2,3}$',
  'チャンネル登録', '구독\\s*(과|와)?\\s*좋아요|좋아요\\s*(와|과)?\\s*구독|구독\\s*부탁', '(mbc|kbs|sbs|ytn|jtbc)\\s*뉴스.*입니다',
  'نانسي\\s*قنقر', 'اشترك(وا)?\\s*(في|ب)\\s*القناه', 'الاشتراك\\s*(في|ب)\\s*القناه',
].join('|'), 'u');
// Whole segment is only one of these: not dialogue.
const STRONG_EXACT = new Set(['youtube', 'يوتيوب', '字幕', 'subtitles', 'subtitle'].map(compactText));
// Whole segment is only music / sound words.
const MUSIC_ONLY_RE = new RegExp('^(?:' + ['music', 'musik', 'musique', 'musica', 'bgm', 'instrumental', 'applause',
  'laughter', 'laughs', 'silence', 'blank audio', 'no audio', 'no speech', 'inaudible', 'singing',
  '音乐', '音樂', '背景音乐', '背景音樂', '纯音乐', '純音樂', '轻音乐', '輕音樂', '配乐', '配樂', '伴奏',
  '掌声', '掌聲', '笑声', '笑聲', '音楽', '拍手', '음악', '박수', 'موسيقى', 'موسيقية', 'تصفيق', 'صمت', 'ضحك',
].map(compactText).map(escapeRe).join('|') + ')+$', 'u');
// Short courtesy words and fillers: real dialogue sometimes, hallucination
// often. A segment made only of these needs Whisper to be sure of it.
// Answer words ("好", "OK", "yeah", "はい", "네") and address words ("大家",
// "everyone") are not on the list: they are not known hallucinations, and as
// parts they turned real lines ("大家好", "You okay?", "好啊") into "courtesy".
const WEAK_PHRASES = [
  'thank you', 'thanks', 'thank you very much', 'thank you so much', 'thanks a lot', 'sorry', 'i\'m sorry',
  'i am sorry', 'so sorry', 'you', 'bye', 'bye bye', 'goodbye', 'see you', 'see you next time',
  'see you soon', 'thanks for listening', 'thank you for listening',
  '谢谢', '谢谢你', '谢谢您', '谢谢大家', '谢谢你们', '多谢', '感谢', '非常感谢', '对不起', '不好意思', '抱歉',
  '再见', '拜拜', '謝謝', '謝謝你', '謝謝大家', '對不起', '再見',
  'ありがとう', 'ありがとうございます', 'ありがとうございました', 'ごめんなさい', 'すみません',
  'おやすみなさい', 'お疲れ様でした',
  '감사합니다', '고맙습니다', '고마워요', '고마워', '죄송합니다', '미안합니다', '미안해요', '미안해', '안녕', '안녕히 계세요',
  'شكرا', 'شكرا لك', 'شكرا لكم', 'شكرا جزيلا', 'آسف', 'آسفة', 'أنا آسف', 'أنا آسفة', 'عذرا', 'معذرة',
  'مع السلامة', 'السلام عليكم',
];
// The end-of-video thank-you family: dropped in another script than the
// dialogue even when Whisper sounds sure ("Thank you." after Chinese lines).
const THANKS_PHRASES = ['thank you', 'thanks', 'thank you very much', 'thank you so much', 'thanks a lot', 'you',
  'شكرا', 'شكرا لك', 'شكرا لكم', 'شكرا جزيلا', 'ありがとう', 'ありがとうございます', 'ありがとうございました', '감사합니다', '고맙습니다'];
// What may follow a cut "thanks for watching" and still be part of it.
const OUTRO_REST_PHRASES = WEAK_PHRASES.concat(['ok', 'okay', 'everyone', 'everybody', 'guys', 'you guys', 'folks', 'y all',
  'all', 'friends', 'and', 'so', 'also', 'then', '大家', '各位', '朋友们', '朋友們', '然后', '还有', '那么', 'みなさん', '皆さん',
  '여러분', 'الجميع', 'يا جماعة', 'و', 'ثم']);
// Each filler matches a whole run ("hmmm", "啊啊啊"), so there is only one way to
// split a text into fillers: no exponential backtracking on "啊"x30 + "你".
const FILLERS = ['h*m+(?!m)', 'u+[hm]+(?![hm])', 'a+h+(?!h)', 'o+h+(?!h)', 'e+h+(?!h)', '嗯+(?!嗯)', '啊+(?!啊)',
  '哦+(?!哦)', '呃+(?!呃)', '唉+(?!唉)', '诶+(?!诶)', '欸+(?!欸)', '哎+(?!哎)', 'あ+(?!あ)', 'え+(?!え)', 'う+ん+(?!ん)',
  '음+(?!음)', '아+(?!아)', '어+(?!어)', 'اه+(?!ه)', 'امم+(?!م)'].join('|');
const phraseAlt = (list) => uniq(list.map(compactText)).sort((a, b) => b.length - a.length).map(escapeRe).join('|');
// Only fillers, or ONE phrase (repeated: "谢谢谢谢", "Thank you. Thank you.")
// with fillers around it. Two different words ("哦，好的，谢谢") are dialogue.
const onePhraseRe = (list) => new RegExp('^(?:(?:' + FILLERS + ')+|(?:' + FILLERS + ')*(' + phraseAlt(list) + ')(?:\\1)*(?:' + FILLERS + ')*)$', 'u');
const WEAK_RE = onePhraseRe(WEAK_PHRASES);
const THANKS_RE = onePhraseRe(THANKS_PHRASES);
const OUTRO_REST_RE = new RegExp('^(?:' + phraseAlt(OUTRO_REST_PHRASES) + '|' + FILLERS + ')+$', 'u');

// Sound descriptions Whisper writes in brackets: "(音乐)", "(laughs)", "*music*".
const LABEL_LATIN = /\b(music|musical|musique|song|singing|sings|humming|hums|instrumental|melody|bgm|applause|clapping|claps|laugh\w*|chuckl\w*|giggl\w*|cheer\w*|sigh\w*|gasp\w*|cries|crying|sobb?\w*|cough\w*|sneez\w*|breath\w*|inaudible|unintelligible|indistinct|silence|silent|blank_audio|no audio|no speech|noise|static|beep\w*|ringing|footsteps|sounds?|sfx|background|foreign language|speaking \w+)\b/i;
// Sound effects and instruments: "(door closes)", "(soft piano)". Only in a
// short bracket (4 words or fewer), so "(I left the door open)" is kept.
const LABEL_SFX = /\b(piano|guitar\w*|violin\w*|drums?|drumming|strumm\w*|chirp\w*|bark\w*|knock\w*|doors?|thunder|rain|wind|bells?|engine|explosions?|gunshots?|whoosh\w*)\b/i;
// "(钢琴曲)", "（鸟叫声）", "(插曲)": 声/曲/instrument words, only in a bracket of
// 8 characters or fewer.
const LABEL_OTHER = /音乐|音樂|配乐|配樂|伴奏|歌|唱|笑|掌声|掌聲|鼓掌|欢呼|歡呼|叹气|嘆氣|哭|咳|喘|呼吸|噪音|杂音|雜音|静音|靜音|无声|無聲|空白|字幕|片头|片尾|片頭|声|聲|曲|钢琴|鋼琴|吉他|琴|鼓|铃|鈴|音楽|拍手|ため息|음악|박수|웃음|한숨|موسيق|تصفيق|ضحك|صمت|غناء|اغنيه/u;
function isSoundLabel(content) {
  const k = compactText(content);
  const words = normText(content).split(' ').filter(Boolean).length;
  return !k || LABEL_LATIN.test(content) || (words <= 4 && LABEL_SFX.test(content)) || (k.length <= 8 && LABEL_OTHER.test(k));
}
// Whole segment that is only sung syllables or a plain music caption:
// "La la la la.", "啦啦啦啦", "轻快的音乐", "音乐声", "موسيقى هادئة". Not "Nana!",
// "我的音乐", "音乐声太大了".
const MUSIC_ADJ = '轻快|輕快|悲伤|悲傷|欢快|歡快|舒缓|舒緩|激昂|紧张|緊張|抒情|温柔|溫柔|浪漫|恐怖|神秘|优美|優美|动感|動感|伤感|傷感|柔和|轻柔|輕柔|安静|安靜';
const MUSIC_TEXT_RE = new RegExp([
  '^(?:la|na|lu|da|doo|dum){3,}$', '^[啦呐哒嘟]{3,}$',
  '^(?:(?:' + MUSIC_ADJ + ')的?)?(?:背景)?(?:音乐|音樂|钢琴曲|鋼琴曲|纯音乐|純音樂)(?:声|聲)$',
  '^(?:' + MUSIC_ADJ + ')的?(?:背景)?(?:音乐|音樂)$',
  '^موسيقي(?:هادئه|حزينه|دراميه|تصويريه|خفيفه|صاخبه|رومانسيه)$',
].join('|'), 'u');

// Removes sound labels and sung lines; reports whether music was marked.
function stripNonSpeech(text) {
  let music = false;
  let s = text.replace(/\[[^\][]{0,80}\]|【[^】【]{0,80}】|〔[^〕〔]{0,80}〕/g, () => { music = true; return ' '; });
  s = s.replace(/[(（]([^()（）]{0,40})[)）]/g, (m, c) => isSoundLabel(c) ? (music = true, ' ') : m);
  s = s.replace(/\*([^*]{1,40})\*/g, (m, c) => isSoundLabel(c) ? (music = true, ' ') : c);
  // Whisper writes singing as "♪ words ♪". Runs of notes count as one mark;
  // a line that opens or closes with an unpaired note is sung too, unless it
  // carries sentence punctuation (lyrics almost never do, dialogue does):
  // "♪♪ 你怎么来了？" is music fading into speech, so only the note goes.
  if (/[♩♪♫♬🎵🎶🎼]/u.test(s)) {
    music = true;
    s = s.replace(/[♩♪♫♬🎵🎶🎼](?:[\s♩♪♫♬🎵🎶🎼]*[♩♪♫♬🎵🎶🎼])?/gu, '♪').replace(/♪[^♪]*♪/g, ' ');
    const t = s.trim();
    s = ((t.startsWith('♪') || t.endsWith('♪')) && !/[？！?!。]/u.test(t)) ? '' : s.replace(/♪/g, ' ');
  }
  return { text: s, music };
}

// "A A A A" (a unit repeated 4+ times in a row, 8+ characters in all) -> "A".
// A unit without a letter is never collapsed, so numbers stay exact
// ("1000" is not "10", "13800000000" is not "1380"), and a short real
// repeat ("对不起对不起对不起", "哈哈哈") is left as it was said.
function collapseRepeats(s) {
  return s.replace(/(.{1,40}?)(?:[\s,.!?;:，。！？、；：…~～-]*\1){3,}/gsu,
    (m, unit) => (m.length >= 8 && /\p{L}/u.test(unit) ? unit : m));
}
// Same measure as Whisper's compression_ratio: utf-8 bytes / zlib bytes.
function compressionRatio(s) {
  const b = Buffer.from(s, 'utf8');
  return b.length ? b.length / zlib.deflateSync(b).length : 0;
}
// A single Chinese/Japanese/Korean character is a word; elsewhere 2+ letters.
function hasWords(s) {
  return /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u.test(s) ||
    (s.match(/[\p{L}\p{N}]/gu) || []).length >= 2;
}
function tidy(s) {
  return s.replace(/\s+/g, ' ').replace(/\s+([，。！？、；：,.!?;:])/g, '$1')
    .replace(/([，。！？、；：,.!?;:])[，。！？、；：,.!?;:]+/g, '$1')
    .replace(/^[\s，。！？、；：,.!?;:]+/, '').trim();
}
function scriptOf(s) {
  if (/[\p{Script=Hiragana}\p{Script=Katakana}]/u.test(s)) return 'kana';
  if (/\p{Script=Hangul}/u.test(s)) return 'hangul';
  const n = (re) => (s.match(re) || []).length;
  const counts = { han: n(/\p{Script=Han}/gu), arabic: n(/\p{Script=Arabic}/gu), latin: n(/\p{Script=Latin}/gu), cyrillic: n(/\p{Script=Cyrillic}/gu) };
  const best = Object.keys(counts).sort((a, b) => counts[b] - counts[a])[0];
  return counts[best] ? best : '';
}
const numOrUndef = (x) => (typeof x === 'number' && Number.isFinite(x) ? x : undefined);

// Judges one segment: returns the cleaned text, or a reason to drop it.
function judgeSegment(rawText, nsp, lp, cr) {
  const v = { text: String(rawText || '').trim(), drop: '', weak: false, looped: false };
  if (nsp !== undefined && lp !== undefined && nsp > FILTER.NO_SPEECH && lp < FILTER.NO_SPEECH_MAX_LOGPROB) { v.drop = 'no_speech'; return v; }
  if (nsp !== undefined && nsp > FILTER.NO_SPEECH_ALONE) { v.drop = 'no_speech'; return v; }
  if (lp !== undefined && lp < FILTER.LOGPROB_FLOOR) { v.drop = 'low_confidence'; return v; }
  if ((cr !== undefined ? cr : compressionRatio(v.text)) > FILTER.COMPRESSION) {
    const once = collapseRepeats(v.text);
    v.looped = once !== v.text;
    if (compressionRatio(once) > FILTER.COMPRESSION) { v.drop = 'repetition_loop'; return v; }
    v.text = once;
  }
  const cleaned = stripNonSpeech(v.text);
  let t = cleaned.text;
  if (CREDIT_RE.test(normText(t))) { v.drop = 'hallucination'; return v; }
  let strong = false;
  // "谢谢观看 下期再见": any order; "谢谢观看谢谢观看谢谢观看谢谢观看": one copy per pass.
  for (let pass = 0, again = true; again && pass < 12; pass++) {
    again = false;
    const key = strongKey(t);
    for (const s of STRONG) {
      if (!key.includes(s.p)) continue;
      s.re = s.re || new RegExp(s.src, 'giu');
      t = t.replace(s.re, () => { strong = again = true; return ' '; });
    }
  }
  if (t !== v.text) t = tidy(t); // only touch the text when something was removed
  const k = compactText(t);
  if (MUSIC_ONLY_RE.test(k) || MUSIC_TEXT_RE.test(k) || (!hasWords(t) && cleaned.music)) { v.drop = 'music'; return v; }
  if (!hasWords(t)) { v.drop = strong ? 'hallucination' : 'no_words'; return v; }
  if (STRONG_EXACT.has(k)) { v.drop = 'hallucination'; return v; }
  // A loop is judged on the words it was made of too: "谢谢。谢谢。…" collapses
  // to "谢。", which is half of a courtesy word.
  v.weak = WEAK_RE.test(k) || (v.looped && (WEAK_RE.test(k + k) || WEAK_RE.test(compactText(stripNonSpeech(String(rawText || '')).text))));
  // "Thanks for watching, (and) see you soon!", "…, guys!", "谢谢观看，记得点赞关注哦":
  // what is left belongs to the made-up line.
  if (strong && (v.weak || (k.length <= 60 && OUTRO_REST_RE.test(k)) || CREDIT_RE.test(normText(t)))) { v.drop = 'hallucination'; return v; }
  if (v.weak && v.looped) { v.drop = 'repetition_loop'; return v; }
  if (v.weak && nsp !== undefined && (nsp > FILTER.WEAK_NO_SPEECH ||
      (nsp > FILTER.WEAK_BOTH_NO_SPEECH && lp !== undefined && lp < FILTER.WEAK_BOTH_LOGPROB))) {
    v.drop = 'weak_phrase_unsure'; return v;
  }
  v.text = t;
  return v;
}

// Applies the filter to Whisper's verbose_json. `opts.context` = the last
// lines heard, `opts.oneWay` = watching media (not a two-person conversation).
function filterTranscript(data, opts = {}) {
  const rawText = String(data.text || '').trim();
  const list = Array.isArray(data.segments) ? data.segments.filter((s) => s && typeof s === 'object') : [];
  const avg = (key) => {
    const xs = list.map((s) => numOrUndef(s[key])).filter((x) => x !== undefined);
    return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : undefined;
  };
  // Segments without their own text (other Whisper hosts) are judged as one
  // piece with Whisper's averaged numbers (the highest compression ratio).
  const perSegment = list.length > 0 && list.every((s) => typeof s.text === 'string') && list.some((s) => s.text.trim());
  const crs = list.map((s) => numOrUndef(s.compression_ratio)).filter((x) => x !== undefined);
  const segs = perSegment
    ? list.map((s) => ({ raw: s.text.trim(), nsp: numOrUndef(s.no_speech_prob), lp: numOrUndef(s.avg_logprob), cr: numOrUndef(s.compression_ratio), start: s.start, end: s.end }))
    : [{ raw: rawText, nsp: avg('no_speech_prob'), lp: avg('avg_logprob'), cr: crs.length ? Math.max(...crs) : undefined }];

  const judged = segs.map((s) => Object.assign(s, judgeSegment(s.raw, s.nsp, s.lp, s.cr)));
  const kept = () => judged.filter((s) => !s.drop);

  // Courtesy words at the end of a clip after real dialogue (the whole run of
  // them, "…谢谢。再见。"): dropped when clearly less sure than the dialogue.
  const ks = kept();
  let tail = ks.length;
  while (tail > 0 && ks[tail - 1].weak) tail--;
  const before = ks.slice(0, tail).filter((s) => !s.weak);
  if (before.length) {
    const lps = before.map((s) => s.lp).filter((x) => x !== undefined);
    const meanLp = lps.length ? lps.reduce((a, b) => a + b, 0) / lps.length : undefined;
    for (const last of ks.slice(tail)) {
      // Same numbers as the dialogue: copied from one window, no evidence.
      if (before.every((s) => s.nsp === last.nsp && s.lp === last.lp)) continue;
      if ((last.nsp !== undefined && last.nsp > FILTER.TRAIL_NO_SPEECH) ||
          (last.lp !== undefined && meanLp !== undefined && last.lp < FILTER.TRAIL_MIN_LOGPROB &&
           last.lp < meanLp - FILTER.TRAIL_LOGPROB_GAP)) {
        last.drop = 'weak_phrase_trailing';
      }
    }
  }
  // Watching media (not a two-language conversation): a courtesy word in
  // another script than the dialogue just heard — the previous lines and the
  // clear speech in this clip, e.g. "Thank you." / "شكراً" after Chinese —
  // comes from the music, not from the actors. A code-switched "Sorry" /
  // "Bye bye" is real often enough that it also needs Whisper to be unsure.
  const sure = (s) => !(s.nsp > FILTER.SURE_NO_SPEECH) && !(s.lp < FILTER.SURE_MIN_LOGPROB);
  const heard = (opts.context || []).concat(kept().filter((s) => !s.weak && sure(s)).map((s) => s.text)).map(scriptOf).filter(Boolean);
  if (opts.oneWay && heard.length && heard.every((c) => c === heard[0])) {
    for (const s of kept()) {
      const sc = scriptOf(s.text);
      if (!s.weak || !sc || sc === heard[0]) continue;
      if (THANKS_RE.test(compactText(s.text)) || (s.nsp !== undefined && s.nsp > FILTER.OTHER_NO_SPEECH) ||
          (s.lp !== undefined && s.lp < FILTER.OTHER_MIN_LOGPROB)) s.drop = 'weak_phrase_other_language';
    }
  }

  const keptSegs = kept();
  const changed = keptSegs.length !== judged.length || keptSegs.some((s) => s.text !== s.raw);
  let text = '';
  if (!changed) text = rawText; // nothing removed or cleaned: Whisper's own text, untouched
  else {
    for (const s of keptSegs) {
      const t = s.text.trim();
      if (t) text += (text && !/[\u3000-\u303F\uFF00-\uFFEF]$/u.test(text) ? ' ' : '') + t;
    }
  }
  if (!hasWords(text)) text = '';

  const r3 = (x) => (x === undefined ? undefined : Math.round(x * 1000) / 1000);
  return {
    text,
    debug: {
      avgNoSpeechProb: avg('no_speech_prob') ?? 0, avgLogProb: avg('avg_logprob') ?? 0,
      rawBeforeFilter: rawText, filtered: !!rawText && !text, perSegment,
      segments: judged.map((s, i) => ({ i, start: s.start, end: s.end, text: s.raw.slice(0, 80),
        kept: !s.drop, reason: s.drop || undefined, cleaned: !s.drop && s.text !== s.raw ? s.text.slice(0, 80) : undefined,
        noSpeechProb: r3(s.nsp), avgLogProb: r3(s.lp), compressionRatio: r3(s.cr) })),
    },
  };
}

async function transcribe(keys, buf, mimeType, ext, lang, filterOpts) {
  let res = await groqWithModels(keys, sttModelsNow(keys), (key, model) => {
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
  const { text, debug } = filterTranscript(data || {}, filterOpts);
  debug.usedOpenAI = usedOpenAI;

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
    const targetLang = String(body.targetLang || 'en').toLowerCase();
    let sourceLang = String(body.sourceLang || 'auto').toLowerCase();
    // Speaking and translation set to the same language almost always means
    // foreign-language media watched with captions in the user's language
    // (e.g. a Chinese video with Arabic chosen in both boxes). Forcing Whisper
    // to that language makes it guess garbled text from the foreign speech, so
    // detect the spoken language instead.
    if (sourceLang === targetLang) sourceLang = 'auto';
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
    const filterOpts = { context, oneWay: !twoWay };
    let stt = await transcribe(groqKeys, buf, mimeType, ext, (sourceLang !== 'auto' && !twoWay) ? sourceLang : '', filterOpts);
    if (stt.res.ok && twoWay && stt.text && stt.detected !== sourceLang && stt.detected !== targetLang) {
      // Detected neither of the two conversation languages: it misheard the
      // language (common with dialects). Transcribe again in the main language.
      const retry = await transcribe(groqKeys, buf, mimeType, ext, sourceLang, filterOpts);
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
    // debug (per-segment Whisper numbers and filter reasons) rides along on
    // every reply, so the filter thresholds can be tuned from real phone audio;
    // the app reads only original/translated/translationError.
    if (detectedLang && detectedLang === toLang) {
      res.status(200).json({ original, translated: original, detectedLang, toLang, debug: stt.debug });
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
      // Also "Sorry, I can't translate this" / "عذراً، لا أستطيع ترجمة..." on
      // noise — unless the line itself talks about translating.
      const looksLikeRefusal = !raw || /no text|nothing to translate|لم يتم توفير|لا يوجد نص|please provide/i.test(raw) ||
        (!/翻译|翻譯|翻訳|번역|translat|ترجم/i.test(original) &&
          /(can ?not|can't|unable to|not able to) translate|(لا\s*(أستطيع|يمكنني|يمكن)|لم\s*(أتمكن|أستطع))\s*(من\s*)?(ال)?ترجم|النص\s*(غير\s*(واضح|مفهوم|مكتمل)|فارغ)/i.test(raw));
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

    res.status(200).json({ original, translated, detectedLang, toLang, translationError, debug: stt.debug });
  } catch (e) {
    res.status(500).json({ error: 'Proxy error: ' + (e && e.message ? e.message : String(e)) });
  }
};
