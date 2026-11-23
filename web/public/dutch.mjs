// Browser port of src/tiny_voca/dutch.py and the shared eSpeak IPA parser.
const SMALL = 'nul een twee drie vier vijf zes zeven acht negen tien elf twaalf dertien veertien vijftien zestien zeventien achttien negentien'.split(' ');
const TENS = ['', '', 'twintig', 'dertig', 'veertig', 'vijftig', 'zestig', 'zeventig', 'tachtig', 'negentig'];
const MONTHS = 'januari februari maart april mei juni juli augustus september oktober november december'.split(' ');
const ABBR = new Map(Object.entries({'d.w.z.':'dat wil zeggen', bijv:'bijvoorbeeld', 'bijv.':'bijvoorbeeld', 'm.a.w.':'met andere woorden', 'a.u.b.':'alstublieft', 'dhr.':'de heer', 'mevr.':'mevrouw', 'dr.':'dokter', 'prof.':'professor', 'nr.':'nummer'}));
const WORD = '[\\p{L}\\p{N}_]';
const ABBR_RE = new RegExp(`(?<!${WORD})(?:${[...ABBR.keys()].sort((a,b) => b.length-a.length).map(x => x.replaceAll('.', '\\.')).join('|')})(?!${WORD})`, 'giu');
const NUMBER = '(?:(?:\\d{1,3}(?:\\.\\d{3})+)|\\d+)';
const PUNCT = /^[,.?!;:]$/;
const SPACE = /[\p{White_Space}\u001c-\u001f]+/gu;
const HAS_WORD = /[\p{L}\p{N}]/u;

function cardinal(n) {
  n = BigInt(n);
  if (n < 20n) return SMALL[Number(n)];
  if (n < 100n) {
    const unit = Number(n % 10n);
    return (unit ? SMALL[unit] + ([2, 3].includes(unit) ? 'ën' : 'en') : '') + TENS[Number(n / 10n)];
  }
  for (const [scale, word] of [[1000000000n, ' miljard '], [1000000n, ' miljoen '], [1000n, 'duizend'], [100n, 'honderd']]) {
    if (n < scale) continue;
    const hi = n / scale, lo = n % scale;
    return ((hi === 1n && scale <= 1000n ? '' : cardinal(hi)) + word + (lo ? cardinal(lo) : '')).trim();
  }
}

const year = n => n >= 1100 && n < 2000 ? SMALL[Math.floor(n / 100)] + 'honderd' + (n % 100 ? cardinal(n % 100) : '') : cardinal(n);
const number = s => BigInt(s.replaceAll('.', ''));

export function spokenText(text) {
  text = text.normalize('NFKC').replace(/[‘’‚‛`´]/g, "'").replace(/[“”„‟«»]/g, '"')
    .replace(/[–―]/g, '—').replace(/\[|\{/g, '(').replace(/\]|\}/g, ')')
    .replace(/\.{3,}/g, '…').replace(/-{2,}|\s-\s/g, '—').replace(SPACE, ' ').trim();
  text = text.replace(/\p{Nd}/gu, d => {
    let n = d.codePointAt(0), zero = n;
    while (zero > 0 && /\p{Nd}/u.test(String.fromCodePoint(zero - 1))) zero--;
    return String((n - zero) % 10);
  });
  const bounded = pattern => new RegExp(`(?<!${WORD})${pattern}(?!${WORD})`, 'gu');
  text = text.replace(new RegExp(`(?<!${WORD})[−-](?=\\s*[\\d€])`, 'gu'), 'min ')
    .replace(ABBR_RE, m => ABBR.get(m.toLowerCase()));
  const date = (match, a, b, c) => {
    const [y, month, day] = (a.length === 4 ? [a,b,c] : [c,b,a]).map(Number);
    const leap = y % 4 === 0 && (y % 100 !== 0 || y % 400 === 0);
    const days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if (!y || !days[month-1] || day < 1 || day > days[month-1]) throw Error(`Ongeldige datum: ${match}`);
    return `${cardinal(day)} ${MONTHS[month-1]} ${year(y)}`;
  };
  text = text.replace(bounded('(\\d{4})-(\\d{1,2})-(\\d{1,2})'), date)
    .replace(bounded('(\\d{1,2})[-/](\\d{1,2})[-/](\\d{4})'), date);
  const amount = (_, a, b) => { const cents = Number((b || '0').padEnd(2, '0')); return `${cardinal(number(a))} euro${cents ? ` ${cardinal(cents)} cent` : ''}`; };
  text = text.replace(new RegExp(`€\\s*(${NUMBER})(?:,(\\d{1,2}))?(?!\\d|,\\d)`, 'g'), amount)
    .replace(new RegExp(`(?<!${WORD})(${NUMBER}),(\\d{1,2})\\s+euro(?!${WORD})`, 'giu'), amount);
  const clock = (match, h, m) => {
    if (+h > 23 || +m > 59) throw Error(`Ongeldige tijd: ${match}`);
    return `${cardinal(h)} uur${+m ? ` ${cardinal(m)}` : ''}`;
  };
  return text.replace(new RegExp(`(?<!${WORD})(\\d{1,2})[:.](\\d{2})\\s+uur(?!${WORD})`, 'giu'), clock)
    .replace(bounded('(\\d{1,2}):(\\d{2})'), clock)
    .replace(bounded(`(${NUMBER}),(\\d+)`), (_, a, b) => `${cardinal(number(a))} komma ${[...b].map(d => SMALL[+d]).join(' ')}`)
    .replace(/(?<=\d)\s*%/g, ' procent').replace(bounded('1[1-9]\\d\\d'), n => year(+n))
    .replace(new RegExp(NUMBER, 'g'), n => cardinal(number(n)))
    .replaceAll('"', '').replaceAll('…', '.').replaceAll('—', ',').replace(SPACE, ' ').trim();
}

export function sentences(text) {
  const body = text.trimEnd();
  const last = body.split(/\s+/).at(-1);
  const marked = HAS_WORD.test(last) ? /[.!?…,;:]$/.test(last.replace(/["'”’)\]»]+$/g, '')) : /[.!?…]|^[,;:]$/.test(last);
  return spokenText(!body || marked ? text : body + '.').split(/(?<=[.!?])\s+/).filter(s => HAS_WORD.test(s));
}

export function createPhonemizer(symbols, trace) {
  const ids = new Map(symbols.map((symbol, i) => [symbol, i]));
  const isPhone = s => ids.has(s) && !s.startsWith('<') && !PUNCT.test(s);
  return text => {
    const found = ['<bos>'], stress = [0];
    const push = (phone, level = 0) => { found.push(phone); stress.push(level); };
    for (const piece of spokenText(text).split(/([,.?!;:])/)) {
      if (PUNCT.test(piece)) {
        if (found.at(-1) !== piece) push(piece);
      } else if (piece.trim()) {
        const words = trace(piece.trim()).replace(/\((?:_?[a-z0-9]_?){2,3}(?:-(?:_?[a-z0-9]_?)+)*\)/g, '').trim().split(/\s+/);
        for (const word of words) {
          const phones = word.split('_').map(unit => [unit.replace(/[ˈˌ()\p{Cf}]/gu, '').normalize('NFC'), unit.includes('ˈ') ? 1 : unit.includes('ˌ') ? 2 : 0]).filter(([phone]) => phone);
          if (!phones.length) continue;
          if (isPhone(found.at(-1))) push('<sp>');
          for (const [phone, level] of phones) {
            if (!isPhone(phone)) throw Error(`Niet-ondersteunde Nederlandse klank: ${phone}`);
            push(phone, level);
          }
        }
      }
    }
    if (!found.some(isPhone)) return null;
    push('<eos>');
    return {tokens: found.map(s => ids.get(s)), stress, symbols: found};
  };
}

export async function loadTrace(options = {locateFile: name => new URL(`./assets/${name}`, import.meta.url).href}) {
  const { default: create } = await import('./assets/espeak.mjs');
  const espeak = await create(options);
  if (espeak.ccall('dutch_init', 'number', [], []) !== 0) throw Error('Kan eSpeak NG niet starten.');
  return text => {
    if (espeak.ccall('dutch_trace', 'number', ['string'], [text]) !== 0) throw Error('eSpeak NG kan de tekst niet lezen.');
    return espeak.FS.readFile('/phones.txt', { encoding: 'utf8' });
  };
}
