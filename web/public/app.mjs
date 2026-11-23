const $ = id => document.getElementById(id);
const worker = new Worker(new URL('./worker.mjs', import.meta.url), {type: 'module'});
let context, player, sampleRate, chunks = [], url, request = 0, ready = false, busy = false, pending = null;
const spinner = ['/', '|', '-', '\\'];
let frame = 0;
const spinnerTimer = setInterval(() => { $('spinner').textContent = spinner[++frame % spinner.length]; }, 120);
function hideLoading() {
  clearInterval(spinnerTimer);
  $('loading').hidden = true;
}
function showError(message) {
  hideLoading();
  const error = $('error');
  error.textContent = message;
  error.hidden = false;
}
function controls() {
  $('play').disabled = !ready || busy;
  $('stop').disabled = !pending && !busy && !player?.playing;
  $('save').disabled = busy || !chunks.length;
}
function stop() {
  pending = null;
  player?.stop();
  controls();
}
function startPending() {
  if (!ready || busy || !pending) return;
  const next = pending;
  pending = null;
  chunks = [];
  $('phones').textContent = '';
  busy = true;
  player.begin();
  worker.postMessage({type: 'speak', id: ++request, ...next});
}
function play() {
  const text = $('text').value.trim();
  if (!text || (!ready && $('loading').hidden) || !$('speed').reportValidity()) return;
  $('error').hidden = true;
  context ??= new AudioContext();
  player ??= livePlayer(context, controls);
  pending = {text, speed: Number($('speed').value)};
  if (busy) player.stop();
  // Unlock playback during the click, even if the model is still loading.
  if (!ready && context.state !== 'running') context.resume().catch(() => {});
  startPending();
  controls();
}
$('stop').onclick = stop;
$('play').onclick = play;
for (const link of document.querySelectorAll('[data-example]')) {
  link.addEventListener('click', event => {
    event.preventDefault();
    $('text').value = link.dataset.example;
    play();
  });
}
worker.onmessage = ({data}) => {
  if (data.id !== undefined && data.id !== request) return;
  switch (data.type) {
    case 'ready':
      ready = true;
      sampleRate = data.sampleRate;
      hideLoading();
      break;
    case 'sentence':
      if (pending) break;
      chunks.push(data.samples);
      player.add(data.samples, sampleRate);
      $('phones').textContent += `${data.symbols.join(' ')}\n`;
      break;
    case 'done':
      busy = false;
      player.end();
      break;
    case 'error':
      busy = false;
      chunks = [];
      player?.stop();
      if (!ready) pending = null;
      if (!pending) showError(data.message);
      break;
  }
  startPending();
  controls();
};
worker.onerror = event => {
  busy = false; ready = false; pending = null;
  player?.stop();
  showError(event.message || 'Kan het spraakmodel niet laden.');
  controls();
};

$('save').onclick = () => {
  const bytes = 2 * chunks.reduce((total, samples) => total + samples.length, 0);
  const wav = new DataView(new ArrayBuffer(44 + bytes));
  function tag(offset, text) { [...text].forEach((c, n) => wav.setUint8(offset + n, c.charCodeAt(0))); }
  tag(0, 'RIFF'); wav.setUint32(4, 36 + bytes, true); tag(8, 'WAVE');
  tag(12, 'fmt '); wav.setUint32(16, 16, true); wav.setUint16(20, 1, true); wav.setUint16(22, 1, true);
  wav.setUint32(24, sampleRate, true); wav.setUint32(28, 2 * sampleRate, true);
  wav.setUint16(32, 2, true); wav.setUint16(34, 16, true); tag(36, 'data'); wav.setUint32(40, bytes, true);
  let offset = 44;
  for (const samples of chunks) for (const x of samples) {
    wav.setInt16(offset, Math.round(Math.max(-1, Math.min(1, x)) * 32767), true);
    offset += 2;
  }
  if (url) URL.revokeObjectURL(url);
  url = URL.createObjectURL(new Blob([wav.buffer], {type: 'audio/wav'}));
  const link = document.createElement('a');
  link.href = url; link.download = 'tiny-voca-nl.wav'; link.click();
};
worker.postMessage({type: 'load'});
