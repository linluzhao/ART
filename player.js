// Frame k: alpha = k / 100
const last = 100;
const images = document.querySelectorAll("img[data-frames]");
const slider = document.getElementById("slider");
const button = document.getElementById("play");
const frames = new Map();
let timer = null;

// Milliseconds per frame
function duration(k) {
  return k === 0 || k === last ? 1000 : 100;
}

function show(k) {
  slider.value = k;
  images.forEach(img => { img.src = "data:image/webp;base64," + frames.get(img)[k]; });
}

function next() {
  const k = Number(slider.value) === last ? 0 : Number(slider.value) + 1;
  show(k);
  timer = setTimeout(next, duration(k));
}

function play() {
  button.textContent = "Pause";
  timer = setTimeout(next, duration(Number(slider.value)));
}

function pause() {
  button.textContent = "Play";
  clearTimeout(timer);
  timer = null;
}

button.addEventListener("click", () => { if (timer) pause(); else play(); });
slider.addEventListener("input", () => { pause(); show(Number(slider.value)); });

// Load all frames
Promise.all([...images].map(img =>
  fetch(img.dataset.frames).then(response => response.json()).then(f => frames.set(img, f))
)).then(() => {
  button.disabled = false;
  slider.disabled = false;
  play();
});
