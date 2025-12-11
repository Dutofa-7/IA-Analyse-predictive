const resultBox = document.getElementById("resultBox");

const model = tf.sequential({
  layers: [
    tf.layers.flatten({ inputShape: [28, 28, 1] }),
    tf.layers.dense({ units: 512, activation: "relu" }),
    tf.layers.dense({ units: 128, activation: "relu" }),
    tf.layers.dense({ units: 10, activation: "softmax" }),
  ],
});

model.compile({
  optimizer: "sgd",
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let isDrawing = false;
let lastX = 0;
let lastY = 0;

ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "black";
ctx.lineWidth = 8;
ctx.lineCap = "round";

canvas.addEventListener("mousedown", (e) => {
  isDrawing = true;
  const rect = canvas.getBoundingClientRect();
  lastX = e.clientX - rect.left;
  lastY = e.clientY - rect.top;
});

canvas.addEventListener("mousemove", (e) => {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  lastX = x;
  lastY = y;
});

canvas.addEventListener("mouseup", () => {
  isDrawing = false;
});
canvas.addEventListener("mouseout", () => {
  isDrawing = false;
});

function clearCanvas() {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  resultBox.innerHTML = `
        <div class="result-label">En attente</div>
        <div style="color: #ccc; font-size: 48px;">—</div>
    `;
  document.getElementById("probabilities").style.display = "none";
}

async function predictDigit() {
  resultBox.innerHTML = '<div class="loading">Prédiction...</div>';

  try {
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(canvas, 0, 0, 28, 28);

    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const pixels = imageData.data;
    const pixelArray = new Float32Array(28 * 28);

    for (let i = 0; i < pixels.length; i += 4) {
      pixelArray[i / 4] = pixels[i] / 255.0;
    }

    let tensor = tf.tensor4d(pixelArray, [1, 28, 28, 1]);

    tensor = tf.tidy(() => {
      let t = tf.tensor4d(pixelArray, [1, 28, 28, 1]);
      t = t.sub(0.1307).div(0.3081);
      return t;
    });

    const predictions = model.predict(tensor);
    const probsData = await predictions.data();
    const probs = Array.from(probsData);

    const pred = predictions.argMax(-1).dataSync()[0];
    const confidence = probs[pred];

    displayResults(pred, confidence, probs);

    tf.dispose([tensor, predictions]);
  } catch (error) {
    resultBox.innerHTML = `<div style="color: red; font-size: 12px;">Erreur: ${error.message}</div>`;
  }
}

function displayResults(pred, conf, probs) {
  const confidence = (conf * 100).toFixed(0);

  resultBox.innerHTML = `
        <div class="result-label">Détecté</div>
        <div class="result-digit">${pred}</div>
        <div class="confidence">${confidence}%</div>
    `;

  const probList = document.getElementById("probList");
  probList.innerHTML = "";

  probs.forEach((prob, digit) => {
    const percentage = (prob * 100).toFixed(0);
    probList.innerHTML += `
            <div class="prob-item">
                <span>${digit}</span>
                <div class="prob-bar"><div class="prob-fill" style="width: ${percentage}%"></div></div>
                <span>${percentage}%</span>
            </div>
        `;
  });

  document.getElementById("probabilities").style.display = "block";
}

clearCanvas();
