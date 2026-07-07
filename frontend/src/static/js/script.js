const analyzeBtn = document.getElementById("analyzeBtn");

const PREDICT_URL = `${CONFIG.API_BASE_URL}/predict`;
const HISTORY_URL = `${CONFIG.API_BASE_URL}/history`;

let lrPieChart = null;
let bertPieChart = null;
let lrGaugeChart = null;
let bertGaugeChart = null;

document.addEventListener("DOMContentLoaded", function () {
  loadHistory();
});

analyzeBtn.addEventListener("click", async function () {
  const articleText = document.getElementById("articleInput").value.trim();

  hideWarning();

  if (articleText === "") {
    showWarning("Please paste a medical news article first.");
    return;
  }

  analyzeBtn.textContent = "Analyzing...";
  analyzeBtn.disabled = true;

  try {
    const response = await fetch(PREDICT_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        text: articleText
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      showWarning(errorData.detail || "Backend returned an error.");
      return;
    }

    const data = await response.json();

    displayResults(data);
    updateDashboardCharts(data);
    loadHistory();

    document.getElementById("dashboardSection").scrollIntoView({
      behavior: "smooth"
    });

  } catch (error) {
    console.error(error);
    showWarning("Failed to connect to backend. Make sure FastAPI is running.");
  } finally {
    analyzeBtn.textContent = "Analyze Article";
    analyzeBtn.disabled = false;
  }
});

function displayResults(data) {
  const lrPrediction = document.getElementById("lrPrediction");
  lrPrediction.textContent = data.lr_results.prediction;
  lrPrediction.className =
    data.lr_results.prediction === "Fake" ? "fake-text" : "real-text";

  document.getElementById("lrReal").textContent =
    toPercent(data.lr_results.probabilities.real);

  document.getElementById("lrFake").textContent =
    toPercent(data.lr_results.probabilities.fake);

  const bertPrediction = document.getElementById("bertPrediction");
  bertPrediction.textContent = data.bert_results.prediction;
  bertPrediction.className =
    data.bert_results.prediction === "Fake" ? "fake-text" : "real-text";

  document.getElementById("bertReal").textContent =
    toPercent(data.bert_results.probabilities.real);

  document.getElementById("bertFake").textContent =
    toPercent(data.bert_results.probabilities.fake);

  const rawHtml = marked.parse(data.ollama_summary);
  const cleanHtml = DOMPurify.sanitize(rawHtml);
  document.getElementById("ollamaSummary").innerHTML = cleanHtml;

  showWords("fakeWords", data.lr_results.explanations.push_fake);
  showWords("realWords", data.lr_results.explanations.push_real);
}

function updateDashboardCharts(data) {
  const lrPrediction = data.lr_results.prediction;
  const bertPrediction = data.bert_results.prediction;

  const lrConfidence =
    lrPrediction === "Fake"
      ? data.lr_results.probabilities.fake
      : data.lr_results.probabilities.real;

  const bertConfidence =
    bertPrediction === "Fake"
      ? data.bert_results.probabilities.fake
      : data.bert_results.probabilities.real;

  drawGaugeChart("lrGaugeChart", "lrConfidenceText", lrConfidence, "lr");
  drawGaugeChart("bertGaugeChart", "bertConfidenceText", bertConfidence, "bert");

  drawPieChart(
    "lrPieChart",
    data.lr_results.probabilities.fake,
    data.lr_results.probabilities.real,
    "lr"
  );

  drawPieChart(
    "bertPieChart",
    data.bert_results.probabilities.fake,
    data.bert_results.probabilities.real,
    "bert"
  );
}

function drawGaugeChart(canvasId, textId, confidence, type) {
  const ctx = document.getElementById(canvasId);
  const percentage = (confidence * 100).toFixed(1);

  document.getElementById(textId).textContent = `${percentage}%`;

  if (type === "lr" && lrGaugeChart) {
    lrGaugeChart.destroy();
  }

  if (type === "bert" && bertGaugeChart) {
    bertGaugeChart.destroy();
  }

  const gaugeNeedle = {
    id: "gaugeNeedle",
    afterDatasetDraw(chart) {
      const { ctx } = chart;
      const meta = chart.getDatasetMeta(0).data[0];

      const centerX = meta.x;
      const centerY = meta.y;
      const radius = meta.outerRadius * 0.82;

      const angle = Math.PI + (confidence * Math.PI);

      const needleX = centerX + Math.cos(angle) * radius;
      const needleY = centerY + Math.sin(angle) * radius;

      ctx.save();

      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(needleX, needleY);
      ctx.lineWidth = 6;
      ctx.strokeStyle = "#0f2742";
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(centerX, centerY, 9, 0, Math.PI * 2);
      ctx.fillStyle = "#0f2742";
      ctx.fill();

      ctx.restore();
    }
  };

  const chart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Low", "", "", "", "High"],
      datasets: [{
        data: [20, 20, 20, 20, 20],
        backgroundColor: [
          "#22c55e",
          "#84cc16",
          "#eab308",
          "#f97316",
          "#ef4444"
        ],
        borderColor: "#ffffff",
        borderWidth: 4,
        circumference: 180,
        rotation: 270,
        cutout: "65%"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          enabled: false
        }
      }
    },
    plugins: [gaugeNeedle]
  });

  if (type === "lr") {
    lrGaugeChart = chart;
  } else {
    bertGaugeChart = chart;
  }
}

function drawPieChart(canvasId, fakeProb, realProb, type) {
  const ctx = document.getElementById(canvasId);

  if (type === "lr" && lrPieChart) {
    lrPieChart.destroy();
  }

  if (type === "bert" && bertPieChart) {
    bertPieChart.destroy();
  }

  const chart = new Chart(ctx, {
    type: "pie",
    data: {
      labels: ["Real", "Fake"],
      datasets: [{
        data: [
          (realProb * 100).toFixed(1),
          (fakeProb * 100).toFixed(1)
        ],
        backgroundColor: [
          "#16a34a",
          "#dc2626"
        ],
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      rotation: Math.PI / 2,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            usePointStyle: true,
            pointStyle: "rect",
            padding: 16
          }
        },
        tooltip: {
          callbacks: {
            label(context) {
              return `${context.label}: ${context.raw}%`;
            }
          }
        }
      }
    }
  });

  if (type === "lr") {
    lrPieChart = chart;
  } else {
    bertPieChart = chart;
  }
}

async function loadHistory() {
  try {
    const response = await fetch(HISTORY_URL);

    if (!response.ok) {
      throw new Error("Failed to fetch history.");
    }

    const historyData = await response.json();
    displayHistory(historyData);

  } catch (error) {
    console.error(error);
  }
}

function displayHistory(historyData) {
  const historyBody = document.getElementById("historyBody");
  historyBody.innerHTML = "";

  if (!historyData || historyData.length === 0) {
    historyBody.innerHTML = `
      <tr>
        <td colspan="7">No history yet.</td>
      </tr>
    `;
    return;
  }

  historyData.forEach(item => {
    const status =
      item.lr_prediction === item.bert_prediction ? "Agree" : "Disagree";

    const row = document.createElement("tr");
    row.className = "history-row";
    row.title = "Click View to open this analysis";

    row.innerHTML = `
      <td>${item.id}</td>
      <td>${item.preview}</td>
      <td>
        <span class="badge ${
          item.lr_prediction === "Fake" ? "badge-fake" : "badge-real"
        }">
          ${item.lr_prediction}
        </span>
      </td>
      <td>
        <span class="badge ${
          item.bert_prediction === "Fake" ? "badge-fake" : "badge-real"
        }">
          ${item.bert_prediction}
        </span>
      </td>
      <td>
        <span class="badge ${
          status === "Agree" ? "badge-agree" : "badge-disagree"
        }">
          ${status}
        </span>
      </td>
      <td>${formatDate(item.created_at)}</td>
      <td>
        <button class="view-btn" type="button">View</button>
      </td>
    `;

    row.querySelector(".view-btn").addEventListener("click", function (event) {
      event.stopPropagation();
      loadHistoryItem(item.id);
    });

    row.addEventListener("click", function () {
      loadHistoryItem(item.id);
    });

    historyBody.appendChild(row);
  });
}

async function loadHistoryItem(id) {
  try {
    hideWarning();

    const response = await fetch(`${HISTORY_URL}/${id}`);

    if (!response.ok) {
      throw new Error("Failed to fetch selected history item.");
    }

    const data = await response.json();

    displayResults(data);
    updateDashboardCharts(data);

    document.getElementById("dashboardSection").scrollIntoView({
      behavior: "smooth"
    });

  } catch (error) {
    console.error(error);
    showWarning("Failed to load selected history record.");
  }
}

function showWords(elementId, words) {
  const box = document.getElementById(elementId);
  box.innerHTML = "";

  if (!words || words.length === 0) {
    box.innerHTML = "<p>No significant words found.</p>";
    return;
  }

  const maxScore = Math.max(...words.map(item => Math.abs(item.score))) || 1;

  words.forEach(item => {
    const container = document.createElement("div");
    container.className = "word-item";

    const label = document.createElement("div");
    label.className = "word-label";

    const word = document.createElement("span");
    word.textContent = item.word;

    const score = document.createElement("span");
    score.textContent = item.score.toFixed(3);

    label.appendChild(word);
    label.appendChild(score);

    const barBg = document.createElement("div");
    barBg.className = "bar-bg";

    const barFill = document.createElement("div");
    barFill.className =
      elementId === "fakeWords"
        ? "bar-fill bar-fake"
        : "bar-fill bar-real";

    const width = (Math.abs(item.score) / maxScore) * 100;
    barFill.style.width = `${width}%`;

    barBg.appendChild(barFill);
    container.appendChild(label);
    container.appendChild(barBg);

    box.appendChild(container);
  });
}

function showWarning(message) {
  const warningBox = document.getElementById("warningMessage");
  warningBox.textContent = message;
  warningBox.style.display = "block";
}

function hideWarning() {
  const warningBox = document.getElementById("warningMessage");
  warningBox.textContent = "";
  warningBox.style.display = "none";
}

function toPercent(value) {
  return (value * 100).toFixed(1) + "%";
}

function formatDate(dateString) {
  return dateString.replace("T", " ").substring(0, 19);
}