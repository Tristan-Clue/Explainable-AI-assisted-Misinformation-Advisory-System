const analyzeBtn = document.getElementById("analyzeBtn");

const PREDICT_URL = `${CONFIG.API_BASE_URL}/predict`;
const HISTORY_URL = `${CONFIG.API_BASE_URL}/history`;

document.addEventListener("DOMContentLoaded", function () {
  loadHistory();
});

analyzeBtn.addEventListener("click", async function () {
  const articleText = document.getElementById("articleInput").value.trim();

  if (articleText === "") {
    alert("Please paste a medical news article first.");
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
      throw new Error("Backend returned an error.");
    }

    const data = await response.json();

    displayResults(data);
    loadHistory();

  } catch (error) {
    console.error(error);
    alert("Failed to connect to backend. Make sure FastAPI is running.");
  } finally {
    analyzeBtn.textContent = "Analyze Article";
    analyzeBtn.disabled = false;
  }
});

function displayResults(data) {
  document.getElementById("lrPrediction").textContent = data.lr_results.prediction;
  document.getElementById("lrFake").textContent = toPercent(data.lr_results.probabilities.fake);
  document.getElementById("lrReal").textContent = toPercent(data.lr_results.probabilities.real);

  document.getElementById("bertPrediction").textContent = data.bert_results.prediction;
  document.getElementById("bertFake").textContent = toPercent(data.bert_results.probabilities.fake);
  document.getElementById("bertReal").textContent = toPercent(data.bert_results.probabilities.real);

  document.getElementById("ollamaSummary").textContent = data.ollama_summary;

  showWords("fakeWords", data.lr_results.explanations.push_fake);
  showWords("realWords", data.lr_results.explanations.push_real);
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
        <td colspan="6">No history yet.</td>
      </tr>
    `;
    return;
  }

  historyData.forEach(item => {
    const status = item.lr_prediction === item.bert_prediction ? "Agree" : "Disagree";

    const row = document.createElement("tr");

    row.innerHTML = `
      <td>${item.id}</td>
      <td>${item.preview}</td>
      <td>${item.lr_prediction}</td>
      <td>${item.bert_prediction}</td>
      <td>${status}</td>
      <td>${formatDate(item.created_at)}</td>
    `;

    historyBody.appendChild(row);
  });
}

function toPercent(value) {
  return (value * 100).toFixed(1) + "%";
}

function showWords(elementId, words) {
  const box = document.getElementById(elementId);
  box.innerHTML = "";

  if (!words || words.length === 0) {
    box.innerHTML = "<p>No significant words found.</p>";
    return;
  }

  words.forEach(item => {
    const p = document.createElement("p");
    p.textContent = `${item.word}: ${item.score.toFixed(3)}`;
    box.appendChild(p);
  });
}

function formatDate(dateString) {
  const date = new Date(dateString);
  return date.toLocaleString();
}