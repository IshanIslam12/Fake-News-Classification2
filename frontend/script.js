// ============================
// 1️⃣  BACKEND API URL
// ============================
// Change this before deploying:
// Local test: "http://127.0.0.1:8000"
// Render: "https://your-render-service.onrender.com"
const API_BASE = "http://127.0.0.1:8000";
const API_URL = `${API_BASE}/predict`;


document.addEventListener("DOMContentLoaded", () => {
  const titleInput = document.getElementById("title");
  const textInput = document.getElementById("text");
  const predictBtn = document.getElementById("predictBtn");
  const resultBox = document.getElementById("result");

  if (!titleInput || !textInput || !predictBtn || !resultBox) {
    console.error("❌ Error: Missing HTML elements.");
    return;
  }

  async function predictFakeNews() {
    const title = titleInput.value.trim();
    const text = textInput.value.trim();

    // ============================
    // 2️⃣  REQUIRE BOTH title + text
    // ============================
    if (!title || !text) {
      resultBox.style.display = "block";
      resultBox.className = "result-box";

      resultBox.innerHTML = `
        <span class="result-title">Missing Input</span>
        <span class="result-detail">Please enter BOTH a title and article text.</span>
      `;
      return;
    }

    // ============================
    // 3️⃣  UI: show loading state
    // ============================
    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting...";
    resultBox.style.display = "block";
    resultBox.className = "result-box";
    resultBox.innerHTML = `
      <span class="result-title">Working...</span>
      <span class="result-detail">Analyzing with the BERT model.</span>
    `;

    try {
      // ============================
      // 4️⃣  Send request to backend
      // ============================
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, text }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      console.log("Prediction response:", data);

      if (!data.ok) {
        throw new Error(data.error || "Prediction failed.");
      }

      // ============================
      // 5️⃣  Extract backend response
      // ============================
      const label = data.label;                   // REAL / FAKE
      const resultText = data.result_text;        // "82% REAL"
      const pctReal = data.percentage_real;       // number
      const pctFake = data.percentage_fake;       // number
      const confidence = data.confidence;         // 0–1

      // ============================
      // 6️⃣  Decide final color
      // ============================
      let boxClass = "result-box";
      if (label === "REAL") boxClass += " real";
      if (label === "FAKE") boxClass += " fake";

      resultBox.className = boxClass;

      // ============================
      // 7️⃣  Build the result output
      // ============================
      resultBox.innerHTML = `
        <span class="result-title">${resultText}</span>
        <span class="result-detail">
          Real: ${pctReal}% • Fake: ${pctFake}%
        </span>
      `;
    } catch (err) {
      console.error("Prediction error:", err);

      resultBox.className = "result-box";
      resultBox.innerHTML = `
        <span class="result-title">Error</span>
        <span class="result-detail">
          ${err.message}. Make sure the backend is running at ${API_BASE}.
        </span>
      `;
    } finally {
      // ============================
      // 8️⃣  Reset button
      // ============================
      predictBtn.disabled = false;
      predictBtn.textContent = "Predict";
    }
  }

  // ============================
  // 9️⃣  Add button listener
  // ============================
  predictBtn.addEventListener("click", predictFakeNews);
});
