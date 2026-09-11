/* AgriSense AI - Prediction Form Handler */

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("crop-form");
  if (!form) return;

  // Sync range slider badge values
  const sliders = form.querySelectorAll("input[type='range']");
  sliders.forEach(slider => {
    const badge = document.getElementById(`${slider.id}-val`);
    if (badge) {
      slider.addEventListener("input", (e) => {
        badge.textContent = e.target.value;
      });
    }
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    await submitRecommendation();
  });
});

async function submitRecommendation() {
  const btn = document.getElementById("submit-btn");
  const resultCard = document.getElementById("result-card");
  const errorBox = document.getElementById("error-box");

  btn.disabled = true;
  btn.innerHTML = "🌱 Analyzing Soil & Climate...";

  errorBox.style.display = "none";
  resultCard.style.display = "none";

  const payload = {
    nitrogen: parseFloat(document.getElementById("nitrogen").value),
    phosphorus: parseFloat(document.getElementById("phosphorus").value),
    potassium: parseFloat(document.getElementById("potassium").value),
    temperature: parseFloat(document.getElementById("temperature").value),
    humidity: parseFloat(document.getElementById("humidity").value),
    ph: parseFloat(document.getElementById("ph").value),
    rainfall: parseFloat(document.getElementById("rainfall").value),
    soil_type: document.getElementById("soil_type").value,
    season: document.getElementById("season").value,
    region: document.getElementById("region").value,
    soil_moisture: parseFloat(document.getElementById("soil_moisture").value),
    irrigation: document.getElementById("irrigation").value,
    sunlight: parseFloat(document.getElementById("sunlight").value)
  };

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail?.errors?.join("<br>") || "Validation or prediction error occurred.");
    }

    renderPredictionResult(data);
  } catch (err) {
    errorBox.innerHTML = `⚠️ ${err.message}`;
    errorBox.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.innerHTML = "🔍 Analyze & Recommend Crop";
  }
}

function renderPredictionResult(data) {
  const resultCard = document.getElementById("result-card");
  
  document.getElementById("recommended-crop-name").textContent = data.recommended_crop;
  document.getElementById("confidence-badge").textContent = `${data.confidence}% Match Confidence`;

  // Top 3 Recommendations
  const topList = document.getElementById("top-crops-list");
  topList.innerHTML = "";

  data.top_3_recommendations.forEach((item, index) => {
    const div = document.createElement("div");
    div.style.marginBottom = "1rem";
    div.innerHTML = `
      <div style="display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 0.2rem;">
        <span>${index + 1}. ${item.crop}</span>
        <span>${item.confidence_percentage}%</span>
      </div>
      <div class="progress-bar-container">
        <div class="progress-bar-fill" style="width: ${item.confidence_percentage}%"></div>
      </div>
    `;
    topList.appendChild(div);
  });

  // Factors Rationale
  const factorsList = document.getElementById("factors-list");
  factorsList.innerHTML = "";
  data.explanation.contributing_factors.forEach(factor => {
    const li = document.createElement("li");
    li.style.marginBottom = "0.4rem";
    li.textContent = factor;
    factorsList.appendChild(li);
  });

  // Warnings
  const warningsBox = document.getElementById("warnings-box");
  if (data.explanation.warnings && data.explanation.warnings.length > 0) {
    warningsBox.style.display = "block";
    warningsBox.innerHTML = "<strong>Agronomic Warnings:</strong><br>" + data.explanation.warnings.join("<br>");
  } else {
    warningsBox.style.display = "none";
  }

  resultCard.style.display = "block";
  resultCard.scrollIntoView({ behavior: "smooth" });
}
