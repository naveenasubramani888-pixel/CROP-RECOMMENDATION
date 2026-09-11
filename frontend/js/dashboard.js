/* AgriSense AI - Analytics Dashboard Script */

document.addEventListener("DOMContentLoaded", async () => {
  if (document.getElementById("stat-total-predictions")) {
    await loadDashboardAnalytics();
  }
  if (document.getElementById("history-table-body")) {
    await loadPredictionHistoryTable();
  }
});

async function loadDashboardAnalytics() {
  try {
    const response = await fetch("/statistics");
    const data = await response.json();

    document.getElementById("stat-total-predictions").textContent = data.total_predictions || 0;
    document.getElementById("stat-most-common-crop").textContent = data.most_recommended_crop || "None";
    document.getElementById("stat-avg-confidence").textContent = data.average_confidence ? `${data.average_confidence}%` : "0%";

    if (data.crop_distribution && Object.keys(data.crop_distribution).length > 0) {
      renderCropDistributionChart(data.crop_distribution);
    }
  } catch (err) {
    console.error("Error loading dashboard statistics:", err);
  }
}

function renderCropDistributionChart(cropCounts) {
  const labels = Object.keys(cropCounts);
  const values = Object.values(cropCounts);

  const data = [{
    labels: labels,
    values: values,
    type: 'pie',
    hole: .4,
    marker: {
      colors: ['#10b981', '#059669', '#34d399', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4']
    }
  }];

  const layout = {
    title: { text: 'Distribution of Recommended Crops', font: { color: '#f8fafc', size: 16 } },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#94a3b8' },
    margin: { t: 40, b: 20, l: 20, r: 20 },
    showlegend: true
  };

  Plotly.newPlot('crop-distribution-chart', data, layout, { responsive: true });
}

async function loadPredictionHistoryTable() {
  try {
    const response = await fetch("/history");
    const data = await response.json();
    const tbody = document.getElementById("history-table-body");
    tbody.innerHTML = "";

    if (!data.history || data.history.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 2rem;">No prediction history recorded yet.</td></tr>`;
      return;
    }

    data.history.forEach(item => {
      const tr = document.createElement("tr");
      const dt = new Date(item.timestamp).toLocaleString();
      const params = item.input_parameters;

      tr.innerHTML = `
        <td>#${item.id}</td>
        <td>${dt}</td>
        <td><strong style="color: #34d399">${item.recommended_crop}</strong></td>
        <td><span class="badge-val">${item.confidence}%</span></td>
        <td>N:${params.nitrogen}, P:${params.phosphorus}, K:${params.potassium} | Temp:${params.temperature}°C, Rain:${params.rainfall}mm</td>
        <td>${params.soil_type} (${params.season})</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading prediction history:", err);
  }
}
