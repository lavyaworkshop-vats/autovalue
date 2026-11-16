/**
 * AutoValue AI - Frontend Application Logic
 * Integrates with FastAPI endpoints (/api/predict, /api/simulate, /api/metadata, /api/model-info)
 */

let mileageChart = null;
let conditionChart = null;

// Preset vehicle templates
const PRESETS = {
  rav4: {
    year: 2021,
    odometer: 32000,
    manufacturer: 'toyota',
    condition: 'excellent',
    cylinders: '4 cylinders',
    fuel: 'hybrid',
    title_status: 'clean',
    transmission: 'automatic',
    drive: '4wd',
    type: 'SUV',
    state: 'ca',
    asking_price: 27500
  },
  f150: {
    year: 2019,
    odometer: 64000,
    manufacturer: 'ford',
    condition: 'good',
    cylinders: '8 cylinders',
    fuel: 'gas',
    title_status: 'clean',
    transmission: 'automatic',
    drive: '4wd',
    type: 'pickup',
    state: 'tx',
    asking_price: 31000
  },
  m3: {
    year: 2018,
    odometer: 45000,
    manufacturer: 'bmw',
    condition: 'like new',
    cylinders: '6 cylinders',
    fuel: 'gas',
    title_status: 'clean',
    transmission: 'automatic',
    drive: 'rwd',
    type: 'sedan',
    state: 'ny',
    asking_price: 52000
  }
};

document.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
  loadMetadata();
  loadModelInfo();
  // Trigger initial valuation with default form values
  submitValuation();
});

function initEventListeners() {
  // Year slider sync
  const yearSlider = document.getElementById('yearInput');
  const yearDisplay = document.getElementById('yearDisplay');
  yearSlider.addEventListener('input', (e) => {
    yearDisplay.textContent = e.target.value;
  });

  // Tolerance margin slider sync
  const marginSlider = document.getElementById('marginInput');
  const marginDisplay = document.getElementById('marginDisplay');
  marginSlider.addEventListener('input', (e) => {
    marginDisplay.textContent = `±${Math.round(e.target.value * 100)}%`;
  });

  // Valuation Form Submit
  const form = document.getElementById('valuationForm');
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    submitValuation();
  });

  // Odometer Quick Pills
  document.querySelectorAll('.odo-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById('odometerInput').value = btn.dataset.miles;
      submitValuation();
    });
  });

  // Condition Pills
  document.querySelectorAll('.condition-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.condition-pill').forEach(b => {
        b.classList.remove('bg-indigo-600', 'text-white', 'border-indigo-400');
        b.classList.add('bg-slate-800', 'text-slate-300', 'border-slate-700');
      });
      btn.classList.remove('bg-slate-800', 'text-slate-300', 'border-slate-700');
      btn.classList.add('bg-indigo-600', 'text-white', 'border-indigo-400');
      document.getElementById('selectedCondition').value = btn.dataset.condition;
    });
  });

  // Preset Buttons
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const presetKey = btn.dataset.preset;
      applyPreset(PRESETS[presetKey]);
    });
  });

  // Navigation Tabs
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.target;
      document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
      document.querySelectorAll('.nav-tab').forEach(t => {
        t.classList.remove('text-indigo-400', 'border-b-2', 'border-indigo-500', 'font-semibold');
        t.classList.add('text-slate-400');
      });
      document.getElementById(target).classList.remove('hidden');
      tab.classList.add('text-indigo-400', 'border-b-2', 'border-indigo-500', 'font-semibold');
      tab.classList.remove('text-slate-400');
    });
  });
}

function applyPreset(data) {
  document.getElementById('yearInput').value = data.year;
  document.getElementById('yearDisplay').textContent = data.year;
  document.getElementById('odometerInput').value = data.odometer;
  document.getElementById('manufacturerSelect').value = data.manufacturer;
  document.getElementById('bodyTypeSelect').value = data.type;
  document.getElementById('cylindersSelect').value = data.cylinders;
  document.getElementById('fuelSelect').value = data.fuel;
  document.getElementById('titleSelect').value = data.title_status;
  document.getElementById('transmissionSelect').value = data.transmission;
  document.getElementById('driveSelect').value = data.drive;
  document.getElementById('stateSelect').value = data.state;
  document.getElementById('askingPriceInput').value = data.asking_price || '';

  // Select condition pill
  document.getElementById('selectedCondition').value = data.condition;
  document.querySelectorAll('.condition-pill').forEach(b => {
    if (b.dataset.condition === data.condition) {
      b.classList.remove('bg-slate-800', 'text-slate-300', 'border-slate-700');
      b.classList.add('bg-indigo-600', 'text-white', 'border-indigo-400');
    } else {
      b.classList.remove('bg-indigo-600', 'text-white', 'border-indigo-400');
      b.classList.add('bg-slate-800', 'text-slate-300', 'border-slate-700');
    }
  });

  submitValuation();
}

function getFormData() {
  const askingVal = document.getElementById('askingPriceInput').value;
  return {
    year: parseInt(document.getElementById('yearInput').value),
    odometer: parseFloat(document.getElementById('odometerInput').value),
    manufacturer: document.getElementById('manufacturerSelect').value,
    condition: document.getElementById('selectedCondition').value,
    cylinders: document.getElementById('cylindersSelect').value,
    fuel: document.getElementById('fuelSelect').value,
    title_status: document.getElementById('titleSelect').value,
    transmission: document.getElementById('transmissionSelect').value,
    drive: document.getElementById('driveSelect').value,
    type: document.getElementById('bodyTypeSelect').value,
    state: document.getElementById('stateSelect').value,
    lat: 37.77,
    long: -122.42,
    asking_price: askingVal ? parseFloat(askingVal) : null,
    margin_pct: parseFloat(document.getElementById('marginInput').value)
  };
}

async function loadMetadata() {
  try {
    const res = await fetch('/api/metadata');
    if (!res.ok) return;
    const data = await res.json();
    populateSelect('manufacturerSelect', data.manufacturers);
    populateSelect('bodyTypeSelect', data.body_types);
    populateSelect('cylindersSelect', data.cylinders);
    populateSelect('fuelSelect', data.fuel_types);
    populateSelect('titleSelect', data.title_statuses);
    populateSelect('transmissionSelect', data.transmissions);
    populateSelect('driveSelect', data.drives);
    populateSelect('stateSelect', data.states, true);
  } catch (err) {
    console.warn("Using default static metadata:", err);
  }
}

function populateSelect(elemId, items, uppercase = false) {
  const sel = document.getElementById(elemId);
  if (!sel) return;
  const currentVal = sel.value;
  sel.innerHTML = '';
  items.forEach(item => {
    const opt = document.createElement('option');
    opt.value = item;
    opt.textContent = uppercase ? item.toUpperCase() : item.charAt(0).toUpperCase() + item.slice(1);
    if (item === currentVal) opt.selected = true;
    sel.appendChild(opt);
  });
}

async function submitValuation() {
  const payload = getFormData();
  const submitBtn = document.getElementById('submitBtn');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnText = document.getElementById('btnText');

  submitBtn.disabled = true;
  btnSpinner.classList.remove('hidden');
  btnText.textContent = 'Calculating Fair Value...';

  try {
    // 1. Fetch prediction & deal classification
    const predRes = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!predRes.ok) throw new Error(`HTTP Error: ${predRes.status}`);
    const predData = await predRes.json();
    renderValuationResult(predData);

    // 2. Fetch simulation data for charts
    const simRes = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_vehicle: payload })
    });
    if (simRes.ok) {
      const simData = await simRes.json();
      renderSimulationCharts(simData, payload.odometer, payload.condition);
    }
  } catch (err) {
    console.error("Valuation calculation failed:", err);
    alert("Valuation failed: " + err.message);
  } finally {
    submitBtn.disabled = false;
    btnSpinner.classList.add('hidden');
    btnText.textContent = 'Calculate Fair Market Value';
  }
}

function renderValuationResult(data) {
  // Fair Value Hero
  document.getElementById('fairValueDisplay').textContent = `$${data.estimated_fair_value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  document.getElementById('rangeMinDisplay').textContent = `$${data.fair_value_range[0].toLocaleString('en-US')}`;
  document.getElementById('rangeMaxDisplay').textContent = `$${data.fair_value_range[1].toLocaleString('en-US')}`;

  // Deal Classification Card
  const dealCard = document.getElementById('dealEvaluationCard');
  const dealBadge = document.getElementById('dealBadge');
  const dealTitle = document.getElementById('dealTitle');
  const dealDesc = document.getElementById('dealDescription');
  const dealDiffContainer = document.getElementById('dealDiffContainer');
  const dealDiff = document.getElementById('dealDiff');

  if (data.asking_price) {
    dealCard.classList.remove('hidden');
    dealTitle.textContent = data.deal_classification;
    dealDesc.textContent = data.deal_summary;

    dealCard.classList.remove('border-emerald-500/40', 'border-rose-500/40', 'border-sky-500/40', 'glow-emerald', 'glow-rose', 'glow-sky');
    dealBadge.className = 'px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider ';

    if (data.deal_badge_color === 'emerald') {
      dealCard.classList.add('border-emerald-500/40', 'glow-emerald');
      dealBadge.classList.add('bg-emerald-500/20', 'text-emerald-400', 'border', 'border-emerald-500/40');
      dealBadge.textContent = 'Undervalued Deal';
      dealDiffContainer.classList.remove('hidden');
      dealDiff.className = 'font-mono text-emerald-400 font-bold';
      dealDiff.textContent = `-$${Math.abs(data.difference_from_fair_value).toLocaleString('en-US', { minimumFractionDigits: 2 })} below fair value`;
    } else if (data.deal_badge_color === 'rose') {
      dealCard.classList.add('border-rose-500/40', 'glow-rose');
      dealBadge.classList.add('bg-rose-500/20', 'text-rose-400', 'border', 'border-rose-500/40');
      dealBadge.textContent = 'Overpriced Listing';
      dealDiffContainer.classList.remove('hidden');
      dealDiff.className = 'font-mono text-rose-400 font-bold';
      dealDiff.textContent = `+$${Math.abs(data.difference_from_fair_value).toLocaleString('en-US', { minimumFractionDigits: 2 })} premium above fair value`;
    } else {
      dealCard.classList.add('border-sky-500/40', 'glow-sky');
      dealBadge.classList.add('bg-sky-500/20', 'text-sky-400', 'border', 'border-sky-500/40');
      dealBadge.textContent = 'Fair Market Price';
      dealDiffContainer.classList.remove('hidden');
      dealDiff.className = 'font-mono text-sky-400 font-bold';
      dealDiff.textContent = `Within normal market spread ($${data.difference_from_fair_value >= 0 ? '+' : ''}${data.difference_from_fair_value.toLocaleString('en-US')})`;
    }
  } else {
    dealCard.classList.add('hidden');
  }

  // Key Value Drivers
  const driversContainer = document.getElementById('keyDriversContainer');
  driversContainer.innerHTML = '';
  data.key_drivers.forEach(d => {
    const isPos = d.impact === 'Positive';
    const tag = document.createElement('div');
    tag.className = `p-3 rounded-xl border ${isPos ? 'bg-emerald-950/20 border-emerald-800/40 text-emerald-300' : 'bg-rose-950/20 border-rose-800/40 text-rose-300'}`;
    tag.innerHTML = `
      <div class="flex items-center justify-between text-xs font-semibold mb-1">
        <span>${isPos ? '▲' : '▼'} ${d.factor}</span>
        <span class="font-mono text-xs opacity-75">${isPos ? '+' : ''}${d.coefficient}</span>
      </div>
      <p class="text-xs text-slate-400 leading-relaxed">${d.description}</p>
    `;
    driversContainer.appendChild(tag);
  });
}

function renderSimulationCharts(simData, currentOdo, currentCond) {
  // Chart 1: Mileage Depreciation Curve
  const mileageCtx = document.getElementById('mileageChart').getContext('2d');
  const labels = simData.mileage_curve.map(p => `${p.odometer / 1000}k mi`);
  const values = simData.mileage_curve.map(p => p.predicted_price);

  if (mileageChart) mileageChart.destroy();

  mileageChart = new Chart(mileageCtx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Depreciation Curve ($)',
        data: values,
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        fill: true,
        tension: 0.35,
        borderWidth: 3,
        pointBackgroundColor: '#818cf8',
        pointRadius: 5,
        pointHoverRadius: 8
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f172a',
          titleColor: '#f1f5f9',
          bodyColor: '#cbd5e1',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          callbacks: {
            label: (ctx) => `Est. Price: $${ctx.parsed.y.toLocaleString()}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8' }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: {
            color: '#94a3b8',
            callback: (val) => `$${val / 1000}k`
          }
        }
      }
    }
  });

  // Chart 2: Condition Impact Bar Chart
  const condCtx = document.getElementById('conditionChart').getContext('2d');
  const condLabels = simData.condition_curve.map(p => p.condition.charAt(0).toUpperCase() + p.condition.slice(1));
  const condValues = simData.condition_curve.map(p => p.predicted_price);
  const condColors = simData.condition_curve.map(p => {
    return p.condition.toLowerCase() === currentCond.toLowerCase() ? '#38bdf8' : '#334155';
  });

  if (conditionChart) conditionChart.destroy();

  conditionChart = new Chart(condCtx, {
    type: 'bar',
    data: {
      labels: condLabels,
      datasets: [{
        label: 'Valuation by Condition ($)',
        data: condValues,
        backgroundColor: condColors,
        borderRadius: 8,
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f172a',
          titleColor: '#f1f5f9',
          bodyColor: '#cbd5e1',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          callbacks: {
            label: (ctx) => `Est. Price: $${ctx.parsed.y.toLocaleString()}`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#94a3b8' }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: {
            color: '#94a3b8',
            callback: (val) => `$${val / 1000}k`
          }
        }
      }
    }
  });
}

async function loadModelInfo() {
  try {
    const res = await fetch('/api/model-info');
    if (!res.ok) return;
    const info = await res.json();
    document.getElementById('metricR2').textContent = info.metrics.R2_Score.toFixed(4);
    document.getElementById('metricMAE').textContent = `$${info.metrics.MAE_USD.toFixed(2)}`;
    document.getElementById('metricRMSE').textContent = `$${info.metrics.RMSE_USD.toFixed(2)}`;
    document.getElementById('metricFeatures').textContent = info.features_count;
  } catch (err) {
    console.warn("Model info fetch failed:", err);
  }
}
