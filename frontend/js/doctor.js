/**
 * MEDIRAG-XAI — Doctor Portal JavaScript
 * Handles: Symptom prediction, SHAP charts, Drug safety, RAG, PDF upload, NER
 */

// ── Real Kaggle Dataset Symptoms (132 features) ─────────────────────────────
const ALL_SYMPTOMS = [
  "itching","skin_rash","nodal_skin_eruptions","continuous_sneezing","shivering","chills",
  "joint_pain","stomach_pain","acidity","ulcers_on_tongue","muscle_wasting","vomiting",
  "burning_micturition","spotting_urination","fatigue","weight_gain","anxiety","cold_hands_and_feets",
  "mood_swings","weight_loss","restlessness","lethargy","patches_in_throat","irregular_sugar_level",
  "cough","high_fever","sunken_eyes","breathlessness","sweating","dehydration","indigestion",
  "headache","yellowish_skin","dark_urine","nausea","loss_of_appetite","pain_behind_the_eyes",
  "back_pain","constipation","abdominal_pain","diarrhoea","mild_fever","yellow_urine",
  "yellowing_of_eyes","acute_liver_failure","fluid_overload","swelling_of_stomach",
  "swelled_lymph_nodes","malaise","blurred_and_distorted_vision","phlegm","throat_irritation",
  "redness_of_eyes","sinus_pressure","runny_nose","congestion","chest_pain","weakness_in_limbs",
  "fast_heart_rate","pain_during_bowel_movements","pain_in_anal_region","bloody_stool",
  "irritation_in_anus","neck_stiffness","word_difficulty_in_finding","dizziness","cramps",
  "bruising","obesity","swollen_legs","swollen_blood_vessels","puffy_face_and_eyes",
  "enlarged_thyroid","brittle_nails","swollen_extremeties","excessive_hunger","extra_marital_contacts",
  "drying_and_tingling_lips","slurred_speech","knee_pain","hip_joint_pain","muscle_weakness",
  "stiff_neck","swelling_joints","movement_stiffness","spinning_movements","loss_of_balance",
  "unsteadiness","weakness_of_one_body_side","loss_of_smell","bladder_discomfort","foul_smell_of_urine",
  "continuous_feel_of_urine","passage_of_gases","internal_itching","toxic_look_(typhos)",
  "depression","irritability","muscle_pain","altered_sensorium","red_spots_over_body","belly_pain",
  "abnormal_menstruation","dischromic_patches","watering_from_eyes","increased_appetite","polyuria",
  "family_history","mucoid_sputum","rusty_sputum","lack_of_concentration","visual_disturbances",
  "receiving_blood_transfusion","receiving_unsterile_injections","coma","stomach_bleeding",
  "distention_of_abdomen","history_of_alcohol_consumption","fluid_overload.1","blood_in_sputum",
  "prominent_veins_on_calf","palpitations","painful_walking","pus_filled_pimples","blackheads",
  "scurring","skin_peeling","silver_like_dusting","small_dents_in_nails","inflammatory_nails",
  "blister","red_sore_around_nose","yellow_crust_ooze"
];

let shapChartInstance = null;
let selectedSymptoms  = new Set();

// ── Symptom Grid ────────────────────────────────────────────────────────────
function renderSymptomGrid(filter = '') {
  const grid = document.getElementById('symptomGrid');
  if (!grid) return;
  grid.innerHTML = '';
  const filtered = ALL_SYMPTOMS.filter(s =>
    s.replace(/_/g, ' ').toLowerCase().includes(filter.toLowerCase())
  );
  filtered.forEach(symptom => {
    const display = symptom.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    const el = document.createElement('div');
    el.className = `symptom-item ${selectedSymptoms.has(symptom) ? 'selected' : ''}`;
    el.setAttribute('data-symptom', symptom);
    el.innerHTML = `
      <div class="symptom-check">${selectedSymptoms.has(symptom) ? '✓' : ''}</div>
      <span>${display}</span>
    `;
    el.onclick = () => toggleSymptom(symptom, el);
    grid.appendChild(el);
  });
}

function toggleSymptom(symptom, el) {
  if (selectedSymptoms.has(symptom)) {
    selectedSymptoms.delete(symptom);
    el.classList.remove('selected');
    el.querySelector('.symptom-check').textContent = '';
  } else {
    selectedSymptoms.add(symptom);
    el.classList.add('selected');
    el.querySelector('.symptom-check').textContent = '✓';
  }
  document.getElementById('selectedCount').textContent = selectedSymptoms.size;
}

function filterSymptoms(val) { renderSymptomGrid(val); }
function clearSymptoms() {
  selectedSymptoms.clear();
  document.getElementById('selectedCount').textContent = '0';
  renderSymptomGrid(document.getElementById('symptomSearch')?.value || '');
}

// ── Disease Prediction ──────────────────────────────────────────────────────
async function runPrediction() {
  const symptoms = [...selectedSymptoms];
  if (symptoms.length === 0) {
    showToast('Please select at least one symptom.', 'warning');
    return;
  }

  const btn = document.getElementById('predictBtn');
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';
  btn.disabled = true;

  try {
    const data = await api.post('/api/predict', {
      symptoms,
      patient_age:    parseInt(document.getElementById('patientAge')?.value) || null,
      patient_gender: document.getElementById('patientGender')?.value || null,
    });

    renderPredictions(data.predictions);
    renderSHAP(data.shap_explanation);
    document.getElementById('predictionResults').style.display = '';
    document.getElementById('shapPanel').style.display = '';
    showToast(`Prediction complete — Top: ${data.predictions[0]?.disease}`, 'success');

    // Auto-fill RAG with top disease
    if (data.predictions[0]) {
      const disease = data.predictions[0].disease;
      document.getElementById('ragQuery').value = `What is ${disease} and how is it treated?`;
    }
  } catch (e) {
    showToast(`Prediction error: ${e.message}`, 'error');
  } finally {
    btn.innerHTML = '<i class="fas fa-brain me-2"></i>Run AI Prediction';
    btn.disabled = false;
  }
}

function renderPredictions(predictions) {
  const container = document.getElementById('predictionContent');
  if (!container) return;

  const rankBadges = ['rank-1', 'rank-2', 'rank-3'];
  const rankNums   = ['1', '2', '3'];

  container.innerHTML = `
    <div class="row g-3">
      ${predictions.map((p, i) => `
        <div class="col-md-4">
          <div class="p-3" style="background:rgba(15,23,42,0.6);border-radius:12px;border:1px solid var(--border);transition:all 0.3s" 
               onmouseover="this.style.borderColor='var(--primary)'" onmouseout="this.style.borderColor='var(--border)'">
            <div class="d-flex align-items-center gap-2 mb-2">
              <div class="prediction-rank ${rankBadges[i]}">${rankNums[i]}</div>
              <div>
                <div style="font-weight:700;font-size:0.9rem">${p.disease}</div>
                <span class="icd-code">ICD: ${p.icd_code}</span>
              </div>
            </div>
            <div class="d-flex justify-content-between mb-2">
              <span style="font-size:0.78rem;color:var(--text-muted)">Confidence</span>
              <span class="badge-pill ${confidenceClass(p.confidence)}">${p.confidence}%</span>
            </div>
            <div class="progress-glass mb-2">
              <div class="progress-fill" style="width:${p.confidence}%;background:${confidenceColor(p.confidence)}"></div>
            </div>
            <div class="d-flex justify-content-between">
              <span style="font-size:0.78rem;color:var(--text-muted)">Severity</span>
              <span class="badge-pill ${severityClass(p.severity)}">${p.severity}</span>
            </div>
          </div>
        </div>
      `).join('')}
    </div>
    <div class="mt-3 alert-glass alert-info-box">
      <i class="fas fa-info-circle"></i>
      <div style="font-size:0.8rem">
        <strong>Note:</strong> These predictions are AI-generated based on ${[...selectedSymptoms].length} selected symptoms. 
        Always confirm with clinical examination and laboratory tests. Not a substitute for professional medical diagnosis.
      </div>
    </div>
  `;
}

// ── SHAP Visualization ──────────────────────────────────────────────────────
function renderSHAP(shap) {
  if (!shap || !shap.features || shap.features.length === 0) return;

  const ctx = document.getElementById('shapChart');
  if (!ctx) return;

  if (shapChartInstance) shapChartInstance.destroy();

  const colors = shap.values.map(v =>
    v > 0 ? 'rgba(14, 165, 233, 0.7)' : 'rgba(244, 63, 94, 0.7)'
  );

  shapChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: shap.features,
      datasets: [{
        label: 'SHAP Feature Importance',
        data:  shap.values,
        backgroundColor: colors,
        borderColor: colors.map(c => c.replace('0.7', '1')),
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` SHAP: ${ctx.raw.toFixed(4)}`
          },
          backgroundColor: 'rgba(15,23,42,0.95)',
          titleColor: '#F1F5F9',
          bodyColor: '#94A3B8',
        },
      },
      scales: {
        x: {
          grid:   { color: 'rgba(148,163,184,0.08)' },
          ticks:  { color: '#94A3B8', font: { size: 11 } },
        },
        y: {
          grid:  { display: false },
          ticks: { color: '#CBD5E1', font: { size: 11 } },
        },
      },
    },
  });

  document.getElementById('shapSummary').innerHTML = `
    <div class="alert-glass alert-info-box">
      <i class="fas fa-chart-bar"></i>
      <div style="font-size:0.8rem">
        <strong>Top Contributing Feature:</strong> "${shap.features[0]}" had the highest impact on the 
        "${shap.disease}" prediction. Blue bars indicate positive contributions, red bars indicate negative contributions.
      </div>
    </div>
  `;
}

// ── Drug Safety ─────────────────────────────────────────────────────────────
async function checkDrug() {
  const drugName = document.getElementById('drugName')?.value?.trim();
  if (!drugName) { showToast('Please select a drug.', 'warning'); return; }

  const condStr  = document.getElementById('patientConditions')?.value || '';
  const allerStr = document.getElementById('patientAllergies')?.value  || '';
  const preg     = document.getElementById('isPregnant')?.checked || false;

  const conditions = condStr.split(',').map(s => s.trim()).filter(Boolean);
  const allergies  = allerStr.split(',').map(s => s.trim()).filter(Boolean);

  try {
    const data = await api.post('/api/check-drug', {
      drug_name:           drugName,
      patient_conditions:  conditions,
      patient_allergies:   allergies,
      is_pregnant:         preg,
    });
    renderDrugResults(data);
  } catch (e) {
    showToast(`Drug check error: ${e.message}`, 'error');
  }
}

function renderDrugResults(data) {
  const container = document.getElementById('drugResults');
  if (!container) return;

  if (!data.drug_found) {
    container.innerHTML = `
      <div class="alert-glass alert-warning-box">
        <i class="fas fa-question-circle"></i>
        <div><strong>${data.message}</strong></div>
      </div>`;
    return;
  }

  const alertsHtml = data.alerts.length === 0
    ? `<div class="alert-glass alert-success-box"><i class="fas fa-check-circle"></i><div><strong>No safety alerts detected</strong> for current conditions.</div></div>`
    : data.alerts.map(a => {
        const type = a.severity === 'CRITICAL' ? 'alert-critical' :
                     a.severity === 'MODERATE'  ? 'alert-warning-box' : 'alert-info-box';
        const icon = a.severity === 'CRITICAL' ? 'fa-times-circle' :
                     a.severity === 'MODERATE'  ? 'fa-exclamation-triangle' : 'fa-info-circle';
        return `
          <div class="alert-glass ${type}">
            <i class="fas ${icon}"></i>
            <div>
              <div style="font-weight:600;font-size:0.82rem">${a.category}</div>
              <div style="font-size:0.8rem;margin-top:2px">${a.message}</div>
            </div>
          </div>`;
      }).join('');

  container.innerHTML = `
    <div style="background:rgba(15,23,42,0.5);border-radius:10px;padding:0.875rem;margin-bottom:0.75rem">
      <div style="font-weight:700;font-size:0.9rem;margin-bottom:0.25rem">${data.drug_name}</div>
      <div style="font-size:0.78rem;color:var(--text-muted)">${data.drug_class}</div>
      <div class="d-flex gap-2 mt-2 flex-wrap">
        <span class="badge-pill badge-primary">Pregnancy: Category ${data.pregnancy_category}</span>
        <span class="badge-pill ${data.alert_count > 0 ? 'badge-danger' : 'badge-success'}">${data.alert_count} Alert(s)</span>
      </div>
    </div>
    ${alertsHtml}
    ${data.side_effects?.length ? `
      <div style="font-size:0.78rem;color:var(--text-muted);margin-top:0.5rem">
        <strong style="color:var(--text-secondary)">Common Side Effects:</strong> ${data.side_effects.join(', ')}
      </div>` : ''}
    <div class="mt-2" style="font-size:0.72rem;color:var(--warning);font-style:italic">${data.disclaimer}</div>
  `;
}

// ── Clinical RAG Evidence ────────────────────────────────────────────────────
async function runRAG() {
  const query = document.getElementById('ragQuery')?.value?.trim();
  if (!query) { showToast('Please enter a clinical question.', 'warning'); return; }

  const btn = document.getElementById('ragBtn');
  btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';
  btn.disabled  = true;

  try {
    const data = await api.post('/api/chat', { message: query, session_id: 'doctor' });
    renderRAGResults(data);
  } catch (e) {
    showToast(`RAG error: ${e.message}`, 'error');
    document.getElementById('ragResults').innerHTML = `
      <div class="alert-glass alert-critical">
        <i class="fas fa-times-circle"></i>
        <div>RAG engine error: ${e.message}. Ensure backend is running and RAG is initialized.</div>
      </div>`;
  } finally {
    btn.innerHTML = '<i class="fas fa-search"></i>';
    btn.disabled  = false;
  }
}

function askRAG(q) {
  document.getElementById('ragQuery').value = q;
  runRAG();
}

function renderRAGResults(data) {
  const container = document.getElementById('ragResults');
  if (!container) return;

  const sourcesHtml = (data.sources || []).map(s =>
    `<span class="badge-pill badge-primary me-1 mb-1"><i class="fas fa-file-alt me-1"></i>${s}</span>`
  ).join('');

  const chunksHtml = (data.chunks || []).map(c => `
    <div class="source-citation">
      <strong><i class="fas fa-file me-1"></i>${c.source}</strong><br>
      <span style="font-size:0.75rem">${c.text}...</span>
    </div>
  `).join('');

  container.innerHTML = `
    <div style="background:rgba(15,23,42,0.5);border-radius:10px;padding:1rem;margin-bottom:0.75rem">
      <div style="font-size:0.82rem;font-weight:600;color:var(--primary);margin-bottom:0.5rem">
        <i class="fas fa-quote-left me-1"></i> Query: ${data.question}
      </div>
      <div style="font-size:0.85rem;line-height:1.7;color:var(--text-primary)">${renderMarkdown(data.answer)}</div>
    </div>
    ${data.sources?.length ? `
      <div style="font-size:0.78rem;color:var(--text-muted);margin-bottom:0.5rem">
        <strong style="color:var(--text-secondary)">📚 Sources:</strong> ${sourcesHtml}
      </div>` : ''}
    ${chunksHtml}
  `;
}

// ── PDF Report Upload ────────────────────────────────────────────────────────
async function uploadReport(input) {
  const file = input.files[0];
  if (!file) return;

  const zone = document.getElementById('uploadZone');
  zone.innerHTML = `<div class="spinner-ring mb-2"></div><div class="loading-text">Analyzing report...</div>`;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const data = await api.postForm('/api/analyze-report', formData);
    renderReportResults(data, file.name);
    if (data.error) {
      showToast('Report processing failed', 'error');
    } else {
      showToast(`Report analyzed: ${data.report_type || 'Medical Report'}`, 'success');
    }
  } catch (e) {
    showToast(`Report analysis error: ${e.message}`, 'error');
  } finally {
    zone.innerHTML = `
      <i class="fas fa-cloud-upload-alt" style="font-size:2.5rem;color:var(--primary);margin-bottom:1rem;display:block"></i>
      <div style="font-weight:600;margin-bottom:0.5rem">Drop PDF or Image Here</div>
      <div style="font-size:0.8rem;color:var(--text-muted)">Supports PDF, JPG, PNG • Max 10MB</div>
      <input type="file" id="reportFile" accept=".pdf,.jpg,.jpeg,.png" style="display:none" onchange="uploadReport(this)"/>
    `;
    document.getElementById('reportFile')?.addEventListener('change', function() { uploadReport(this); });
  }
}

function handleFileDrop(event) {
  event.preventDefault();
  document.getElementById('uploadZone').classList.remove('drag-over');
  const file = event.dataTransfer.files[0];
  if (file) {
    const dt  = new DataTransfer();
    dt.items.add(file);
    const inp = document.getElementById('reportFile');
    if (inp) { inp.files = dt.files; uploadReport(inp); }
  }
}

function renderReportResults(data, filename) {
  const container = document.getElementById('reportResults');
  if (!container) return;

  const abnormals = data.abnormal_values || [];
  const labValues = data.lab_values || {};
  const hasError = !!data.error;
  const numLabValues = Object.keys(labValues).length;

  const labRows = Object.entries(labValues).map(([k, v]) => {
    const isAbnormal = abnormals.some(a => a.test.toLowerCase().includes(k.replace('_', ' ')));
    return `
      <tr>
        <td>${k.replace(/_/g, ' ').toUpperCase()}</td>
        <td><strong style="color:${isAbnormal ? 'var(--danger)' : 'var(--success)'}">
          ${v} ${isAbnormal ? '⚠️' : '✓'}
        </strong></td>
        <td>${isAbnormal ? `<span class="badge-pill badge-danger">ABNORMAL</span>` : `<span class="badge-pill badge-success">Normal</span>`}</td>
      </tr>
    `;
  }).join('');

  const recsHtml = (data.recommendations || []).map(r =>
    `<div class="d-flex gap-2 mb-1"><i class="fas fa-chevron-right" style="color:var(--primary);flex-shrink:0;margin-top:4px;font-size:0.7rem"></i><span style="font-size:0.82rem">${r}</span></div>`
  ).join('');

  container.innerHTML = `
    <div class="card-glass">
      <div class="card-header-glass">
        <div class="card-icon card-icon-amber"><i class="fas fa-file-medical-alt"></i></div>
        <div>
          <div class="card-title-glass">${filename}</div>
          <div class="card-subtitle-glass">${data.report_type || 'Medical Report'}</div>
        </div>
        <div class="ms-auto">
          ${hasError
            ? `<span class="badge-pill badge-danger">Processing Error</span>`
            : numLabValues === 0
            ? `<span class="badge-pill badge-secondary">No Data Extracted</span>`
            : abnormals.length > 0
            ? `<span class="badge-pill badge-danger">${abnormals.length} Abnormal Value(s)</span>`
            : `<span class="badge-pill badge-success">All Normal</span>`}
        </div>
      </div>
      <div class="p-4">
        <div class="alert-glass ${hasError ? 'alert-critical' : numLabValues === 0 ? 'alert-info-box' : abnormals.length > 0 ? 'alert-warning-box' : 'alert-success-box'} mb-3">
          <i class="fas ${hasError ? 'fa-times-circle' : numLabValues === 0 ? 'fa-info-circle' : abnormals.length > 0 ? 'fa-exclamation-triangle' : 'fa-check-circle'}"></i>
          <div>${data.error || data.summary}</div>
        </div>
        ${labRows ? `
          <h6 style="color:var(--text-secondary);font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.75rem">Extracted Lab Values</h6>
          <div style="overflow-x:auto;margin-bottom:1.25rem">
            <table class="lab-table">
              <thead><tr><th>Test</th><th>Value</th><th>Status</th></tr></thead>
              <tbody>${labRows}</tbody>
            </table>
          </div>` : ''}
        ${recsHtml ? `
          <h6 style="color:var(--text-secondary);font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.75rem">AI Recommendations</h6>
          <div>${recsHtml}</div>` : ''}
      </div>
    </div>
  `;
}

// ── Clinical NER ─────────────────────────────────────────────────────────────
async function runNER() {
  const text = document.getElementById('nerText')?.value?.trim();
  if (!text) { showToast('Please enter clinical text.', 'warning'); return; }

  try {
    const data = await api.post('/api/ner', { text });
    renderNERResults(data.entities);
  } catch (e) {
    showToast(`NER error: ${e.message}`, 'error');
  }
}

function renderNERResults(entities) {
  const container = document.getElementById('nerResults');
  if (!container) return;

  const typeMap = {
    DISEASE:         { class: 'entity-disease',  icon: '🦠', label: 'Disease' },
    DRUG:            { class: 'entity-drug',     icon: '💊', label: 'Drug' },
    SYMPTOM:         { class: 'entity-symptom',  icon: '🌡️', label: 'Symptom' },
    MEDICAL_HISTORY: { class: 'entity-history',  icon: '📋', label: 'History' },
    LAB_VALUES:      { class: 'entity-lab',      icon: '🧪', label: 'Lab Value' },
  };

  const total = Object.values(entities).reduce((s, v) => s + v.length, 0);
  if (total === 0) {
    container.innerHTML = `<div style="font-size:0.82rem;color:var(--text-muted);text-align:center;padding:0.5rem">No clinical entities detected.</div>`;
    return;
  }

  container.innerHTML = Object.entries(typeMap).map(([type, cfg]) => {
    const items = entities[type] || [];
    if (!items.length) return '';
    return `
      <div style="margin-bottom:0.75rem">
        <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:0.35rem">
          ${cfg.icon} ${cfg.label}
        </div>
        <div>${items.map(item => `<span class="entity-tag ${cfg.class}">${item}</span>`).join('')}</div>
      </div>
    `;
  }).join('');
}

// ── Global clearAll ──────────────────────────────────────────────────────────
function clearAll() {
  clearSymptoms();
  document.getElementById('predictionResults').style.display = 'none';
  document.getElementById('shapPanel').style.display = 'none';
  document.getElementById('drugResults').innerHTML = '';
  document.getElementById('nerResults').innerHTML = '';
  document.getElementById('reportResults').innerHTML = '';
  document.getElementById('ragResults').innerHTML = '';
  if (shapChartInstance) { shapChartInstance.destroy(); shapChartInstance = null; }
  showToast('All panels cleared.', 'info');
}

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  renderSymptomGrid();
});
