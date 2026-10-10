(() => {
  // Same origin by default (FastAPI serves this page).
  // If the frontend is hosted separately, set this to your API URL, e.g. "https://my-api.onrender.com"
  const API_BASE = "";

  const ARC_LENGTH = Math.PI * 90; // length of the gauge arc
  const form = document.getElementById("loan-form");
  const submitBtn = document.getElementById("submit-btn");
  const sampleBtn = document.getElementById("sample-btn");
  const resultEl = document.getElementById("result");
  const gaugeFill = document.getElementById("gauge-fill");
  const tick = document.getElementById("threshold-tick");
  const probNumber = document.getElementById("prob-number");
  const verdictText = document.getElementById("verdict-text");
  const statusEl = document.getElementById("status");
  const pctOutput = document.getElementById("pct-income");
  const factThreshold = document.getElementById("fact-threshold");
  const factMargin = document.getElementById("fact-margin");
  const factPct = document.getElementById("fact-pct");

  const incomeInput = form.elements["person_income"];
  const amountInput = form.elements["loan_amnt"];

  let countFrame = null;
  let slowTimer = null;

  /* ---------- loan % of income (calculated) ---------- */
  function currentRatio() {
    const income = parseFloat(incomeInput.value);
    const amount = parseFloat(amountInput.value);
    if (!(income > 0) || !(amount > 0)) return null;
    return amount / income;
  }

  function updateRatio() {
    const r = currentRatio();
    if (r === null) {
      pctOutput.textContent = "Calculated from the amounts";
      pctOutput.classList.remove("has-value");
    } else {
      pctOutput.textContent = (r * 100).toFixed(1) + "% of income";
      pctOutput.classList.add("has-value");
    }
  }
  incomeInput.addEventListener("input", updateRatio);
  amountInput.addEventListener("input", updateRatio);

  /* ---------- helpers ---------- */
  const setState = (state, risk) => {
    resultEl.dataset.state = state;
    if (risk) resultEl.dataset.risk = risk; else delete resultEl.dataset.risk;
  };

  function setStatus(message, isError = false) {
    statusEl.textContent = message;
    statusEl.classList.toggle("error", isError);
  }

  function setLoading(isLoading) {
    submitBtn.classList.toggle("loading", isLoading);
    submitBtn.disabled = isLoading;
  }

  function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

  function countUp(el, target, duration = 1300) {
    cancelAnimationFrame(countFrame);
    const start = performance.now();
    const step = (now) => {
      const t = Math.min((now - start) / duration, 1);
      el.textContent = (target * easeOutCubic(t)).toFixed(1) + "%";
      if (t < 1) countFrame = requestAnimationFrame(step);
    };
    countFrame = requestAnimationFrame(step);
  }

  function placeThreshold(t) {
    const clamped = Math.min(Math.max(t, 0), 1);
    const angle = Math.PI * clamped;
    const point = (r) => ({
      x: 110 - r * Math.cos(angle),
      y: 110 - r * Math.sin(angle),
    });
    const a = point(78);
    const b = point(102);
    tick.setAttribute("x1", a.x.toFixed(2));
    tick.setAttribute("y1", a.y.toFixed(2));
    tick.setAttribute("x2", b.x.toFixed(2));
    tick.setAttribute("y2", b.y.toFixed(2));
  }

  function resetResult() {
    cancelAnimationFrame(countFrame);
    gaugeFill.style.strokeDashoffset = ARC_LENGTH;
    probNumber.textContent = "\u2013";
    verdictText.textContent = "Waiting for an application";
    factThreshold.textContent = factMargin.textContent = factPct.textContent = "\u2013";
    setStatus("");
    setState("idle");
  }

  /* ---------- validation ---------- */
  function validate() {
    let firstBad = null;
    form.querySelectorAll("input[type=number], select").forEach((el) => {
      const ok = el.checkValidity();
      el.classList.toggle("invalid", !ok);
      if (!ok && !firstBad) firstBad = el;
    });
    if (firstBad) {
      firstBad.focus();
      firstBad.reportValidity();
      return false;
    }
    return true;
  }

  form.addEventListener("input", (e) => e.target.classList.remove("invalid"));

  /* ---------- build payload ---------- */
  function buildPayload() {
    const f = form.elements;
    const income = parseFloat(f["person_income"].value);
    const amount = parseFloat(f["loan_amnt"].value);
    return {
      person_age: parseInt(f["person_age"].value, 10),
      person_income: income,
      person_home_ownership: f["person_home_ownership"].value,
      person_emp_length: parseFloat(f["person_emp_length"].value),
      loan_intent: f["loan_intent"].value,
      loan_grade: f["loan_grade"].value,
      loan_amnt: amount,
      loan_int_rate: parseFloat(f["loan_int_rate"].value),
      loan_percent_income: Math.round((amount / income) * 10000) / 10000,
      cb_person_default_on_file: f["cb_person_default_on_file"].value,
      cb_person_cred_hist_length: parseInt(f["cb_person_cred_hist_length"].value, 10),
    };
  }

  function errorMessage(status, body) {
    if (status === 422 && body && Array.isArray(body.detail)) {
      const first = body.detail[0];
      const field = first.loc ? first.loc[first.loc.length - 1] : "a field";
      return `The server rejected ${String(field).replace(/_/g, " ")}: ${first.msg}.`;
    }
    if (status >= 500) return "The server hit an error while scoring this application. Check the server logs.";
    return `Request failed (${status}). Please try again.`;
  }

  /* ---------- submit ---------- */
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!validate()) return;

    const payload = buildPayload();
    setLoading(true);
    setState("loading");
    setStatus("Scoring application\u2026");
    probNumber.textContent = "\u2013";
    gaugeFill.style.strokeDashoffset = 0;

    // Free Render instances sleep when idle; tell the user why it is slow.
    slowTimer = setTimeout(() => {
      setStatus("The server is waking up. The first request can take up to a minute.");
    }, 4000);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      let body = null;
      try { body = await res.json(); } catch (_) {}
      if (!res.ok) throw new Error(errorMessage(res.status, body));
      showResult(body, payload);
    } catch (err) {
      resetResult();
      const offline = err instanceof TypeError;
      setStatus(offline ? "Could not reach the server. Check your connection and try again." : err.message, true);
    } finally {
      clearTimeout(slowTimer);
      setLoading(false);
    }
  });

  function showResult(data, payload) {
    const p = Math.min(Math.max(data.default_probability, 0), 1);
    const t = data.best_threshold;
    const risk = data.default_prediction === 1 ? "high" : "low";

    setStatus("");
    setState("done", risk);
    placeThreshold(t);

    // Restart the transition from empty so the sweep always plays.
    gaugeFill.style.transition = "none";
    gaugeFill.style.strokeDasharray = ARC_LENGTH;
    gaugeFill.style.strokeDashoffset = ARC_LENGTH;
    void gaugeFill.getBoundingClientRect();
    gaugeFill.style.transition = "";
    gaugeFill.style.strokeDashoffset = ARC_LENGTH * (1 - p);

    countUp(probNumber, p * 100);
    verdictText.textContent = data.Result;

    const diff = (p - t) * 100;
    factThreshold.textContent = (t * 100).toFixed(1) + "%";
    factMargin.textContent = `${Math.abs(diff).toFixed(1)} pts ${diff >= 0 ? "above" : "below"}`;
    factPct.textContent = (payload.loan_percent_income * 100).toFixed(1) + "%";

    if (window.matchMedia("(max-width: 900px)").matches) {
      resultEl.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  /* ---------- sample + reset ---------- */
  sampleBtn.addEventListener("click", () => {
    const sample = {
      person_age: 28,
      person_income: 55000,
      person_emp_length: 4,
      loan_amnt: 12000,
      loan_int_rate: 11.5,
      cb_person_cred_hist_length: 5,
      loan_intent: "EDUCATION",
    };
    Object.entries(sample).forEach(([k, v]) => {
      form.elements[k].value = v;
      form.elements[k].classList.remove("invalid");
    });
    form.elements["person_home_ownership"].value = "RENT";
    form.elements["loan_grade"].value = "C";
    form.elements["cb_person_default_on_file"].value = "N";
    updateRatio();
  });

  form.addEventListener("reset", () => {
    // Let the browser restore defaults first, then refresh the dependent UI.
    setTimeout(() => {
      form.querySelectorAll(".invalid").forEach((el) => el.classList.remove("invalid"));
      updateRatio();
      resetResult();
    }, 0);
  });

  // Initial state
  gaugeFill.style.strokeDasharray = ARC_LENGTH;
  gaugeFill.style.strokeDashoffset = ARC_LENGTH;
  placeThreshold(0.5);
  updateRatio();
})();
