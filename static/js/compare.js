/* compare.js — the professor's own model bake-off.
 *
 * Two surfaces share this file:
 *   [data-compare-skill]       on the skill page: the same question to N models
 *                              (POST data-endpoint {message, candidates})
 *   #compare-dialog + buttons  on the grading page: one real submission graded by
 *     [data-action='compare-models'] N models (POST /api/submissions/{id}/compare)
 *
 * Candidate options come from <script id="candidate-json"> rendered by the
 * server: [{provider, model, label, available}]. Nothing here is persisted.
 */
(function () {
  "use strict";

  var candidates = [];
  try {
    var node = document.getElementById("candidate-json");
    if (node) candidates = JSON.parse(node.textContent) || [];
  } catch (e) { candidates = []; }

  function el(tag, className, text) {
    var n = document.createElement(tag);
    if (className) n.className = className;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  async function post(url, body) {
    var r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body)
    });
    var data = null;
    try { data = await r.json(); } catch (e) { data = null; }
    if (!r.ok) {
      var detail = data && data.detail;
      if (Array.isArray(detail)) detail = detail.map(function (d) { return d.msg; }).join("; ");
      throw new Error(detail || r.status + " " + r.statusText);
    }
    return data;
  }

  function fillSelect(select, preferIndex) {
    select.replaceChildren();
    candidates.forEach(function (c, i) {
      var option = el("option", null, c.label + (c.available ? "" : " — no key"));
      option.value = i;
      if (!c.available && c.provider !== "auto") option.disabled = true;
      select.appendChild(option);
    });
    var pick = preferIndex;
    if (pick === undefined || (candidates[pick] && !candidates[pick].available && candidates[pick].provider !== "auto")) {
      pick = 0;
      for (var i = 0; i < candidates.length; i++) {
        if (candidates[i].available && candidates[i].provider !== "auto") { pick = i; break; }
      }
    }
    select.value = String(pick);
  }

  function chosen(selects) {
    var out = [];
    selects.forEach(function (s) {
      var c = candidates[Number(s.value)];
      if (c) out.push({ provider: c.provider, model: c.model });
    });
    return out;
  }

  function head(result) {
    var h = el("div", "compare-head");
    var title = (result.provider || "?") + " · " + (result.model || "?");
    if (result.used_mock) title += " (mock)";
    h.appendChild(el("strong", null, title));
    var meta = [];
    if (typeof result.elapsed_ms === "number") meta.push((result.elapsed_ms / 1000).toFixed(1) + "s");
    if (result.note) meta.push(result.note);
    if (meta.length) h.appendChild(el("span", "fine", meta.join(" · ")));
    return h;
  }

  /* ---- skill page: same question, N replies ------------------------------- */

  document.querySelectorAll("[data-compare-skill]").forEach(function (root) {
    var endpoint = root.getAttribute("data-endpoint");
    var selects = Array.prototype.slice.call(root.querySelectorAll("select[data-compare-pick]"));
    var form = root.querySelector("form");
    var text = root.querySelector("textarea");
    var out = root.querySelector("[data-compare-out]");
    var note = root.querySelector("[data-compare-note]");
    if (!selects.length || !form || !text || !out) return;
    selects.forEach(function (s, i) { fillSelect(s, i === 0 ? 0 : undefined); });

    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      var message = text.value.trim();
      if (!message) { text.focus(); return; }
      var button = form.querySelector("[type='submit']");
      if (button) button.disabled = true;
      if (note) note.textContent = "Asking " + selects.length + " models…";
      out.replaceChildren();
      try {
        var data = await post(endpoint, { message: message, candidates: chosen(selects) });
        (data.results || []).forEach(function (result) {
          var col = el("div", "compare-col card");
          col.appendChild(head(result));
          if (result.error) col.appendChild(el("p", "form-error", result.error));
          col.appendChild(el("div", "compare-reply", result.reply || ""));
          out.appendChild(col);
        });
        if (note) note.textContent = "Nothing here is saved. Pin the model you prefer in the Model section above.";
      } catch (err) {
        if (note) note.textContent = err.message;
      } finally {
        if (button) button.disabled = false;
      }
    });
  });

  /* ---- grading page: one submission, N grades ------------------------------ */

  var dialog = document.getElementById("compare-dialog");
  if (dialog) {
    var dSelects = Array.prototype.slice.call(dialog.querySelectorAll("select[data-compare-pick]"));
    var dOut = dialog.querySelector("[data-compare-out]");
    var dNote = dialog.querySelector("[data-compare-note]");
    var dRun = dialog.querySelector("[data-compare-run]");
    var dClose = dialog.querySelector("[data-compare-close]");
    var dTitle = dialog.querySelector("[data-compare-title]");
    var currentId = null;
    dSelects.forEach(function (s, i) { fillSelect(s, i === 0 ? 0 : undefined); });

    document.addEventListener("click", function (event) {
      var btn = event.target.closest("[data-action='compare-models']");
      if (!btn) return;
      currentId = btn.getAttribute("data-submission-id");
      if (dTitle) dTitle.textContent = "Compare models on " + (btn.getAttribute("data-label") || "this submission");
      if (dOut) dOut.replaceChildren();
      if (dNote) dNote.textContent = "Grades the same submission with each model. Nothing is saved; the current result stays as it is.";
      if (typeof dialog.showModal === "function" && !dialog.open) dialog.showModal();
    });
    if (dClose) dClose.addEventListener("click", function () { dialog.close(); });

    function renderGrades(results) {
      dOut.replaceChildren();
      var table = el("table", "table table--dense compare-table");
      var thead = el("thead");
      var hr = el("tr");
      hr.appendChild(el("th", null, "Criterion"));
      results.forEach(function (r) {
        var th = el("th");
        th.appendChild(head(r));
        hr.appendChild(th);
      });
      thead.appendChild(hr);
      table.appendChild(thead);
      var tbody = el("tbody");
      var keys = [];
      results.forEach(function (r) {
        ((r.grade && r.grade.criteria) || []).forEach(function (c) {
          if (keys.indexOf(c.key) === -1) keys.push(c.key);
        });
      });
      keys.forEach(function (key) {
        var tr = el("tr");
        tr.appendChild(el("td", null, key.replace(/_/g, " ")));
        results.forEach(function (r) {
          var crit = ((r.grade && r.grade.criteria) || []).filter(function (c) { return c.key === key; })[0];
          var td = el("td");
          if (crit) {
            if (crit.score !== null && crit.score !== undefined) {
              td.appendChild(el("strong", null, crit.score + (crit.max_points ? " / " + crit.max_points : "")));
            } else {
              td.appendChild(el("span", "fine", crit.manual ? "yours" : "unscored"));
            }
            if (crit.comment) td.appendChild(el("p", "fine", crit.comment));
          } else {
            td.textContent = "—";
          }
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      var total = el("tr");
      total.appendChild(el("td", null, "Total"));
      results.forEach(function (r) {
        var td = el("td");
        if (r.grade && r.grade.max_score) td.appendChild(el("strong", null, r.grade.overall_score + " / " + r.grade.max_score + " (" + r.grade.percentage + "%)"));
        else if (r.error) td.appendChild(el("span", "form-error", r.error));
        else td.appendChild(el("span", "fine", "feedback only"));
        tr = null;
        total.appendChild(td);
      });
      tbody.appendChild(total);
      var fb = el("tr");
      fb.appendChild(el("td", null, "Feedback"));
      results.forEach(function (r) {
        var td = el("td");
        td.appendChild(el("div", "compare-reply", (r.grade && r.grade.summary_feedback) || ""));
        fb.appendChild(td);
      });
      tbody.appendChild(fb);
      table.appendChild(tbody);
      dOut.appendChild(table);
    }

    if (dRun) {
      dRun.addEventListener("click", async function () {
        if (!currentId) return;
        dRun.disabled = true;
        if (dNote) dNote.textContent = "Grading with " + dSelects.length + " models — this can take a minute…";
        try {
          var data = await post("/api/submissions/" + currentId + "/compare", { candidates: chosen(dSelects) });
          renderGrades(data.results || []);
          if (dNote) dNote.textContent = "Nothing was saved. To use a model for real, pin it on the skill or set it as the preferred provider in Settings.";
        } catch (err) {
          if (dNote) dNote.textContent = err.message;
        } finally {
          dRun.disabled = false;
        }
      });
    }
  }
})();
