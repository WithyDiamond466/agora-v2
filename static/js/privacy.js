/* privacy.js — Privacy Guard UI (Increment 1, Feature B).
 *
 * Three independent mounts; each no-ops when its element is absent:
 *
 *   #privacy-settings  (templates/settings.html)  — mode radios, the local
 *                      sweep switch, the pseudonym map ("who is Person-A?")
 *                      and the local model card with its health status.
 *   #privacy-banner    (templates/grading.html)   — the mode banner, the
 *                      warn-mode pre-flight, and the per-submission scan chip
 *                      that opens the PrivacyScan report.
 *   [data-feedback-for] blocks (templates/grading.html) — professor-facing
 *                      feedback with codes swapped back to real names, plus a
 *                      toggle to read exactly what the AI saw.
 *
 * ENDPOINTS (as served by app/routers/settings.py + app/routers/grading.py;
 * every one is overridable from a data-* attribute on the mount, and every
 * failure degrades to "that piece is hidden" rather than a broken page):
 *
 *   GET  /api/settings/privacy   → {mode, llm_sweep, llm_sweep_active,
 *                                   local_model: {enabled, base_url, model},
 *                                   courses: [{id, name, pseudonyms}]}
 *   POST /api/settings/privacy   {mode?} | {llm_sweep?} | {local_model: {...}}
 *   GET  /api/settings/local-model/health
 *                                → {ok, model, base_url, message, label}
 *   GET  /api/settings/privacy/pseudonyms?course_id=N
 *                                → {courses: [{course_id, course_name, count,
 *                                    entries: [{id, code, original_text, kind,
 *                                    source, created_at}]}]}
 *   GET  /api/submissions/{id}/privacy        → {scan, headline, mode}
 *   POST /api/submissions/{id}/privacy/scan   → {scan}  (detection only)
 *   GET  /api/submissions/{id}/result/display → {summary_feedback: {raw,
 *                                    display, replacements: [{code, name}]},
 *                                    criteria: [{key, raw, display, ...}]}
 *
 *   scan = {id, submission_id, mode, created_at, total, swapped, headline,
 *           report: {findings: [{text, code, kind, source, occurrences}],
 *                    counts, warnings, llm_sweep}}
 *
 * grading.js awaits window.AgoraPrivacy.confirmRun() before it starts a batch —
 * that is the only seam between the two files.
 *
 * Everything is written with textContent: scan findings carry real student PII
 * and model-written feedback is untrusted.
 */
(function () {
  "use strict";

  /* ---- helpers ------------------------------------------------------------ */

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function announce(text) {
    var live = document.getElementById("live-region");
    if (live) live.textContent = text;
  }

  async function call(url, method, body) {
    var init = { method: method || "GET", headers: { Accept: "application/json" } };
    if (body !== undefined && body !== null) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    var response;
    try {
      response = await fetch(url, init);
    } catch (err) {
      return { ok: false, status: 0, data: null, error: "No answer from the server." };
    }
    var data = null;
    try { data = await response.json(); } catch (err) { data = null; }
    var detail = data && data.detail;
    if (Array.isArray(detail)) {
      detail = detail.map(function (d) { return d.msg || "invalid value"; }).join("; ");
    }
    return {
      ok: response.ok,
      status: response.status,
      data: data,
      error: typeof detail === "string" ? detail : response.status + " " + response.statusText
    };
  }

  var MODES = ["off", "warn", "swap"];

  function readMode(data) {
    if (!data || typeof data !== "object") return null;
    var mode = data.mode || (data.privacy && data.privacy.mode);
    mode = String(mode || "").toLowerCase();
    return MODES.indexOf(mode) === -1 ? null : mode;
  }

  function scanOf(data) {
    if (!data || typeof data !== "object") return null;
    var scan = data.scan !== undefined ? data.scan : data;
    if (!scan || typeof scan !== "object") return null;
    var report = scan.report && typeof scan.report === "object" ? scan.report : {};
    var findings = Array.isArray(report.findings) ? report.findings : [];
    var counts = report.counts || {};
    return {
      submission_id: scan.submission_id,
      mode: String(scan.mode || "").toLowerCase(),
      created_at: scan.created_at,
      total: Number(scan.total !== undefined ? scan.total : counts.total) || 0,
      swapped: !!scan.swapped,
      headline: scan.headline || data.headline || "",
      findings: findings,
      warnings: Array.isArray(report.warnings) ? report.warnings : [],
      sweep: report.llm_sweep || {}
    };
  }

  var SOURCE_LABEL = {
    roster: "roster name",
    pattern: "pattern rule",
    regex: "pattern rule",
    llm: "local model sweep",
    sweep: "local model sweep",
    map: "saved code"
  };

  function sourceLabel(value) {
    var key = String(value || "").toLowerCase();
    return SOURCE_LABEL[key] || key || "—";
  }

  function shortDate(value) {
    if (!value) return "";
    var when = new Date(value);
    if (isNaN(when.getTime())) return String(value).slice(0, 10);
    return when.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }

  /* ====================================================================== */
  /* Settings → Privacy                                                     */
  /* ====================================================================== */

  var settingsRoot = document.getElementById("privacy-settings");
  if (settingsRoot) {
    var sUrls = {
      privacy: settingsRoot.getAttribute("data-api-privacy") || "/api/settings/privacy",
      pseudonyms: settingsRoot.getAttribute("data-api-pseudonyms") || "/api/settings/privacy/pseudonyms",
      health: settingsRoot.getAttribute("data-api-local-health") || "/api/settings/local-model/health"
    };
    var statusLine = settingsRoot.querySelector("[data-privacy-status]");
    var sweepBox = settingsRoot.querySelector("[data-privacy-sweep]");
    var sweepNote = settingsRoot.querySelector("[data-privacy-sweep-note]");
    var courseSelect = settingsRoot.querySelector("[data-pseudonym-course]");
    var mapBody = settingsRoot.querySelector("[data-pseudonym-body]");
    var mapNote = settingsRoot.querySelector("[data-pseudonym-note]");
    var mapTable = settingsRoot.querySelector("[data-pseudonym-table]");
    /* The local model card is its own .form-section, so these live outside
       #privacy-settings — query the document, not the mount. */
    var localForm = document.querySelector("[data-local-form]");
    var localHealth = document.querySelector("[data-local-health]");
    var localDetail = document.querySelector("[data-local-detail]");
    var localName = document.querySelector("[data-local-name]");

    var setStatus = function (text) { if (statusLine) statusLine.textContent = text || ""; };

    function applySettings(data) {
      var mode = readMode(data);
      if (mode) {
        var radio = settingsRoot.querySelector("[data-privacy-mode][value='" + mode + "']");
        if (radio) radio.checked = true;
      }
      if (sweepBox && data.llm_sweep !== undefined) sweepBox.checked = !!data.llm_sweep;
      if (sweepNote) {
        sweepNote.textContent = data.llm_sweep === false
          ? "Off — only the roster and the pattern rules run."
          : data.llm_sweep_active === false
            ? "On, but the local model is not answering, so only the deterministic pass runs."
            : "On — the local model reads the text after the deterministic pass.";
      }
      var local = data.local_model || {};
      if (localForm) {
        if (localForm.elements.base_url && local.base_url) localForm.elements.base_url.value = local.base_url;
        if (localForm.elements.model && local.model) localForm.elements.model.value = local.model;
        if (localForm.elements.enabled) localForm.elements.enabled.checked = local.enabled !== false;
      }
      if (localName && local.model) localName.textContent = local.model;

      if (courseSelect && Array.isArray(data.courses)) {
        var current = courseSelect.value;
        courseSelect.replaceChildren();
        if (!data.courses.length) {
          var none = el("option", null, "no courses yet");
          none.value = "";
          courseSelect.appendChild(none);
        } else {
          data.courses.forEach(function (course) {
            var option = el("option", null,
              course.name + " · " + (course.pseudonyms || 0) + " code" +
              (course.pseudonyms === 1 ? "" : "s"));
            option.value = course.id;
            courseSelect.appendChild(option);
          });
          if (current) courseSelect.value = current;
          loadPseudonyms(courseSelect.value);
        }
      }
    }

    async function loadSettings() {
      var res = await call(sUrls.privacy, "GET");
      if (!res.ok) {
        setStatus("Agora could not read the privacy settings (" + res.error +
          "). The default — swapping identifiers — applies.");
        var fallback = settingsRoot.querySelector("[data-privacy-mode][value='swap']");
        if (fallback) fallback.checked = true;
        return;
      }
      applySettings(res.data || {});
    }

    async function save(payload, note) {
      setStatus("Saving…");
      var res = await call(sUrls.privacy, "POST", payload);
      if (!res.ok) { setStatus("Could not save that: " + res.error); return null; }
      applySettings(res.data || {});
      setStatus(note);
      announce(note);
      return res.data;
    }

    settingsRoot.addEventListener("change", function (event) {
      var radio = event.target.closest("[data-privacy-mode]");
      if (radio && radio.checked) {
        save({ mode: radio.value },
          radio.value === "swap"
            ? "Saved. Submissions leave as pseudonymized text."
            : radio.value === "warn"
              ? "Saved. You will see the findings before each grading run."
              : "Saved. Submissions go out exactly as written — names included.");
        return;
      }
      if (sweepBox && event.target === sweepBox) {
        save({ llm_sweep: sweepBox.checked },
          sweepBox.checked ? "Saved. The local model sweeps for missed names."
                           : "Saved. Only the deterministic pass runs.");
      }
    });

    /* -- the pseudonym map -- */

    function showMapNote(text) {
      if (mapNote) mapNote.textContent = text || "";
      if (mapTable) mapTable.hidden = !!text;
    }

    async function loadPseudonyms(courseId) {
      if (!mapBody) return;
      var url = sUrls.pseudonyms + (courseId ? "?course_id=" + encodeURIComponent(courseId) : "");
      var res = await call(url, "GET");
      mapBody.replaceChildren();
      if (!res.ok) {
        showMapNote("Could not read the pseudonym map: " + res.error);
        return;
      }
      var groups = (res.data && res.data.courses) || [];
      var entries = [];
      groups.forEach(function (group) {
        (group.entries || []).forEach(function (entry) { entries.push(entry); });
      });
      if (!entries.length) {
        showMapNote("No codes for this course yet. Roster students get theirs from their " +
          "number; everyone else is added the first time they appear in a submission.");
        return;
      }
      showMapNote("");
      entries.forEach(function (entry) {
        var row = el("tr");
        row.appendChild(el("td", "mono", entry.code || "—"));
        row.appendChild(el("td", null, entry.original_text || entry.original || "—"));
        row.appendChild(el("td", null, entry.kind || "person"));
        row.appendChild(el("td", null, sourceLabel(entry.source)));
        row.appendChild(el("td", null, entry.created_at ? shortDate(entry.created_at) : "—"));
        mapBody.appendChild(row);
      });
    }

    if (courseSelect) {
      courseSelect.addEventListener("change", function () { loadPseudonyms(courseSelect.value); });
    }

    /* -- the local model card -- */

    function setHealth(state, text) {
      if (!localHealth) return;
      localHealth.className = "pill pill--" + state;
      localHealth.textContent = text;
    }

    async function checkHealth() {
      setHealth("grading", "checking…");
      if (localDetail) localDetail.textContent = "";
      var res = await call(sUrls.health, "GET");
      if (!res.ok) {
        setHealth("failed", "not reachable");
        if (localDetail) localDetail.textContent = res.error;
        return;
      }
      var data = res.data || {};
      if (data.ok) {
        setHealth("graded", "connected");
        if (localName && data.model) localName.textContent = data.model;
        if (localDetail) {
          localDetail.textContent = [data.model, data.base_url].filter(Boolean).join(" · ");
        }
      } else {
        setHealth("failed", "not reachable");
        if (localDetail) {
          localDetail.textContent = data.message ||
            "Nothing answered at " + (data.base_url || "the address above") + ".";
        }
      }
    }

    if (localForm) {
      localForm.addEventListener("submit", async function (event) {
        event.preventDefault();
        var submit = localForm.querySelector("[type='submit']");
        if (submit) { submit.disabled = true; submit.classList.add("is-busy"); }
        await save({
          local_model: {
            base_url: localForm.elements.base_url ? localForm.elements.base_url.value.trim() : "",
            model: localForm.elements.model ? localForm.elements.model.value.trim() : "",
            enabled: localForm.elements.enabled ? localForm.elements.enabled.checked : true
          }
        }, "Saved the local model address.");
        if (submit) { submit.disabled = false; submit.classList.remove("is-busy"); }
        checkHealth();
      });
    }

    var healthButton = document.querySelector("[data-local-check]");
    if (healthButton) healthButton.addEventListener("click", checkHealth);

    loadSettings();
    checkHealth();
  }

  /* ====================================================================== */
  /* Grading page — banner, pre-flight, scan chips, feedback substitution    */
  /* ====================================================================== */

  var gradingRoot = document.getElementById("grading-root");
  var bannerRoot = document.getElementById("privacy-banner");

  if (gradingRoot && bannerRoot) {
    var gUrls = {
      mode: bannerRoot.getAttribute("data-api-mode") || "/api/settings/privacy",
      scan: bannerRoot.getAttribute("data-api-scan") || "/api/submissions/{id}/privacy",
      rescan: bannerRoot.getAttribute("data-api-rescan") || "/api/submissions/{id}/privacy/scan",
      display: bannerRoot.getAttribute("data-api-display") || "/api/submissions/{id}/result/display"
    };
    var settingsHref = bannerRoot.getAttribute("data-settings-href") || "/settings";
    var state = { mode: null, scans: {}, feedback: {} };

    function url(pattern, id) { return pattern.replace("{id}", encodeURIComponent(id)); }

    /* -- the standing banner ------------------------------------------------ */

    var BANNER = {
      swap: {
        cls: "notice notice--info",
        title: "Detected identifiers are swapped before cloud requests.",
        body: "Detected names, emails, phone numbers and id numbers become stable codes — " +
          "Student-07, Person-A, [EMAIL-1] — and the same person keeps the same code, so you " +
          "and the model can still follow who is who. Swapping happens in text, so a PDF goes " +
          "out as locally extracted text rather than the original file: layout and handwriting " +
          "fidelity are lost. Cloud grading stops without sending an image or a document with no " +
          "extractable text. Detection can miss identifying details; review the scan before using coursework."
      },
      warn: {
        cls: "notice notice--warn",
        title: "Warn mode: you decide, run by run.",
        body: "Agora scans each submission for identifiers and shows you what it found before " +
          "the batch goes out. Nothing is replaced unless you switch to swapping."
      },
      off: {
        cls: "notice notice--error",
        title: "The Privacy Guard is off.",
        body: "Submissions go to the cloud provider exactly as written — student names, emails " +
          "and anything else in the file included."
      }
    };

    function renderBanner() {
      var existing = bannerRoot.querySelector("[data-privacy-notice]");
      if (existing) existing.remove();
      var meta = BANNER[state.mode];
      if (!meta) return;
      var box = el("details", meta.cls + " privacy-note");
      box.setAttribute("data-privacy-notice", "");
      var summary = el("summary", null, meta.title);
      box.appendChild(summary);
      box.appendChild(el("p", null, meta.body));
      var link = el("a", null, "Change this in Settings → Privacy");
      link.href = settingsHref + "#privacy-settings";
      box.appendChild(link);
      bannerRoot.insertBefore(box, bannerRoot.firstChild);
    }

    /* -- warn-mode pre-flight ----------------------------------------------- */

    function queueLabel(submissionId) {
      var row = document.querySelector('.queue-row[data-submission-id="' + submissionId + '"]');
      if (!row) return "submission " + submissionId;
      var num = row.querySelector(".snum");
      var name = row.querySelector(".q-file .name");
      return [num ? num.textContent.trim() : "", name ? name.textContent.trim() : ""]
        .filter(Boolean).join(" ") || "submission " + submissionId;
    }

    function preflightPanel(rows, warnings, resolve) {
      var old = bannerRoot.querySelector(".privacy-preflight");
      if (old) old.remove();

      var panel = el("section", "card privacy-preflight");
      panel.setAttribute("aria-label", "Privacy check before grading");
      var head = el("div", "card-head");
      head.appendChild(el("h3", null, "Before this batch goes out"));
      head.appendChild(el("span", "fine", "warn mode"));
      panel.appendChild(head);

      var total = rows.reduce(function (sum, row) { return sum + (row.total || 0); }, 0);
      panel.appendChild(el("p", null, total
        ? "Agora found " + total + " identifier" + (total === 1 ? "" : "s") + " across " +
          rows.length + " submission" + (rows.length === 1 ? "" : "s") + ". Sent as written, " +
          "they go to the provider unchanged."
        : "Agora found no identifiers in " + rows.length + " submission" +
          (rows.length === 1 ? "" : "s") + "."));

      if (rows.length) {
        var table = el("table", "table table--dense");
        var thead = el("thead");
        var hrow = el("tr");
        ["Submission", "Found", "What it saw"].forEach(function (label) {
          var th = el("th", label === "Found" ? "num" : null, label);
          th.scope = "col";
          hrow.appendChild(th);
        });
        thead.appendChild(hrow);
        table.appendChild(thead);
        var tbody = el("tbody");
        rows.forEach(function (row) {
          var tr = el("tr");
          tr.appendChild(el("td", null, queueLabel(row.submission_id)));
          tr.appendChild(el("td", "num", String(row.total || 0)));
          var found = el("td");
          (row.findings || []).slice(0, 6).forEach(function (finding) {
            found.appendChild(el("span", "tag tag--neutral", finding.text || finding.code || "identifier"));
          });
          if ((row.findings || []).length > 6) {
            found.appendChild(el("span", "fine", "+" + (row.findings.length - 6) + " more"));
          }
          if (!(row.findings || []).length) found.appendChild(el("span", "fine", "—"));
          tr.appendChild(found);
          tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        panel.appendChild(table);
      }

      warnings.forEach(function (warning) {
        panel.appendChild(el("p", "fine privacy-warning", "⚠ " + warning));
      });
      panel.appendChild(el("p", "fine",
        "Switching to swapping changes the setting for every course, and sends submissions as " +
        "extracted text instead of the original file."));

      var foot = el("div", "form-foot");
      [
        { choice: "cancel", cls: "btn", label: "Cancel" },
        { choice: "as-written", cls: "btn", label: "Send as written" },
        { choice: "swap", cls: "btn btn--primary", label: "Switch to swapping, then grade" }
      ].forEach(function (spec) {
        var button = el("button", spec.cls, spec.label);
        button.type = "button";
        button.setAttribute("data-privacy-choice", spec.choice);
        foot.appendChild(button);
      });
      panel.appendChild(foot);

      panel.addEventListener("click", async function (event) {
        var button = event.target.closest("[data-privacy-choice]");
        if (!button) return;
        var choice = button.getAttribute("data-privacy-choice");
        if (choice === "cancel") { panel.remove(); resolve(false); return; }
        if (choice === "as-written") {
          panel.remove();
          announce("Grading with the text as written.");
          resolve(true);
          return;
        }
        button.disabled = true;
        button.textContent = "Switching…";
        var res = await call(gUrls.mode, "POST", { mode: "swap" });
        if (!res.ok) {
          button.disabled = false;
          button.textContent = "Switch to swapping, then grade";
          var error = panel.querySelector(".form-error") || el("p", "form-error");
          error.textContent = "Could not switch modes: " + res.error;
          error.hidden = false;
          panel.appendChild(error);
          return;
        }
        state.mode = readMode(res.data) || "swap";
        renderBanner();
        panel.remove();
        announce("Now swapping identifiers. Grading.");
        resolve(true);
      });

      bannerRoot.appendChild(panel);
      var primary = panel.querySelector(".btn--primary");
      if (primary) primary.focus();
      panel.scrollIntoView({ block: "nearest" });
    }

    function preflightUnavailable(message, resolve) {
      var old = bannerRoot.querySelector(".privacy-preflight");
      if (old) old.remove();
      var panel = el("section", "card privacy-preflight");
      panel.appendChild(el("strong", null, "The privacy check could not run."));
      panel.appendChild(el("p", null, message +
        " Grading will do whatever the server's current mode does."));
      var foot = el("div", "form-foot");
      [
        { choice: "cancel", cls: "btn", label: "Cancel" },
        { choice: "go", cls: "btn btn--primary", label: "Grade anyway" }
      ].forEach(function (spec) {
        var button = el("button", spec.cls, spec.label);
        button.type = "button";
        button.setAttribute("data-privacy-choice", spec.choice);
        foot.appendChild(button);
      });
      panel.appendChild(foot);
      panel.addEventListener("click", function (event) {
        var button = event.target.closest("[data-privacy-choice]");
        if (!button) return;
        panel.remove();
        resolve(button.getAttribute("data-privacy-choice") === "go");
      });
      bannerRoot.appendChild(panel);
      panel.scrollIntoView({ block: "nearest" });
    }

    /* The scan is a local-model read of each submission — seconds apiece — so
       the wait is visible and abandonable, and the batch is capped. */
    var SCAN_LIMIT = 12;
    var scanning = { cancelled: false };

    function waitingPanel(done, total) {
      var panel = bannerRoot.querySelector(".privacy-preflight--waiting");
      if (done >= total) {
        if (panel) panel.remove();
        return;
      }
      if (!panel) {
        panel = el("section", "card privacy-preflight privacy-preflight--waiting");
        panel.appendChild(el("p", null, ""));
        var stop = el("button", "btn btn--sm", "Stop");
        stop.type = "button";
        stop.addEventListener("click", function () { scanning.cancelled = true; });
        panel.appendChild(stop);
        bannerRoot.appendChild(panel);
      }
      panel.firstChild.textContent =
        "Reading each submission for identifiers… " + done + " of " + total;
    }

    /* grading.js awaits this before POSTing a batch. */
    async function confirmRun(submissionIds) {
      if (state.mode !== "warn") return true;
      var all = submissionIds || [];
      var ids = all.slice(0, SCAN_LIMIT);
      if (!ids.length) return true;

      var rows = [];
      var warnings = [];
      var failed = 0;
      scanning.cancelled = false;
      for (var i = 0; i < ids.length; i++) {
        waitingPanel(i, ids.length);
        if (scanning.cancelled) break;
        var res = await call(url(gUrls.rescan, ids[i]), "POST", {});
        if (!res.ok) { failed += 1; continue; }
        var scan = scanOf(res.data);
        if (!scan) continue;
        scan.submission_id = scan.submission_id || ids[i];
        state.scans[scan.submission_id] = scan;
        paintScan(scan.submission_id, scan);
        rows.push(scan);
        scan.warnings.forEach(function (warning) {
          if (warnings.indexOf(warning) === -1) warnings.push(warning);
        });
      }
      waitingPanel(ids.length, ids.length);
      if (scanning.cancelled) return false;

      if (!rows.length) {
        return new Promise(function (resolve) {
          preflightUnavailable(failed
            ? "The scanner answered with an error for every submission."
            : "Nothing could be scanned.", resolve);
        });
      }
      if (failed) warnings.push(failed + " submission(s) could not be scanned.");
      if (all.length > ids.length) {
        warnings.push("Checked the first " + ids.length + " of " + all.length +
          " submissions — the rest go out under the same decision.");
      }
      return new Promise(function (resolve) { preflightPanel(rows, warnings, resolve); });
    }

    window.AgoraPrivacy = {
      confirmRun: confirmRun,
      mode: function () { return state.mode; }
    };

    /* -- per-submission scan chip + report ---------------------------------- */

    function chipFor(submissionId, scan) {
      var chip = el("button", "privacy-chip");
      chip.type = "button";
      chip.setAttribute("data-submission-id", submissionId);
      if (!scan) {
        chip.className += " privacy-chip--unscanned";
        chip.appendChild(el("span", "privacy-chip-label", "not scanned"));
        chip.appendChild(el("span", "privacy-chip-view", "scan now"));
        return chip;
      }
      var variant = scan.swapped ? "swapped"
        : scan.mode === "off" ? "exposed"
        : scan.total ? "found" : "clean";
      chip.className += " privacy-chip--" + variant;
      chip.appendChild(el("span", "privacy-chip-label",
        scan.headline || (scan.total + " identifiers")));
      chip.appendChild(el("span", "privacy-chip-view", "view report"));
      chip.setAttribute("aria-expanded", "false");
      return chip;
    }

    function reportBody(scan) {
      var wrap = el("div", "privacy-report-body");
      wrap.appendChild(el("p", "fine",
        "Scanned in " + (scan.mode || "unknown") + " mode" +
        (scan.created_at ? " on " + shortDate(scan.created_at) : "") +
        (scan.sweep && scan.sweep.ran
          ? " · local sweep: " + (scan.sweep.model || scan.sweep.provider || "on")
          : " · deterministic pass only") +
        (scan.findings.length
          ? ". Names below never left this machine."
          : ". Nothing here left this machine.")));

      if (scan.findings.length) {
        var table = el("table", "table table--dense");
        var thead = el("thead");
        var hrow = el("tr");
        ["Code", "Stood in for", "Kind", "Caught by", "Times"].forEach(function (label) {
          var th = el("th", label === "Times" ? "num" : null, label);
          th.scope = "col";
          hrow.appendChild(th);
        });
        thead.appendChild(hrow);
        table.appendChild(thead);
        var tbody = el("tbody");
        scan.findings.forEach(function (finding) {
          var tr = el("tr");
          tr.appendChild(el("td", "mono", finding.code || "— not swapped —"));
          tr.appendChild(el("td", null, finding.text || "—"));
          tr.appendChild(el("td", null, finding.kind || "person"));
          tr.appendChild(el("td", null, sourceLabel(finding.source)));
          tr.appendChild(el("td", "num", String(finding.occurrences || 1)));
          tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        wrap.appendChild(table);
      } else {
        wrap.appendChild(el("p", "fine", "Nothing matched — this text carried no identifiers."));
      }

      scan.warnings.forEach(function (warning) {
        wrap.appendChild(el("p", "fine privacy-warning", "⚠ " + warning));
      });
      return wrap;
    }

    function paintScan(submissionId, scan) {
      var slot = document.querySelector('[data-privacy-chip="' + submissionId + '"]');
      var report = document.querySelector('[data-privacy-report="' + submissionId + '"]');
      if (!slot) return;
      var chip = chipFor(submissionId, scan);
      slot.replaceChildren(chip);
      if (report) report.hidden = true;

      if (!scan) {
        chip.addEventListener("click", async function () {
          chip.disabled = true;
          chip.classList.add("is-busy");
          var res = await call(url(gUrls.rescan, submissionId), "POST", {});
          if (!res.ok) {
            chip.disabled = false;
            chip.classList.remove("is-busy");
            chip.replaceChildren(el("span", "privacy-chip-label", "scan failed: " + res.error));
            return;
          }
          var fresh = scanOf(res.data);
          state.scans[submissionId] = fresh;
          paintScan(submissionId, fresh);
        });
        return;
      }

      if (!report) return;
      report.replaceChildren(reportBody(scan));
      chip.addEventListener("click", function () {
        report.hidden = !report.hidden;
        chip.setAttribute("aria-expanded", report.hidden ? "false" : "true");
      });
    }

    async function loadScan(submissionId) {
      if (!submissionId || state.scans[submissionId] !== undefined) return;
      state.scans[submissionId] = null;
      var res = await call(url(gUrls.scan, submissionId), "GET");
      if (!res.ok) return;
      var scan = scanOf(res.data);
      state.scans[submissionId] = scan;
      paintScan(submissionId, scan);
    }

    /* -- feedback: codes back to real names --------------------------------- */

    function splitByCodes(text, replacements) {
      var codes = replacements.map(function (r) { return r.code; }).filter(Boolean)
        .sort(function (a, b) { return b.length - a.length; });
      if (!codes.length) return [{ text: text }];
      var pattern = new RegExp("(" + codes.map(function (code) {
        return code.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      }).join("|") + ")", "g");
      var byCode = {};
      replacements.forEach(function (r) { if (r.code) byCode[r.code] = r; });
      return text.split(pattern).filter(function (piece) { return piece !== ""; })
        .map(function (piece) {
          return byCode[piece] ? { text: piece, sub: byCode[piece] } : { text: piece };
        });
    }

    function paintText(node, entry, view) {
      node.replaceChildren();
      if (!entry.replacements.length) {
        node.textContent = view === "raw" ? entry.raw : (entry.display || entry.raw);
        return;
      }
      splitByCodes(entry.raw, entry.replacements).forEach(function (piece) {
        if (!piece.sub) { node.appendChild(document.createTextNode(piece.text)); return; }
        var name = piece.sub.name || piece.text;
        if (view === "raw") {
          var code = el("span", "code-mark", piece.text);
          code.title = "really: " + name;
          node.appendChild(code);
        } else {
          var real = el("span", "subbed", name);
          real.title = "the model saw “" + piece.text + "”";
          node.appendChild(real);
        }
      });
    }

    function feedbackToggle(pane, entries, count) {
      var block = entries[0].node;
      var existing = pane.querySelector(".feedback-toggle");
      if (existing) existing.remove();

      var bar = el("div", "feedback-toggle");
      bar.setAttribute("role", "group");
      bar.setAttribute("aria-label", "Feedback view");
      bar.appendChild(el("span", "fine",
        count + " name" + (count === 1 ? "" : "s") + " put back"));

      [
        { view: "names", label: "Real names" },
        { view: "raw", label: "What the AI saw" }
      ].forEach(function (spec) {
        var button = el("button", "btn btn--quiet btn--sm", spec.label);
        button.type = "button";
        button.setAttribute("data-feedback-view", spec.view);
        button.setAttribute("aria-pressed", spec.view === "names" ? "true" : "false");
        button.addEventListener("click", function () {
          entries.forEach(function (item) { paintText(item.node, item.entry, spec.view); });
          bar.querySelectorAll("[data-feedback-view]").forEach(function (other) {
            other.setAttribute("aria-pressed",
              other.getAttribute("data-feedback-view") === spec.view ? "true" : "false");
          });
        });
        bar.appendChild(button);
      });

      block.parentNode.insertBefore(bar, block);
    }

    async function loadFeedback(submissionId) {
      if (!submissionId || state.feedback[submissionId] !== undefined) return;
      state.feedback[submissionId] = null;
      var block = document.querySelector('[data-feedback-for="' + submissionId + '"]');
      if (!block) return;
      var pane = block.closest(".result-pane");
      /* The toggle is mounted into the pane; without one we would rewrite the
         feedback nodes and then throw before the "what the AI saw" control
         exists, stranding the page in the re-substituted view. */
      if (!pane) return;
      var res = await call(url(gUrls.display, submissionId), "GET");
      if (!res.ok) return;

      var data = res.data || {};
      var summary = data.summary_feedback || {};
      var replacements = (data.replacements || summary.replacements || [])
        .filter(function (r) { return r && r.code && r.name; });
      if (!replacements.length) return;      // nothing was ever swapped in this text

      var entries = [{
        node: block,
        entry: {
          raw: summary.raw || block.textContent,
          display: summary.display || "",
          replacements: summary.replacements || replacements
        }
      }];

      (data.criteria || []).forEach(function (crit) {
        if (!crit || !crit.key) return;
        var row = pane && pane.querySelector('[data-crit-key="' + crit.key + '"] .crit-comment');
        if (!row) return;
        entries.push({
          node: row,
          entry: {
            raw: crit.raw || row.textContent,
            display: crit.display || "",
            replacements: crit.replacements || []
          }
        });
      });

      state.feedback[submissionId] = entries;
      entries.forEach(function (item) { paintText(item.node, item.entry, "names"); });
      feedbackToggle(pane, entries, replacements.length);

      var editor = pane && pane.querySelector(".feedback-editor textarea");
      if (editor && !editor.getAttribute("data-privacy-hinted")) {
        editor.setAttribute("data-privacy-hinted", "true");
        editor.parentNode.appendChild(el("p", "hint",
          "You are editing the text as the model wrote it — codes like " +
          replacements[0].code + " read as real names above."));
      }
    }

    /* -- wiring -------------------------------------------------------------- */

    function visiblePaneId() {
      var pane = document.querySelector(".result-pane:not([hidden])");
      return pane ? pane.getAttribute("data-pane-for") : null;
    }

    function hydrateVisible() {
      var id = visiblePaneId();
      if (!id) return;
      loadScan(id);
      loadFeedback(id);
    }

    document.addEventListener("click", function (event) {
      // grading.js swaps the pane on this same click; hydrate just after it.
      if (event.target.closest(".queue-row")) window.setTimeout(hydrateVisible, 0);
    });

    (async function bootGrading() {
      var res = await call(gUrls.mode, "GET");
      state.mode = res.ok ? readMode(res.data) : null;
      renderBanner();
      hydrateVisible();
    })();
  }
})();
