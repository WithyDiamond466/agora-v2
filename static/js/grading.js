/* grading.js — the grading queue: upload + student mapping, batch grading with
 * per-row status polling, retries, and the result pane swap.
 *
 * Endpoints come from #grading-root data attributes so the grading router can
 * override them (defaults in templates/grading.html):
 *   data-api-list       GET  → [{id, status, overall_score, max_score, error}]
 *                              (an object with a "submissions" array also works)
 *   data-api-upload     POST multipart: files[] + student_ids[] (parallel) and
 *                              a "mapping" JSON field of [{filename, student_id}]
 *   data-api-grade-all  POST {submission_ids: [...]}   (empty = everything pending)
 *   data-api-grade-one  POST, "{id}" = submission id
 *
 * The page renders every result pane server-side; JS only shows the right one,
 * so selection works without JavaScript too (the row name is a real link).
 */
(function () {
  "use strict";

  var root = document.getElementById("grading-root");
  if (!root) return;

  var queue = document.getElementById("grading-queue");
  var panes = document.getElementById("grading-panes");
  var progressBox = document.getElementById("grading-progress");
  var progressBar = progressBox && progressBox.querySelector(".progress");
  var progressFill = progressBox && progressBox.querySelector(".progress > span");
  var progressNote = progressBox && progressBox.querySelector(".progress-note");
  var live = document.getElementById("live-region");

  var urls = {
    list: root.getAttribute("data-api-list"),
    upload: root.getAttribute("data-api-upload"),
    gradeAll: root.getAttribute("data-api-grade-all"),
    gradeOne: root.getAttribute("data-api-grade-one"),
    mapping: root.getAttribute("data-api-mapping"),
    release: root.getAttribute("data-api-release"),
    exportUrl: root.getAttribute("data-api-export"),
    seen: root.getAttribute("data-api-seen"),
    approve: root.getAttribute("data-api-approve"),
    withhold: root.getAttribute("data-api-withhold")
  };

  var roster = [];
  try {
    var rosterNode = document.getElementById("roster-json");
    if (rosterNode) roster = JSON.parse(rosterNode.textContent) || [];
  } catch (e) { roster = []; }

  var pollTimer = null;
  var pending = [];   // [{file, studentId, confidence, reason}]

  /* ---- tiny helpers -------------------------------------------------------- */

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function announce(text) { if (live) live.textContent = text; }

  async function send(url, method, body, multipart) {
    var init = { method: method, headers: { Accept: "application/json" } };
    if (body !== undefined && body !== null) {
      if (multipart) {
        init.body = body;
      } else {
        init.headers["Content-Type"] = "application/json";
        init.body = JSON.stringify(body);
      }
    }
    var response = await fetch(url, init);
    var data = null;
    try { data = await response.json(); } catch (e) { data = null; }
    if (!response.ok) {
      var detail = data && data.detail;
      if (Array.isArray(detail)) detail = detail.map(function (d) { return d.msg; }).join("; ");
      throw new Error(detail || response.status + " " + response.statusText);
    }
    return data;
  }

  /* ---- human review record --------------------------------------------------- */

  function setReviewState(id, state) {
    var pane = panes && panes.querySelector('.result-pane[data-pane-for="' + id + '"]');
    var row = rowFor(id);
    [pane, row].forEach(function (node) {
      if (node) node.setAttribute("data-review", state);
    });
    var pill = pane && pane.querySelector("[data-review-pill]");
    if (pill) { pill.className = "pill pill--" + state; pill.textContent = state; }
    var rowPill = row && row.querySelector("[data-cell='pill'] .pill");
    if (rowPill && row.getAttribute("data-status") === "graded") {
      rowPill.className = "pill pill--" + state;
      rowPill.textContent = state;
    }
  }

  /* Record opening separately from explicit approval. */
  var seenInFlight = {};
  function markSeen(id) {
    if (!urls.seen || !panes) return;
    var pane = panes.querySelector('.result-pane[data-pane-for="' + id + '"]');
    if (!pane || pane.getAttribute("data-review") !== "unreviewed") return;
    if (seenInFlight[id]) return;
    seenInFlight[id] = true;
    send(urls.seen.replace("{id}", String(id)), "POST", {}, false).then(function (data) {
      var state = data && data.review && data.review.state;
      if (state) setReviewState(id, state);
    }).catch(function () {
      seenInFlight[id] = false;   // try again on the next open
    });
  }

  /* ---- result pane selection ----------------------------------------------- */

  function select(id) {
    if (!panes) return;
    panes.querySelectorAll(".result-pane").forEach(function (pane) {
      pane.hidden = pane.getAttribute("data-pane-for") !== String(id);
    });
    markSeen(id);
    if (queue) {
      queue.querySelectorAll(".queue-row").forEach(function (row) {
        if (row.getAttribute("data-submission-id") === String(id)) {
          row.setAttribute("aria-selected", "true");
        } else {
          row.removeAttribute("aria-selected");
        }
      });
    }
    try {
      var url = new URL(window.location.href);
      url.searchParams.set("submission", id);
      window.history.replaceState({}, "", url);
    } catch (e) { /* ignore */ }
  }

  if (queue) {
    queue.addEventListener("click", function (event) {
      if (event.target.closest("input[type='checkbox']")) return;
      if (event.target.closest("[data-action='retry']")) return;
      var row = event.target.closest(".queue-row");
      if (!row) return;
      event.preventDefault();               // the name link is the no-JS fallback
      select(row.getAttribute("data-submission-id"));
    });
    queue.addEventListener("keydown", function (event) {
      var row = event.target.closest(".queue-row");
      if (!row) return;
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select(row.getAttribute("data-submission-id"));
      }
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        var rows = Array.prototype.slice.call(queue.querySelectorAll(".queue-row"));
        var next = rows[rows.indexOf(row) + (event.key === "ArrowDown" ? 1 : -1)];
        if (next) { next.focus(); select(next.getAttribute("data-submission-id")); }
      }
    });
  }

  /* ---- filename → student heuristics ---------------------------------------- */

  function tokens(name) {
    return name
      .replace(/\.[a-z0-9]+$/i, "")
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter(Boolean);
  }

  function guess(filename) {
    var parts = tokens(filename);
    var best = { student: null, score: 0, reason: "" };

    roster.forEach(function (student) {
      var nameParts = String(student.name).toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
      var score = 0;
      var hits = [];

      nameParts.forEach(function (piece) {
        if (piece.length < 3) return;
        if (parts.indexOf(piece) !== -1) { score += 3; hits.push(piece); }
        else if (parts.some(function (p) { return p.length > 3 && (p.indexOf(piece) === 0 || piece.indexOf(p) === 0); })) {
          score += 1;
          hits.push(piece);
        }
      });

      parts.forEach(function (piece, index) {
        var value = parseInt(piece, 10);
        if (!isNaN(value) && String(value) === String(parseInt(piece, 10)) && value === student.number) {
          // "…_03" right after a "student"/"snum" marker is a strong signal
          var marker = index > 0 && /student|stu|num|no|id|#/.test(parts[index - 1]);
          score += marker ? 3 : 2;
          hits.push("#" + student.number);
        }
      });

      if (score > best.score) best = { student: student, score: score, reason: hits.join(" + ") };
    });

    return best;
  }

  function studentSelect(selectedId) {
    var select = el("select");
    var none = el("option", null, "— unassigned —");
    none.value = "";
    select.appendChild(none);
    roster.forEach(function (student) {
      var option = el("option", null, "#" + student.number + "  " + student.name);
      option.value = student.id;
      if (selectedId && Number(selectedId) === Number(student.id)) option.selected = true;
      select.appendChild(option);
    });
    return select;
  }

  function renderMapping() {
    var panel = document.getElementById("upload-mapping");
    var table = document.getElementById("upload-mapping-table");
    if (!panel || !table) return;
    var body = table.querySelector("tbody");
    body.replaceChildren();

    pending.forEach(function (entry, index) {
      var row = el("tr");
      row.appendChild(el("td", "mono", entry.file.name));

      var cell = el("td");
      var select = studentSelect(entry.studentId);
      select.addEventListener("change", function () {
        pending[index].studentId = select.value ? Number(select.value) : null;
        pending[index].confidence = select.value ? "confirmed" : "none";
        renderMapping();
      });
      cell.appendChild(select);
      row.appendChild(cell);

      var match = el("td");
      if (entry.confidence === "high") {
        match.appendChild(el("span", "pill pill--graded", "matched"));
      } else if (entry.confidence === "confirmed") {
        match.appendChild(el("span", "pill pill--graded", "you chose"));
      } else if (entry.confidence === "low") {
        match.appendChild(el("span", "pill pill--warning", "check this"));
      } else {
        match.appendChild(el("span", "pill pill--failed", "no match"));
      }
      if (entry.reason) match.appendChild(el("span", "fine mapping-reason", entry.reason));
      row.appendChild(match);

      var actions = el("td");
      var drop = el("button", "btn btn--quiet btn--sm", "Remove");
      drop.type = "button";
      drop.addEventListener("click", function () {
        pending.splice(index, 1);
        renderMapping();
      });
      actions.appendChild(drop);
      row.appendChild(actions);

      body.appendChild(row);
    });

    panel.hidden = pending.length === 0;
  }

  function addFiles(files) {
    Array.prototype.forEach.call(files, function (file) {
      var already = pending.some(function (entry) {
        return entry.file.name === file.name && entry.file.size === file.size;
      });
      if (already) return;
      var best = guess(file.name);
      pending.push({
        file: file,
        studentId: best.student ? best.student.id : null,
        confidence: best.score >= 3 ? "high" : (best.score > 0 ? "low" : "none"),
        reason: best.reason
      });
    });
    renderMapping();
  }

  var dropzone = document.getElementById("upload-dropzone");
  var picker = document.getElementById("upload-input");
  if (dropzone && picker) {
    dropzone.addEventListener("click", function () { picker.click(); });
    dropzone.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); picker.click(); }
    });
    ["dragenter", "dragover"].forEach(function (name) {
      dropzone.addEventListener(name, function (event) {
        event.preventDefault();
        dropzone.classList.add("is-over");
      });
    });
    ["dragleave", "drop"].forEach(function (name) {
      dropzone.addEventListener(name, function () { dropzone.classList.remove("is-over"); });
    });
    dropzone.addEventListener("drop", function (event) {
      event.preventDefault();
      if (event.dataTransfer && event.dataTransfer.files) addFiles(event.dataTransfer.files);
    });
    picker.addEventListener("change", function () {
      addFiles(picker.files);
      picker.value = "";
    });
  }

  var confirmUpload = document.getElementById("upload-confirm");
  if (confirmUpload) {
    confirmUpload.addEventListener("click", async function () {
      var errorBox = document.getElementById("upload-error");
      if (errorBox) { errorBox.hidden = true; errorBox.textContent = ""; }
      if (!pending.length) return;

      var unmatched = pending.filter(function (entry) { return !entry.studentId; });
      if (unmatched.length && !window.confirm(
        unmatched.length + " file(s) have no student. Upload them unassigned?")) {
        return;
      }

      var body = new FormData();
      var mapping = [];
      pending.forEach(function (entry) {
        body.append("files", entry.file);
        body.append("student_ids", entry.studentId ? String(entry.studentId) : "");
        mapping.push({ filename: entry.file.name, student_id: entry.studentId });
      });
      body.append("mapping", JSON.stringify(mapping));

      confirmUpload.disabled = true;
      confirmUpload.textContent = "Uploading…";
      try {
        var result = await send(urls.upload, "POST", body, true);
        var notes = [];
        if (result && result.rejected && result.rejected.length) {
          notes.push(result.rejected.length + " file(s) rejected: " +
            result.rejected.map(function (r) { return (r.filename || "?") + " — " + (r.reason || "unsupported"); }).join("; "));
        }
        if (result && result.needs_confirmation) {
          notes.push(result.needs_confirmation + " file(s) still need a student — assign them from the queue.");
        }
        if (notes.length && errorBox) {
          errorBox.textContent = notes.join(" ");
          errorBox.hidden = false;
          confirmUpload.textContent = "Uploaded — refreshing…";
          window.setTimeout(function () { window.location.reload(); }, 2500);
          return;
        }
        window.location.reload();
      } catch (err) {
        confirmUpload.disabled = false;
        confirmUpload.textContent = "Upload and add to queue";
        if (errorBox) { errorBox.textContent = err.message; errorBox.hidden = false; }
        else window.alert(err.message);
      }
    });
  }

  /* ---- batch grading + polling ------------------------------------------------ */

  function rowFor(id) {
    return queue ? queue.querySelector('.queue-row[data-submission-id="' + id + '"]') : null;
  }

  function setRow(row, item) {
    var status = String(item.status || "").toLowerCase();
    if (!status) status = item.grade_result || item.overall_score !== undefined ? "graded" : "pending";
    row.setAttribute("data-status", status);

    var pillCell = row.querySelector("[data-cell='pill']");
    if (pillCell) {
      var label = status === "graded" ? (item.review_state || "unreviewed")
        : status === "grading" ? "grading"
        : status === "failed" ? "failed" : "pending";
      pillCell.replaceChildren(el("span", "pill pill--" + label, label));
      if (status === "graded") row.setAttribute("data-review", item.review_state || "unreviewed");
    }

    var scoreCell = row.querySelector("[data-cell='score']");
    if (scoreCell) {
      var score = item.overall_score !== undefined && item.overall_score !== null
        ? item.overall_score
        : (item.score !== undefined ? item.score : null);
      var max = item.max_score !== undefined ? item.max_score : null;
      if (item.incomplete) { score = "Incomplete"; max = null; }
      else if (item.mode === "feedback") { score = "Feedback only"; max = null; }
      if (score === null || score === undefined) {
        scoreCell.textContent = "—";
      } else {
        scoreCell.replaceChildren(document.createTextNode(String(Math.round(score * 10) / 10)));
        if (max) scoreCell.appendChild(el("span", "max", "/" + Math.round(max)));
      }
    }

    var retryCell = row.querySelector("[data-cell='retry']");
    if (retryCell) {
      retryCell.replaceChildren();
      if (status === "failed") {
        var retry = el("button", "q-retry", "Retry");
        retry.type = "button";
        retry.setAttribute("data-action", "retry");
        retryCell.appendChild(retry);
      }
    }
  }

  function showProgress(done, total, note) {
    if (!progressBox) return;
    progressBox.hidden = false;
    var pct = total ? Math.round((done / total) * 100) : 0;
    if (progressFill) progressFill.style.width = pct + "%";
    if (progressBar) {
      progressBar.setAttribute("aria-valuenow", String(done));
      progressBar.setAttribute("aria-valuemax", String(total));
    }
    if (progressNote) progressNote.textContent = note;
  }

  function items(data) {
    if (Array.isArray(data)) return data;
    if (data && Array.isArray(data.submissions)) return data.submissions;
    if (data && Array.isArray(data.items)) return data.items;
    return [];
  }

  async function poll() {
    if (!urls.list) return;
    var data;
    try {
      data = await send(urls.list, "GET", null, false);
    } catch (err) {
      showProgress(0, 0, "Lost contact with the server: " + err.message);
      stopPolling();
      return;
    }

    var list = items(data);
    var total = 0;
    var done = 0;
    var running = 0;
    var waiting = 0;
    var current = null;

    list.forEach(function (item) {
      var row = rowFor(item.id);
      if (row) setRow(row, item);
      var status = String(item.status || "").toLowerCase();
      total += 1;
      if (status === "graded" || status === "failed") done += 1;
      if (status === "pending") waiting += 1;
      if (status === "grading") { running += 1; if (!current) current = item; }
    });

    // The status endpoint already rolls these up; trust it when it is there.
    if (typeof data.total === "number") total = data.total;
    if (typeof data.done === "number") done = data.done;
    if (data.counts) {
      running = data.counts.grading || 0;
      waiting = data.counts.pending || 0;
    }

    if (running || waiting) {
      var label = current && current.student_number
        ? " · grading #" + current.student_number
        : (running ? " · " + running + " in flight" : "");
      showProgress(done, total, done + " of " + total + " graded" + label +
        " · failures retry from the row");
    }

    if (!running && !waiting) {
      stopPolling();
      showProgress(done, total, done + " of " + total + " graded · refreshing results…");
      announce("Grading finished.");
      window.setTimeout(function () { window.location.reload(); }, 700);
    }
  }

  function startPolling() {
    if (pollTimer) return;
    poll();
    pollTimer = window.setInterval(poll, 2000);
  }

  function stopPolling() {
    if (pollTimer) { window.clearInterval(pollTimer); pollTimer = null; }
  }

  window.addEventListener("beforeunload", stopPolling);

  async function gradeAll(button) {
    var checked = [];
    if (queue) {
      queue.querySelectorAll("[data-queue-check]:checked").forEach(function (box) {
        var row = box.closest(".queue-row");
        var status = row ? row.getAttribute("data-status") : "";
        if (status !== "graded") checked.push(Number(box.value));
      });
    }

    /* Privacy Guard seam (Increment 1): in warn mode privacy.js scans the
       batch, shows what it found and lets the professor decide before anything
       is sent. It resolves false to cancel the run. */
    var label = button.textContent;
    if (window.AgoraPrivacy && typeof window.AgoraPrivacy.confirmRun === "function") {
      var proceed = await window.AgoraPrivacy.confirmRun(checked);
      if (!proceed) return;
    }
    var body = { submission_ids: checked };

    button.disabled = true;
    button.textContent = "Grading…";
    showProgress(0, checked.length, "Starting the batch…");
    try {
      await send(urls.gradeAll, "POST", body, false);
      announce("Grading started.");
      startPolling();
    } catch (err) {
      button.disabled = false;
      button.textContent = label;
      showProgress(0, 0, "Could not start grading: " + err.message);
    }
  }

  async function gradeOne(id, trigger) {
    var url = (urls.gradeOne || "").replace("{id}", String(id));
    if (!url) return;

    /* Same Privacy Guard gate as gradeAll: in warn mode the professor decides
       per grading run, and grading a single row is a run. Without this the
       banner on this very page promises a check that never happens. */
    if (window.AgoraPrivacy && typeof window.AgoraPrivacy.confirmRun === "function") {
      var proceed = await window.AgoraPrivacy.confirmRun([Number(id)]);
      if (!proceed) return;
    }

    var row = rowFor(id);
    if (row) setRow(row, { id: id, status: "grading" });
    if (trigger) trigger.disabled = true;
    try {
      await send(url, "POST", {}, false);
      startPolling();
    } catch (err) {
      if (trigger) trigger.disabled = false;
      if (row) setRow(row, { id: id, status: "failed" });
      showProgress(0, 0, "Could not grade that submission: " + err.message);
    }
  }

  document.addEventListener("click", function (event) {
    var gradeAllBtn = event.target.closest("[data-action='grade-all']");
    if (gradeAllBtn) { gradeAll(gradeAllBtn); return; }

    var retry = event.target.closest("[data-action='retry']");
    if (retry) {
      var confirmText = retry.getAttribute("data-confirm");
      if (confirmText && !window.confirm(confirmText)) return;
      var row = retry.closest(".queue-row");
      var pane0 = retry.closest(".result-pane");
      var id = retry.getAttribute("data-submission-id") ||
        (row && row.getAttribute("data-submission-id")) ||
        (pane0 && pane0.getAttribute("data-pane-for"));
      if (id) gradeOne(id, retry);
      return;
    }

    var edit = event.target.closest("[data-action='edit-feedback']");
    if (edit) {
      var pane = edit.closest(".result-pane");
      var editor = pane && pane.querySelector(".feedback-editor");
      var block = pane && pane.querySelector(".feedback-block");
      if (editor) { editor.hidden = false; if (block) block.hidden = true; editor.querySelector("textarea").focus(); }
      return;
    }

    var cancel = event.target.closest("[data-action='cancel-feedback']");
    if (cancel) {
      var pane2 = cancel.closest(".result-pane");
      var editor2 = pane2 && pane2.querySelector(".feedback-editor");
      var block2 = pane2 && pane2.querySelector(".feedback-block");
      if (editor2) editor2.hidden = true;
      if (block2) block2.hidden = false;
      return;
    }

    var assign = event.target.closest("[data-action='assign-student']");
    if (assign) {
      var pane1 = assign.closest(".result-pane");
      var chooser = pane1 && pane1.querySelector("[data-assign-select]");
      var studentId = chooser && chooser.value;
      if (!studentId) { window.alert("Pick a student first."); return; }
      assign.disabled = true;
      send(urls.mapping, "POST", {
        mapping: [{
          submission_id: Number(assign.getAttribute("data-submission-id")),
          student_id: Number(studentId)
        }]
      }, false).then(function () {
        window.location.reload();
      }).catch(function (err) {
        assign.disabled = false;
        window.alert("Could not assign that student: " + err.message);
      });
      return;
    }

    var next = event.target.closest("[data-action='approve-next']");
    if (next && queue) {
      var current = queue.querySelector('.queue-row[aria-selected="true"]');
      if (!current) return;
      var currentId = current.getAttribute("data-submission-id");
      var currentPane = panes.querySelector('.result-pane[data-pane-for="' + currentId + '"]');
      if (currentPane && currentPane.querySelector(".criterion-editor[open], .feedback-editor:not([hidden])")) {
        window.alert("Save or close your edits before approving this result.");
        return;
      }
      next.disabled = true;
      send(urls.approve.replace("{id}", currentId), "POST", {}, false).then(function (data) {
        setReviewState(currentId, data.review.state);
        var rows = Array.prototype.slice.call(queue.querySelectorAll(".queue-row"));
        var start = rows.indexOf(current) + 1;
        var ordered = rows.slice(start).concat(rows.slice(0, start));
        for (var i = 0; i < ordered.length; i++) {
          var state = ordered[i].getAttribute("data-review");
          if (ordered[i].getAttribute("data-status") === "graded" && (state === "seen" || state === "unreviewed")) {
            select(ordered[i].getAttribute("data-submission-id"));
            ordered[i].focus();
            return;
          }
        }
        announce("All available results are approved or withheld. Release approved feedback when ready.");
      }).catch(function (err) { window.alert(err.message); })
        .finally(function () { next.disabled = false; });
      return;
    }

    var hold = event.target.closest("[data-action='withhold']");
    if (hold) {
      var holdConfirm = hold.getAttribute("data-confirm");
      if (holdConfirm && !window.confirm(holdConfirm)) return;
      var holdId = hold.getAttribute("data-submission-id");
      var withheld = hold.getAttribute("data-withheld") !== "false";
      hold.disabled = true;
      send((urls.withhold || "").replace("{id}", String(holdId)), "POST", { withheld: withheld }, false)
        .then(function (data) {
          try { sessionStorage.setItem("agora-flash", (data && data.message) || "Saved."); } catch (e) {}
          window.location.reload();
        })
        .catch(function (err) {
          hold.disabled = false;
          window.alert("Could not update this result: " + err.message);
        });
      return;
    }

    var saveManual = event.target.closest("[data-action='save-manual-scores']");
    if (saveManual) {
      var paneM = saveManual.closest(".result-pane");
      var note = paneM && paneM.querySelector("[data-manual-note]");
      var edits = [];
      if (paneM) {
        var invalid = false;
        paneM.querySelectorAll("[data-criterion-edit]").forEach(function (field) {
          var input = field.querySelector("[data-crit-score]");
          var comment = field.querySelector("[data-crit-comment]");
          var entry = { key: field.getAttribute("data-crit-key"), comment: comment.value };
          if (input) {
            if (!input.reportValidity()) invalid = true;
            if (input.value !== "") entry.score = Number(input.value);
          }
          edits.push(entry);
        });
        if (invalid) return;
      }
      if (!edits.length) { if (note) note.textContent = "Enter a score first."; return; }
      saveManual.disabled = true;
      send(saveManual.getAttribute("data-result-api"), "PATCH", { criteria: edits }, false)
        .then(function () {
          try { sessionStorage.setItem("agora-flash", "Rubric edits saved."); } catch (e) {}
          window.location.reload();
        })
        .catch(function (err) {
          saveManual.disabled = false;
          if (note) note.textContent = err.message;
        });
      return;
    }

    var releaseBtn = event.target.closest("[data-action='release']");
    if (releaseBtn) { openRelease(); return; }
  });

  /* ---- release dialog ---------------------------------------------------------- */

  var dialog = document.getElementById("release-dialog");

  function q(sel) { return dialog ? dialog.querySelector(sel) : null; }

  function personLabel(row) {
    var num = row.student_number !== null && row.student_number !== undefined
      ? "#" + (row.student_number < 10 ? "0" : "") + row.student_number + " " : "";
    return num + (row.student_name || "Unassigned");
  }

  async function openRelease() {
    if (!dialog || !urls.release) return;
    var intro = q("[data-release-intro]");
    var list = q("[data-release-list]");
    var unseenRow = q("[data-release-unseen-row]");
    var unseenCount = q("[data-release-unseen-count]");
    var unseenBox = q("[data-release-unseen]");
    var manual = q("[data-release-manual]");
    var termsNote = q("[data-release-terms]");
    var error = q("[data-release-error]");
    var confirmBtn = q("[data-release-confirm]");

    if (error) { error.hidden = true; error.textContent = ""; }
    if (list) list.replaceChildren();
    if (intro) intro.textContent = "Checking what is ready…";
    if (unseenRow) unseenRow.hidden = true;
    if (unseenBox) unseenBox.checked = false;
    if (manual) manual.hidden = true;
    if (termsNote) termsNote.hidden = true;
    if (confirmBtn) confirmBtn.disabled = true;
    if (typeof dialog.showModal === "function" && !dialog.open) dialog.showModal();

    var summary;
    try {
      summary = await send(urls.release, "GET", null, false);
    } catch (err) {
      if (intro) intro.textContent = "Could not load the release summary: " + err.message;
      return;
    }
    var counts = summary.counts || {};
    var ready = (summary.releasable || []).length;
    var unseen = (summary.unseen || []).length;
    var accepted = summary.terms && summary.terms.accepted;

    if (intro) {
      if (!counts.graded) intro.textContent = "Nothing is graded yet.";
      else if (!ready && !unseen) intro.textContent = counts.released + " of " + counts.graded +
        " already released" + (counts.withheld ? ", " + counts.withheld + " withheld" : "") + ". Nothing new to release.";
      else intro.textContent = "Release " + ready + " approved result" + (ready === 1 ? "" : "s") +
        " to students" + (counts.released ? " (" + counts.released + " already released)" : "") +
        (counts.withheld ? ", keeping " + counts.withheld + " withheld" : "") + ".";
    }
    if (list) {
      (summary.releasable || []).forEach(function (row) {
        var item = el("li", null, personLabel(row));
        if (row.needs_manual_scores) item.appendChild(el("span", "fine", " · " + row.needs_manual_scores + " criterion still unscored"));
        list.appendChild(item);
      });
    }
    if (unseenRow && unseen) {
      unseenRow.hidden = false;
      if (unseenCount) unseenCount.textContent = String(unseen);
    }
    if (manual && summary.needs_manual_scores) {
      manual.hidden = false;
      manual.textContent = summary.needs_manual_scores + " criterion score" +
        (summary.needs_manual_scores === 1 ? " is" : "s are") + " still yours to fill in; incomplete results cannot be released.";
    }
    if (termsNote && !accepted) termsNote.hidden = false;
    if (confirmBtn) confirmBtn.disabled = !accepted || !ready;
  }

  if (dialog) {
    var cancel = q("[data-release-cancel]");
    if (cancel) cancel.addEventListener("click", function () { dialog.close(); });
    var confirmRelease = q("[data-release-confirm]");
    if (confirmRelease) {
      confirmRelease.addEventListener("click", async function () {
        var unseenBox = q("[data-release-unseen]");
        var error = q("[data-release-error]");
        confirmRelease.disabled = true;
        try {
          var data = await send(urls.release, "POST", { include_unseen: !!(unseenBox && unseenBox.checked) }, false);
          try { sessionStorage.setItem("agora-flash", (data && data.message) || "Released."); } catch (e) {}
          window.location.reload();
        } catch (err) {
          confirmRelease.disabled = false;
          if (error) { error.textContent = err.message; error.hidden = false; }
        }
      });
    }
  }

  // The initially selected pane counts as opened too.
  if (queue) {
    var initial = queue.querySelector('.queue-row[aria-selected="true"]');
    if (initial) markSeen(initial.getAttribute("data-submission-id"));
  }

  // A batch may already be running when the page loads.
  if (queue && queue.querySelector('.queue-row[data-status="grading"]')) startPolling();
})();
