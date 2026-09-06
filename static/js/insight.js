/* insight.js — the Student Insight layer UI (Increment 1, Feature A).
 *
 * Two mounts, both optional; the file no-ops on pages that lack them.
 *
 *   #student-card    (templates/student_detail.html)
 *     data-api-card          GET  → the StudentCard
 *     data-api-refresh       POST → rebuild it, returns the new card
 *     data-api-observations  GET  → the observation feed
 *   #nudges-panel    (templates/course_detail.html)
 *     data-api-nudges        GET  → the course's open nudges
 *     data-api-dismiss       POST, "{id}" = nudge id
 *
 * Endpoint contract (docs/INCREMENT_1.md § API + UI). Shapes are read
 * defensively — a bare array, or {card|observations|nudges|items: [...]},
 * both work — so a small amount of backend drift degrades instead of breaking:
 *
 *   card         {summary, trajectory, model, updated_at,
 *                 strengths:  [{text, evidence: [observation_id]}],
 *                 weaknesses: [{text, evidence: [observation_id]}],
 *                 misconception_state: [{tag, status, evidence: [...]}]}
 *   observation  {id, kind, text, data, created_at, assignment_id, submission_id}
 *   nudge        {id, text, assignment_id, created_at,
 *                 evidence: {tag, student_ids, observation_ids}}
 *
 * Every claim in the card renders with links to the graded submission that
 * evidences it (SPEC: no unsourced assertions about a student). All text is
 * written with textContent — model-authored prose is untrusted.
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

  /* Non-throwing fetch: callers branch on .ok instead of try/catch, because
     "the router is not mounted yet" has to look like an empty state. */
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

  function listOf(data, key) {
    if (Array.isArray(data)) return data;
    if (!data || typeof data !== "object") return [];
    if (Array.isArray(data[key])) return data[key];
    if (Array.isArray(data.items)) return data.items;
    return [];
  }

  function cardOf(data) {
    if (!data || typeof data !== "object") return null;
    if (data.card && typeof data.card === "object") return data.card;
    return data;
  }

  function readJson(id) {
    var node = document.getElementById(id);
    if (!node) return null;
    try { return JSON.parse(node.textContent); } catch (err) { return null; }
  }

  function shortDate(value) {
    if (!value) return "";
    var when = new Date(value);
    if (isNaN(when.getTime())) return String(value);
    return when.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }

  /* ====================================================================== */
  /* Student card                                                           */
  /* ====================================================================== */

  var cardRoot = document.getElementById("student-card");
  var feedRoot = document.getElementById("observation-feed");

  if (cardRoot) {
    var ctx = readJson("student-context") || {};
    var gradingHref = ctx.grading_href || "/grading/{id}";
    var submissionIndex = {};
    (ctx.submissions || []).forEach(function (s) { submissionIndex[String(s.id)] = s; });

    var urls = {
      card: cardRoot.getAttribute("data-api-card"),
      refresh: cardRoot.getAttribute("data-api-refresh"),
      observations: cardRoot.getAttribute("data-api-observations")
    };

    var observationIndex = {};
    var cardBody = cardRoot.querySelector("[data-card-body]");
    var cardMeta = cardRoot.querySelector("[data-card-meta]");
    var cardError = cardRoot.querySelector("[data-card-error]");
    var refreshBtn = cardRoot.querySelector("[data-action='refresh-card']");

    var MIS_STATUS = {
      active: { glyph: "▲", note: "Still showing up in the last two graded assignments." },
      resolving: { glyph: "◐", note: "Absent from the most recent assignment, present before it." },
      resolved: { glyph: "✓", note: "Absent from the last two graded assignments." }
    };

    var OBS_KIND = {
      criterion_low: { label: "low criterion", cls: "obs-kind--low", glyph: "▼" },
      criterion_high: { label: "full marks", cls: "obs-kind--high", glyph: "▲" },
      misconception: { label: "misconception", cls: "obs-kind--mis", glyph: "◆" },
      strength: { label: "strength", cls: "obs-kind--strength", glyph: "✓" }
    };

    /* One evidence id → a link to the graded submission it came from. */
    function evidenceLink(obsId) {
      var obs = observationIndex[String(obsId)];
      var anchor = el("a", "ev-link");
      var submission = obs && obs.submission_id !== undefined && obs.submission_id !== null
        ? submissionIndex[String(obs.submission_id)]
        : null;

      if (submission) {
        // straight to the graded result this claim came from
        anchor.href = gradingHref.replace("{id}", String(submission.assignment_id)) +
          "?submission=" + encodeURIComponent(submission.id);
        anchor.textContent = submission.assignment_name || "the graded work";
      } else {
        anchor.href = "#obs-" + encodeURIComponent(obsId);
        anchor.textContent = obs && obs.assignment_name ? obs.assignment_name : "evidence";
      }
      if (obs && obs.text) anchor.title = obs.text;
      return anchor;
    }

    function evidenceRow(ids) {
      var wrap = el("span", "ev-links");
      var list = Array.isArray(ids) ? ids : [];
      if (!list.length) {
        wrap.appendChild(el("span", "fine", "no linked evidence"));
        return wrap;
      }
      wrap.appendChild(el("span", "ev-label", "from"));
      list.forEach(function (id, index) {
        if (index) wrap.appendChild(document.createTextNode(" · "));
        wrap.appendChild(evidenceLink(id));
      });
      return wrap;
    }

    function claimList(entries, emptyText) {
      var list = el("ul", "ev-list");
      if (!entries || !entries.length) {
        list.appendChild(el("li", "ev-empty", emptyText));
        return list;
      }
      entries.forEach(function (entry) {
        var item = el("li");
        item.appendChild(el("span", "ev-text", entry && entry.text ? entry.text : String(entry)));
        item.appendChild(evidenceRow(entry && entry.evidence));
        list.appendChild(item);
      });
      return list;
    }

    function misconceptionChips(entries) {
      var wrap = el("div", "mis-chips");
      if (!entries || !entries.length) {
        wrap.appendChild(el("p", "fine", "No misconception has been tagged for this student yet."));
        return wrap;
      }
      entries.forEach(function (entry) {
        var status = String(entry.status || "active").toLowerCase();
        var meta = MIS_STATUS[status] || MIS_STATUS.active;
        var chip = el("span", "mis-chip mis-chip--" + status);
        chip.appendChild(el("span", "mis-tag", entry.tag || "untagged"));
        var state = el("span", "mis-state", meta.glyph + " " + status);
        state.title = meta.note;
        chip.appendChild(state);
        chip.appendChild(evidenceRow(entry.evidence));
        wrap.appendChild(chip);
      });
      return wrap;
    }

    function renderEmptyCard(message) {
      cardBody.replaceChildren();
      var empty = el("div", "empty");
      empty.appendChild(el("h3", null, "No card yet"));
      empty.appendChild(el("p", null, message));
      cardBody.appendChild(empty);
      if (cardMeta) cardMeta.textContent = "";
    }

    function renderCard(card) {
      if (!card || (!card.summary && !card.trajectory &&
                    !(card.strengths || []).length && !(card.weaknesses || []).length)) {
        renderEmptyCard("Once this student's work is graded, Agora keeps a short, " +
          "evidence-linked read on where they are. You can also build it now.");
        return;
      }

      cardBody.replaceChildren();

      if (card.summary) cardBody.appendChild(el("p", "card-summary", card.summary));
      if (card.trajectory) {
        var traj = el("p", "card-trajectory");
        traj.appendChild(el("span", "card-trajectory-label", "Trajectory"));
        traj.appendChild(document.createTextNode(card.trajectory));
        cardBody.appendChild(traj);
      }

      var columns = el("div", "grid grid--2 card-claims");
      var left = el("div");
      left.appendChild(el("h3", null, "Strengths"));
      left.appendChild(claimList(card.strengths, "Nothing consistent enough to call a strength yet."));
      var right = el("div");
      right.appendChild(el("h3", null, "Where it slips"));
      right.appendChild(claimList(card.weaknesses, "No criterion is repeatedly below half marks — see Misconceptions below."));
      columns.appendChild(left);
      columns.appendChild(right);
      cardBody.appendChild(columns);

      var misHead = el("h3", "card-mis-head", "Misconceptions");
      cardBody.appendChild(misHead);
      cardBody.appendChild(misconceptionChips(card.misconception_state || card.misconceptions));

      if (cardMeta) {
        cardMeta.replaceChildren();
        var bits = [];
        if (card.updated_at) bits.push("updated " + shortDate(card.updated_at));
        if (card.model) bits.push(/^(template|mock-)/.test(card.model) ? "written from the record" : "written by " + card.model);
        if (bits.length) cardMeta.appendChild(el("span", "model-badge", bits.join(" · ")));
      }
    }

    function renderObservations(list) {
      if (!feedRoot) return;
      feedRoot.replaceChildren();
      if (!list.length) {
        var empty = el("div", "empty");
        empty.appendChild(el("h3", null, "No observations yet"));
        empty.appendChild(el("p", null,
          "Every graded submission adds a few plain facts here — a criterion missed, " +
          "a criterion aced, a tag that keeps coming back."));
        feedRoot.appendChild(empty);
        return;
      }

      var feed = el("ol", "obs-feed");
      list.forEach(function (obs) {
        var kind = OBS_KIND[String(obs.kind)] || { label: String(obs.kind || "note"), cls: "", glyph: "•" };
        var row = el("li", "obs-row " + kind.cls);
        row.id = "obs-" + obs.id;

        row.appendChild(el("span", "obs-kind " + kind.cls, kind.glyph + " " + kind.label));
        row.appendChild(el("p", "obs-text", obs.text || ""));

        var meta = el("p", "obs-meta");
        var submission = obs.submission_id !== undefined && obs.submission_id !== null
          ? submissionIndex[String(obs.submission_id)]
          : null;
        if (submission) {
          var link = el("a", null, submission.assignment_name || "the graded submission");
          link.href = gradingHref.replace("{id}", String(submission.assignment_id)) +
            "?submission=" + encodeURIComponent(submission.id);
          meta.appendChild(link);
        } else if (obs.assignment_name) {
          meta.appendChild(el("span", null, obs.assignment_name));
        }
        var detail = [];
        if (obs.data && obs.data.criterion_key) {
          detail.push(String(obs.data.criterion_key).replace(/_/g, " "));
        }
        if (obs.data && obs.data.score !== undefined && obs.data.max_points !== undefined) {
          detail.push(obs.data.score + "/" + obs.data.max_points);
        }
        if (obs.data && obs.data.tag) detail.push(obs.data.tag);
        if (obs.created_at) detail.push(shortDate(obs.created_at));
        if (detail.length) {
          if (meta.childNodes.length) meta.appendChild(document.createTextNode(" · "));
          meta.appendChild(document.createTextNode(detail.join(" · ")));
        }
        if (meta.childNodes.length) row.appendChild(meta);

        feed.appendChild(row);
      });
      var obsRows = feed.children;
      if (obsRows.length > 5) {
        var obsHidden = obsRows.length - 5;
        for (var oi = 5; oi < obsRows.length; oi++) { obsRows[oi].hidden = true; }
        var obsMore = el("li", "obs-row obs-row--more");
        var obsBtn = el("button", "btn btn--quiet btn--sm", "Show " + obsHidden + " more");
        obsBtn.type = "button";
        obsBtn.addEventListener("click", function () {
          for (var oj = 5; oj < feed.children.length; oj++) { feed.children[oj].hidden = false; }
          obsMore.remove();
        });
        obsMore.appendChild(obsBtn);
        feed.appendChild(obsMore);
      }
      feedRoot.appendChild(feed);
    }

    function showError(text) {
      if (!cardError) return;
      cardError.textContent = text;
      cardError.hidden = !text;
    }

    async function loadObservations() {
      if (!urls.observations) return [];
      var res = await call(urls.observations, "GET");
      if (!res.ok) return [];
      var list = listOf(res.data, "observations");
      observationIndex = {};
      list.forEach(function (obs) { observationIndex[String(obs.id)] = obs; });
      list = list.slice().sort(function (a, b) {
        return String(b.created_at || "").localeCompare(String(a.created_at || "")) ||
          (Number(b.id) - Number(a.id));
      });
      renderObservations(list);
      return list;
    }

    async function loadCard() {
      if (!urls.card) return;
      var res = await call(urls.card, "GET");
      if (!res.ok) {
        if (res.status === 404 || res.status === 0) {
          renderEmptyCard("Nothing has been consolidated for this student yet.");
        } else {
          renderEmptyCard(res.error);
        }
        return;
      }
      renderCard(cardOf(res.data));
    }

    if (refreshBtn && urls.refresh) {
      refreshBtn.addEventListener("click", async function () {
        var label = refreshBtn.textContent;
        refreshBtn.disabled = true;
        refreshBtn.classList.add("is-busy");
        refreshBtn.textContent = "Refreshing…";
        showError("");
        announce("Rebuilding the student card.");
        var res = await call(urls.refresh, "POST", {});
        refreshBtn.disabled = false;
        refreshBtn.classList.remove("is-busy");
        refreshBtn.textContent = label;
        if (!res.ok) {
          showError("Could not rebuild the card: " + res.error);
          return;
        }
        await loadObservations();
        renderCard(cardOf(res.data));
        announce("Student card updated.");
      });
    }

    (async function boot() {
      await loadObservations();     // first: the card's evidence links need the index
      await loadCard();
    })();
  }

  /* ====================================================================== */
  /* "Worth a look" — course nudges                                         */
  /* ====================================================================== */

  var nudgeRoot = document.getElementById("nudges-panel");
  if (nudgeRoot) {
    var nudgeUrls = {
      list: nudgeRoot.getAttribute("data-api-nudges"),
      dismiss: nudgeRoot.getAttribute("data-api-dismiss") || "/api/nudges/{id}/dismiss"
    };
    var nudgeBody = nudgeRoot.querySelector("[data-nudge-body]");
    var nudgeCount = nudgeRoot.querySelector("[data-nudge-count]");
    var courseCtx = readJson("course-context") || {};
    var nudgeGradingHref = courseCtx.grading_href || "/grading/{id}";
    var assignmentIndex = {};
    (courseCtx.assignments || []).forEach(function (a) { assignmentIndex[String(a.id)] = a; });
    var studentIndex = {};
    (courseCtx.students || []).forEach(function (s) { studentIndex[String(s.id)] = s; });

    function nudgeEmpty() {
      nudgeBody.replaceChildren();
      var empty = el("div", "empty");
      empty.appendChild(el("h3", null, "Nothing worth flagging"));
      empty.appendChild(el("p", null,
        "When three or more students miss the same thing on the same assignment, " +
        "it shows up here with the work that proves it."));
      nudgeBody.appendChild(empty);
      if (nudgeCount) nudgeCount.textContent = "";
    }

    function renderNudges(list) {
      var open = list.filter(function (n) { return !n.dismissed_at; });
      if (!open.length) { nudgeEmpty(); return; }

      nudgeBody.replaceChildren();
      if (nudgeCount) {
        nudgeCount.textContent = open.length + " signal" + (open.length === 1 ? "" : "s");
      }

      var ul = el("ul", "nudge-list");
      open.forEach(function (nudge) {
        var item = el("li", "nudge");
        item.setAttribute("data-nudge-id", nudge.id);

        var ev0 = nudge.evidence || {};
        var nCount = Array.isArray(ev0.student_ids) ? ev0.student_ids.length : 0;
        var aName = (nudge.assignment_id !== undefined && nudge.assignment_id !== null
          && assignmentIndex[String(nudge.assignment_id)])
          ? assignmentIndex[String(nudge.assignment_id)].name : "";
        var line = el("p", "nudge-text");
        line.appendChild(el("strong", null, nCount + " student" + (nCount === 1 ? "" : "s")));
        if (aName) { line.appendChild(document.createTextNode(" · " + aName)); }
        item.appendChild(line);

        var evidence = nudge.evidence || {};
        var meta = el("p", "nudge-ev");
        if (evidence.tag) meta.appendChild(el("span", "tag", evidence.tag));


        var ids = Array.isArray(evidence.student_ids) ? evidence.student_ids : [];
        ids.forEach(function (sid) {
          var student = studentIndex[String(sid)];
          var slink = el("a", "snum");
          slink.href = "/students/" + encodeURIComponent(sid);
          slink.textContent = student
            ? "#" + String(student.number).padStart(2, "0")
            : "student " + sid;
          if (student) slink.title = student.name;
          meta.appendChild(slink);
        });
        if (meta.childNodes.length) item.appendChild(meta);

        var dismiss = el("button", "btn btn--quiet btn--sm nudge-dismiss", "Dismiss");
        dismiss.type = "button";
        dismiss.setAttribute("data-action", "dismiss-nudge");
        item.appendChild(dismiss);

        ul.appendChild(item);
      });
      var rows = ul.children;
      if (rows.length > 3) {
        var hiddenCount = rows.length - 3;
        for (var ri = 3; ri < rows.length; ri++) { rows[ri].hidden = true; }
        var moreItem = el("li", "nudge nudge--more");
        var moreBtn = el("button", "btn btn--quiet btn--sm", "Show " + hiddenCount + " more");
        moreBtn.type = "button";
        moreBtn.addEventListener("click", function () {
          for (var rj = 3; rj < ul.children.length; rj++) { ul.children[rj].hidden = false; }
          moreItem.remove();
        });
        moreItem.appendChild(moreBtn);
        ul.appendChild(moreItem);
      }
      nudgeBody.appendChild(ul);
    }

    nudgeBody.addEventListener("click", async function (event) {
      var button = event.target.closest("[data-action='dismiss-nudge']");
      if (!button) return;
      var item = button.closest(".nudge");
      var id = item && item.getAttribute("data-nudge-id");
      if (!id) return;
      button.disabled = true;
      button.textContent = "Dismissing…";
      var res = await call(nudgeUrls.dismiss.replace("{id}", encodeURIComponent(id)), "POST", {});
      if (!res.ok) {
        button.disabled = false;
        button.textContent = "Dismiss";
        var err = item.querySelector(".nudge-error") || el("p", "nudge-error form-error");
        err.textContent = "Could not dismiss that: " + res.error;
        err.hidden = false;
        item.appendChild(err);
        return;
      }
      item.remove();
      announce("Nudge dismissed.");
      if (!nudgeBody.querySelector(".nudge")) nudgeEmpty();
      else if (nudgeCount) {
        var left = nudgeBody.querySelectorAll(".nudge").length;
        nudgeCount.textContent = left + " signal" + (left === 1 ? "" : "s");
      }
    });

    (async function bootNudges() {
      if (!nudgeUrls.list) { nudgeRoot.hidden = true; return; }
      var res = await call(nudgeUrls.list, "GET");
      if (!res.ok) { nudgeRoot.hidden = true; return; }   // insight router not mounted
      nudgeRoot.hidden = false;
      renderNudges(listOf(res.data, "nudges"));
    })();
  }
})();
