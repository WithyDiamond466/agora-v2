/* chat.js — the floating assistant panel (and the skill "test drive" mini chat).
 *
 * Wire protocol (POST to the panel's data-chat-endpoint, default /api/chat):
 *   request  {message, session_id, context: {page, course_id, assignment_id,
 *             student_id, skill_id}}
 *   response {reply|message|text|content, session_id,
 *             tool_calls: [{name|tool, input|arguments}],
 *             actions|action: [...]}
 *
 * Actions:
 *   {type: "navigate"|"navigate_to", url|path|page, course_id?, assignment_id?,
 *    student_id?}                     → location change, no confirmation
 *   {type: "confirm_grading"|"start_grading", assignment_id, assignment_name?,
 *    course_name?, skill_name?, rubric_name?, submission_count?}
 *                                     → renders a .chat-confirm card; the
 *                                       assistant opens the grading queue and
 *                                       never starts a run.
 *
 * Every piece of model/tool text is inserted with textContent: chat content is
 * untrusted, and tool lines are shown, never hidden.
 */
(function () {
  "use strict";

  var OPEN_KEY = "agora-chat-open";
  var SESSION_KEY = "agora-chat-session";
  var MOCK_PROVIDER_MARKER = "→ answered by the mock provider (no API key configured)";

  /* ---- DOM helpers -------------------------------------------------------- */

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function pick(obj, names) {
    for (var i = 0; i < names.length; i++) {
      var value = obj ? obj[names[i]] : undefined;
      if (value !== undefined && value !== null && value !== "") return value;
    }
    return null;
  }

  function asArray(value) {
    if (value === undefined || value === null) return [];
    return Array.isArray(value) ? value : [value];
  }

  function replyText(data) {
    var text = pick(data, ["reply", "message", "text", "answer", "content", "output"]);
    if (text === null) return null;
    if (typeof text === "string") return text;
    // Anthropic-style content blocks: [{type:"text", text:"…"}]
    if (Array.isArray(text)) {
      return text
        .map(function (block) {
          if (typeof block === "string") return block;
          return block && typeof block.text === "string" ? block.text : "";
        })
        .filter(Boolean)
        .join("\n");
    }
    return null;
  }

  function toolLine(call) {
    var name = pick(call, ["name", "tool", "tool_name"]) || "tool";
    var input = call ? (call.input !== undefined ? call.input : call.arguments) : null;
    var rendered = "";
    if (input && typeof input === "object") {
      rendered = Object.keys(input)
        .map(function (key) { return key + ": " + String(input[key]); })
        .join(", ");
    } else if (input !== undefined && input !== null) {
      rendered = String(input);
    }
    var line = "→ " + name + "(" + rendered + ")";
    return (call && call.ok === false) ? line + " — no match, nothing returned" : line;
  }

  function pageUrl(action, context) {
    var direct = pick(action, ["url", "path", "href"]);
    if (direct && String(direct).charAt(0) === "/") return String(direct);

    var page = String(pick(action, ["page", "target", "name", "url", "path"]) || "").toLowerCase();
    var courseId = pick(action, ["course_id"]) || context.course_id;
    var assignmentId = pick(action, ["assignment_id"]) || context.assignment_id;
    var studentId = pick(action, ["student_id"]) || context.student_id;
    var skillId = pick(action, ["skill_id"]) || context.skill_id;

    if (page.indexOf("analytic") !== -1) {
      return courseId ? "/courses/" + courseId + "/analytics" : "/";
    }
    if (page.indexOf("grading") !== -1 || page.indexOf("assignment") !== -1) {
      return assignmentId ? "/grading/" + assignmentId : "/";
    }
    if (page.indexOf("student") !== -1) return studentId ? "/students/" + studentId : "/";
    if (page.indexOf("course") !== -1) return courseId ? "/courses/" + courseId : "/";
    if (page.indexOf("skill") !== -1) return skillId ? "/skills/" + skillId : "/skills";
    if (page.indexOf("setting") !== -1) return "/settings";
    if (page === "home" || page === "index" || page === "") return "/";
    return "/";
  }

  /* ---- one chat surface (the panel, or a skill test drive) ---------------- */

  function Chat(root, log, form, input, options) {
    this.root = root;
    this.log = log;
    this.form = form;
    this.input = input;
    this.endpoint = root.getAttribute("data-chat-endpoint") || "/api/chat";
    this.sessionId = null;
    this.context = options.context || {};
    this.pending = null;
    var self = this;

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      self.send(input.value);
    });
    input.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        self.send(input.value);
      }
    });
  }

  Chat.prototype.scroll = function () {
    this.log.scrollTop = this.log.scrollHeight;
  };

  Chat.prototype.add = function (className, text) {
    var node = el("div", className, text);
    this.log.appendChild(node);
    this.scroll();
    return node;
  };

  Chat.prototype.send = async function (text) {
    var message = (text || "").trim();
    if (!message || this.pending) return;
    this.input.value = "";
    this.add("msg msg--user", message);
    this.pending = this.add("msg msg--assistant is-thinking", "…");

    var payload = { message: message, session_id: this.sessionId, context: this.context };
    Object.keys(this.context).forEach(function (key) { payload[key] = this.context[key]; }, this);

    try {
      var response = await fetch(this.endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(payload)
      });
      var data = null;
      try { data = await response.json(); } catch (e) { data = null; }
      this.pending.remove();
      this.pending = null;

      if (!response.ok) {
        var detail = data && typeof data.detail === "string" ? data.detail : response.status + " " + response.statusText;
        this.add("msg msg--assistant msg--error", "The assistant could not answer: " + detail);
        return;
      }
      this.render(data || {});
    } catch (err) {
      if (this.pending) { this.pending.remove(); this.pending = null; }
      this.add("msg msg--assistant msg--error", "The assistant is unreachable: " + err.message);
    }
  };

  Chat.prototype.render = function (data) {
    var sid = pick(data, ["session_id", "session", "chat_session_id"]);
    if (sid) {
      this.sessionId = sid;
      try {
        sessionStorage.setItem(SESSION_KEY, JSON.stringify(
          { id: sid, course_id: this.context.course_id || null }));
      } catch (e) { /* ignore */ }
    }

    asArray(data.tool_calls || data.tools || data.trace).forEach(function (call) {
      this.add("msg msg--tool", toolLine(call));
    }, this);

    var text = replyText(data);
    if (text) this.add("msg msg--assistant", text);

    var actions = asArray(data.actions !== undefined ? data.actions : data.action);
    actions.forEach(function (action) { this.act(action); }, this);

    // /api/skills/{id}/try reports provider problems in `error` beside `reply`.
    if (data.error) this.add("msg msg--assistant msg--error", String(data.error));
    if (data.used_mock || data.provider === "mock") this.add("msg msg--tool", MOCK_PROVIDER_MARKER);

    if (!text && !actions.length && !data.error) {
      this.add("msg msg--assistant", "(no reply)");
    }
  };

  Chat.prototype.restore = async function () {
    var saved = null;
    try { saved = JSON.parse(sessionStorage.getItem(SESSION_KEY) || "null"); } catch (e) { saved = null; }
    if (!saved || !saved.id) return false;
    if ((saved.course_id || null) !== (this.context.course_id || null)) return false;
    var data = null;
    try {
      var response = await fetch("/api/chat/history?session_id=" + encodeURIComponent(saved.id),
        { headers: { Accept: "application/json" } });
      if (!response.ok) return false;
      data = await response.json();
    } catch (e) { return false; }
    var messages = (data && data.messages) || [];
    if (!messages.length) return false;
    this.sessionId = data.session_id || saved.id;
    messages.forEach(function (m) {
      if (m.role === "user") { this.add("msg msg--user", m.content || ""); return; }
      asArray(m.tool_calls).forEach(function (call) {
        this.add("msg msg--tool", toolLine(call));
      }, this);
      if (m.content) this.add("msg msg--assistant", m.content);
      if (m.provider === "mock") this.add("msg msg--tool", MOCK_PROVIDER_MARKER);
    }, this);
    return true;
  };

  Chat.prototype.act = function (action) {
    if (!action || typeof action !== "object") return;
    var type = String(pick(action, ["type", "action", "kind"]) || "").toLowerCase();

    if (type.indexOf("navigate") !== -1) {
      var url = pageUrl(action, this.context);
      this.add("msg msg--tool", "→ opening " + url);
      window.setTimeout(function () { window.location.assign(url); }, 350);
      return;
    }
    if (type.indexOf("grading") !== -1 || type.indexOf("confirm") !== -1) {
      this.confirmCard(action);
      return;
    }
    // Unknown action types are shown, never silently executed.
    this.add("msg msg--tool", "→ " + (type || "action") + " (no handler — nothing ran)");
  };

  Chat.prototype.confirmCard = function (action) {
    var self = this;
    var assignmentId = pick(action, ["assignment_id", "assignmentId", "id"]);
    var count = pick(action, ["ungraded_count", "submission_count", "count", "pending", "n"]);

    var card = el("div", "chat-confirm");
    card.appendChild(el("div", "confirm-title",
      count ? "Grade " + count + " submission" + (Number(count) === 1 ? "" : "s") + "?" : "Start grading?"));

    var bits = [
      pick(action, ["assignment_name", "assignment"]),
      pick(action, ["course_name", "course"]),
      pick(action, ["skill_name", "skill"]),
      pick(action, ["rubric_name", "rubric"])
    ].filter(Boolean);
    if (action.has_rubric === false) bits.push("no rubric attached");
    if (action.has_skill === false) bits.push("no grading skill attached");
    card.appendChild(el("div", "confirm-body",
      (bits.length ? bits.join(" · ") + " — " : "") + "nothing runs until you confirm."));

    var actions = el("div", "confirm-actions");
    var go = el("button", "btn btn--primary btn--sm", "Review privacy & grade");
    go.type = "button";
    var cancel = el("button", "btn btn--sm", "Cancel");
    cancel.type = "button";
    actions.appendChild(go);
    actions.appendChild(cancel);
    card.appendChild(actions);
    this.log.appendChild(card);
    this.scroll();

    cancel.addEventListener("click", function () {
      card.remove();
      self.add("msg msg--tool", "→ cancelled — nothing was graded");
    });

    go.addEventListener("click", function () {
      if (!assignmentId) {
        self.add("msg msg--assistant msg--error", "No assignment was supplied — nothing ran.");
        card.remove();
        return;
      }
      card.remove();
      self.add("msg msg--tool", "→ opening the privacy-gated grading queue");
      window.location.assign("/grading/" + assignmentId);
    });
  };

  /* ---- panel wiring -------------------------------------------------------- */

  function contextFrom(node) {
    var context = {};
    ["course_id", "assignment_id", "student_id", "skill_id"].forEach(function (key) {
      var value = node.getAttribute("data-" + key.replace(/_/g, "-"));
      if (value) context[key] = isNaN(Number(value)) ? value : Number(value);
    });
    var page = node.getAttribute("data-page");
    if (page) context.page = page;
    return context;
  }

  function setOpen(panel, fab, open) {
    panel.hidden = !open;
    fab.setAttribute("aria-expanded", open ? "true" : "false");
    try { sessionStorage.setItem(OPEN_KEY, open ? "1" : "0"); } catch (e) { /* ignore */ }
    if (open) {
      var input = panel.querySelector("#chat-input");
      if (input) input.focus();
    }
  }

  function init() {
    var panel = document.getElementById("chat-panel");
    var fab = document.getElementById("chat-fab");
    if (panel && fab) {
      var log = document.getElementById("chat-log");
      var form = document.getElementById("chat-form");
      var input = document.getElementById("chat-input");
      var panelChat = null;
      if (log && form && input) {
        panelChat = new Chat(panel, log, form, input, { context: contextFrom(panel) });
      }
      fab.addEventListener("click", function () { setOpen(panel, fab, panel.hidden); });
      var close = panel.querySelector(".chat-close");
      if (close) close.addEventListener("click", function () { setOpen(panel, fab, false); });
      document.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && !panel.hidden) setOpen(panel, fab, false);
      });
      var remembered = null;
      try { remembered = sessionStorage.getItem(OPEN_KEY); } catch (e) { /* ignore */ }
      if (remembered === "1") {
        if (panelChat) {
          panelChat.restore().then(function (restored) {
            if (restored) setOpen(panel, fab, true);
            else { try { sessionStorage.setItem(OPEN_KEY, "0"); } catch (e) { /* ignore */ } }
          });
        }
      }
    }

    document.querySelectorAll("[data-mini-chat]").forEach(function (root) {
      var log = root.querySelector("[data-mini-log]");
      var form = root.querySelector("[data-mini-form]");
      var input = form && form.querySelector("textarea");
      if (log && form && input) {
        new Chat(root, log, form, input, { context: contextFrom(root) });
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  document.addEventListener("click", function (ev) {
    var b = ev.target.closest ? ev.target.closest("[data-mini-starter]") : null;
    if (!b) return;
    var box = document.querySelector("[data-mini-form] textarea");
    if (!box) return;
    box.value = b.textContent;
    box.focus();
  });
})();
