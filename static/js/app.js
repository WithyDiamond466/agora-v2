/* app.js — small declarative glue shared by every page.
 *
 * Pages render server-side; this file only wires the write actions, using
 * attributes on the markup rather than per-page scripts:
 *
 *   [data-panel-toggle="id"]   show/hide the panel with that id
 *   form[data-api]             fetch-submit the form to that endpoint
 *     data-method="POST|PATCH|DELETE"   (default POST)
 *     data-encoding="json|multipart"    (default json; multipart for files)
 *     data-after="reload | redirect:/path/{id} | none"
 *     "{field}" in data-api is replaced by that field's value (and the field
 *     is then left out of the body) — e.g. /api/courses/{course_id}/roster/import
 *   button[data-api]           one-shot request (data-confirm for destructive)
 *   [data-result="#sel"]       write the response message into that element
 *   [data-reveal="inputId"]    show/hide a password field
 *   [data-file-proxy="inputId"] make a .dropzone drive a hidden file input
 *   [data-action="print"]      window.print()
 *
 * Everything writes text with textContent — server messages are untrusted.
 */
(function () {
  "use strict";

  /* ---- helpers ---------------------------------------------------------- */

  function messageFrom(data, fallback) {
    if (!data || typeof data !== "object") return fallback;
    var detail = data.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail.length) {
      return detail
        .map(function (d) {
          var where = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] + ": " : "";
          return where + (d.msg || "invalid value");
        })
        .join("; ");
    }
    if (typeof data.message === "string") return data.message;
    if (typeof data.status === "string") return data.status;
    return fallback;
  }

  function summarize(data) {
    if (!data || typeof data !== "object") return "Done.";
    if (typeof data.detail === "string") return data.detail;
    if (typeof data.message === "string") return data.message;
    if (typeof data.status === "string") return data.status;
    if (typeof data.created === "number") {
      return data.created + " created, " + (data.updated || 0) + " updated, " +
        (data.skipped || 0) + " unchanged.";
    }
    if (data.ok === true) return "OK.";
    return "Done.";
  }

  function showError(scope, text) {
    var box = scope && scope.querySelector("[data-form-error]");
    if (box) {
      box.textContent = text;
      box.hidden = false;
      return;
    }
    var live = document.getElementById("live-region");
    if (live) live.textContent = text;
    window.alert(text);
  }

  function clearError(scope) {
    var box = scope && scope.querySelector("[data-form-error]");
    if (box) { box.textContent = ""; box.hidden = true; }
  }

  async function request(url, method, body, isMultipart) {
    var init = { method: method, headers: { Accept: "application/json" } };
    if (body !== undefined && body !== null) {
      if (isMultipart) {
        init.body = body;                       // browser sets the boundary
      } else {
        init.headers["Content-Type"] = "application/json";
        init.body = JSON.stringify(body);
      }
    }
    var response = await fetch(url, init);
    var data = null;
    try { data = await response.json(); } catch (e) { data = null; }
    if (!response.ok) {
      throw new Error(messageFrom(data, response.status + " " + response.statusText));
    }
    return data;
  }

  function fillUrl(url, form) {
    var used = [];
    var filled = url.replace(/\{([a-zA-Z_][\w-]*)\}/g, function (match, name) {
      var field = form ? form.elements[name] : null;
      if (!field) return match;
      used.push(name);
      return encodeURIComponent(field.value);
    });
    return { url: filled, used: used };
  }

  /* Empty typed inputs are omitted, not sent as "" — an empty date or number
     would fail server-side validation instead of meaning "not set". Text and
     textarea keep their empty string, so a field can be cleared. */
  var OMIT_WHEN_EMPTY = ["date", "datetime-local", "time", "month", "week",
                         "number", "email", "url", "tel"];

  function jsonBody(form, skip) {
    var body = {};
    Array.prototype.forEach.call(form.elements, function (el) {
      if (!el.name || el.disabled || el.type === "file" || el.type === "submit" ||
          el.type === "button" || skip.indexOf(el.name) !== -1) {
        return;
      }
      if (el.type === "checkbox") {
        // data-multi: a group of boxes sharing a name posts the checked values as a list.
        if (el.hasAttribute("data-multi")) {
          if (!Array.isArray(body[el.name])) body[el.name] = [];
          if (el.checked) body[el.name].push(el.value);
          return;
        }
        body[el.name] = el.checked;
        return;
      }
      if (el.type === "radio") { if (el.checked) body[el.name] = el.value; return; }
      var value = el.value;
      if (value === "" && (OMIT_WHEN_EMPTY.indexOf(el.type) !== -1 ||
                           el.hasAttribute("data-omit-empty"))) {
        return;
      }
      body[el.name] = el.type === "number" ? Number(value) : value;
    });
    return body;
  }

  function multipartBody(form, skip) {
    var body = new FormData();
    Array.prototype.forEach.call(form.elements, function (el) {
      if (!el.name || el.disabled || el.type === "submit" || el.type === "button" ||
          skip.indexOf(el.name) !== -1) {
        return;
      }
      if (el.type === "file") {
        Array.prototype.forEach.call(el.files || [], function (file) { body.append(el.name, file); });
        return;
      }
      if (el.type === "checkbox") { if (el.checked) body.append(el.name, "true"); return; }
      if (el.type === "radio") { if (el.checked) body.append(el.name, el.value); return; }
      body.append(el.name, el.value);
    });
    return body;
  }

  function after(el, data) {
    var directive = el.getAttribute("data-after") || "none";
    var target = el.getAttribute("data-result");
    if (target) {
      var out = document.querySelector(target);
      if (out) out.textContent = summarize(data);
    }
    if (directive === "reload") {
      try { sessionStorage.setItem("agora-flash", summarize(data)); } catch (e) {}
      window.location.reload();
    } else if (directive.indexOf("redirect:") === 0) {
      var path = directive.slice("redirect:".length).replace(/\{(\w+)\}/g, function (m, key) {
        return data && data[key] !== undefined ? encodeURIComponent(data[key]) : "";
      });
      window.location.assign(path);
    }
  }

  function busy(el, state) {
    if (!el) return;
    el.disabled = state;
    el.classList.toggle("is-busy", state);
  }

  /* ---- panels ------------------------------------------------------------ */

  document.addEventListener("click", function (event) {
    var toggle = event.target.closest("[data-panel-toggle]");
    if (!toggle) return;
    var panel = document.getElementById(toggle.getAttribute("data-panel-toggle"));
    if (!panel) return;
    event.preventDefault();
    panel.hidden = !panel.hidden;
    if (!panel.hidden) {
      var first = panel.querySelector("input, select, textarea, button");
      if (first) first.focus();
      panel.scrollIntoView({ block: "nearest" });
    }
  });

  /* ---- forms ------------------------------------------------------------- */

  document.addEventListener("submit", async function (event) {
    var form = event.target.closest("form[data-api]");
    if (!form) return;
    event.preventDefault();
    if (form.checkValidity && !form.checkValidity()) { form.reportValidity(); return; }

    clearError(form);
    var multipart = (form.getAttribute("data-encoding") || "json") === "multipart";
    var target = fillUrl(form.getAttribute("data-api"), form);
    var method = (form.getAttribute("data-method") || "POST").toUpperCase();
    var body = multipart ? multipartBody(form, target.used) : jsonBody(form, target.used);
    var submit = form.querySelector("[type='submit']") ||
      document.querySelector("[type='submit'][form='" + form.id + "']");

    busy(submit, true);
    try {
      var data = await request(target.url, method, body, multipart);
      after(form, data);
    } catch (err) {
      showError(form, err.message);
    } finally {
      busy(submit, false);
    }
  });

  /* ---- one-shot buttons --------------------------------------------------- */

  document.addEventListener("click", async function (event) {
    var btn = event.target.closest("button[data-api], a[data-api]");
    if (!btn) return;
    event.preventDefault();
    var confirmText = btn.getAttribute("data-confirm");
    if (confirmText && !window.confirm(confirmText)) return;

    var form = btn.closest("form");
    var target = fillUrl(btn.getAttribute("data-api"), form);
    var method = (btn.getAttribute("data-method") || "POST").toUpperCase();
    busy(btn, true);
    try {
      var data = await request(target.url, method, method === "DELETE" ? null : {}, false);
      after(btn, data);
    } catch (err) {
      var out = btn.getAttribute("data-result") && document.querySelector(btn.getAttribute("data-result"));
      if (out) out.textContent = err.message;
      else showError(form, err.message);
    } finally {
      busy(btn, false);
    }
  });

  /* ---- misc enhancements --------------------------------------------------- */

  document.addEventListener("click", function (event) {
    var reveal = event.target.closest("[data-reveal]");
    if (reveal) {
      var input = document.getElementById(reveal.getAttribute("data-reveal"));
      if (input) {
        var show = input.type === "password";
        input.type = show ? "text" : "password";
        reveal.textContent = show ? "Hide" : "Show";
        reveal.setAttribute("aria-pressed", show ? "true" : "false");
      }
      return;
    }
    var printer = event.target.closest("[data-action='print']");
    if (printer) { window.print(); return; }

    var proxy = event.target.closest("[data-file-proxy]");
    if (proxy) {
      var picker = document.getElementById(proxy.getAttribute("data-file-proxy"));
      if (picker) picker.click();
    }
  });

  document.querySelectorAll("[data-file-proxy]").forEach(function (zone) {
    var picker = document.getElementById(zone.getAttribute("data-file-proxy"));
    if (!picker) return;
    ["dragenter", "dragover"].forEach(function (name) {
      zone.addEventListener(name, function (e) { e.preventDefault(); zone.classList.add("is-over"); });
    });
    ["dragleave", "drop"].forEach(function (name) {
      zone.addEventListener(name, function () { zone.classList.remove("is-over"); });
    });
    zone.addEventListener("drop", function (e) {
      e.preventDefault();
      if (e.dataTransfer && e.dataTransfer.files.length) {
        picker.files = e.dataTransfer.files;
        picker.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });
    zone.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); picker.click(); }
    });
  });

  document.addEventListener("change", function (event) {
    var input = event.target;
    if (input.matches && input.matches("input[type='file'][data-autosubmit]") && input.files.length) {
      var form = input.closest("form");
      if (form) form.requestSubmit ? form.requestSubmit() : form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
    // Provider select filters the model list (skill builder).
    var select = event.target.closest("[data-model-select]");
    if (select) {
      var models = document.getElementById(select.getAttribute("data-model-select"));
      if (!models) return;
      var first = null;
      Array.prototype.forEach.call(models.options, function (option) {
        var match = option.dataset.provider === select.value;
        option.hidden = !match;
        option.disabled = !match;
        if (match && !first) first = option;
      });
      if (first && models.selectedOptions[0] && models.selectedOptions[0].hidden) {
        models.value = first.value;
      }
    }
  });

  document.addEventListener("change", function (event) {
    var input = event.target;
    if (!input || input.type !== "file") return;
    var out = input.parentNode && input.parentNode.querySelector("[data-file-name]");
    if (!out) return;
    if (input.files && input.files.length) {
      out.textContent = input.files[0].name;
      out.hidden = false;
    } else {
      out.textContent = "";
      out.hidden = true;
    }
  });

  (function renderFlash() {
    var text = null;
    try { text = sessionStorage.getItem("agora-flash"); sessionStorage.removeItem("agora-flash"); } catch (e) {}
    if (!text) return;
    var el = document.createElement("div");
    el.className = "flash";
    el.setAttribute("role", "status");
    el.textContent = text;
    document.body.appendChild(el);
    var live = document.getElementById("live-region");
    if (live) live.textContent = text;
    setTimeout(function () { el.classList.add("is-out"); }, 7000);
    setTimeout(function () { if (el.parentNode) el.parentNode.removeChild(el); }, 7400);
  })();

  /* Terms of use gate: until accepted, every page says so (cloud grading,
     release and export refuse with a 409 pointing at /terms). */
  (function termsBanner() {
    if (window.location.pathname === "/terms") return;
    var main = document.getElementById("main");
    if (!main || !window.fetch) return;
    fetch("/api/terms/status", { headers: { Accept: "application/json" } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (status) {
        if (!status || status.accepted) return;
        var banner = document.createElement("div");
        banner.className = "terms-banner";
        banner.setAttribute("role", "status");
        var text = document.createElement("span");
        var strong = document.createElement("strong");
        strong.textContent = status.previous_version ? "The terms of use changed. " : "Terms of use not accepted. ";
        text.appendChild(strong);
        text.appendChild(document.createTextNode(
          "Grading with a cloud model, releasing feedback and exporting are locked until you accept them."));
        var link = document.createElement("a");
        link.className = "btn btn--sm btn--primary";
        link.href = "/terms";
        link.textContent = "Read and accept";
        banner.appendChild(text);
        banner.appendChild(link);
        main.insertBefore(banner, main.firstChild);
      })
      .catch(function () { /* offline or old server: no banner */ });
  })();

  (function settingsSpy() {
    var index = document.querySelector(".settings-index");
    if (!index || !("IntersectionObserver" in window)) return;
    var links = {};
    [].forEach.call(index.querySelectorAll("a[href^='#']"), function (a) {
      links[a.getAttribute("href").slice(1)] = a;
    });
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var link = links[entry.target.id];
        if (!link || !entry.isIntersecting) return;
        [].forEach.call(index.querySelectorAll("a"), function (a) { a.removeAttribute("aria-current"); });
        link.setAttribute("aria-current", "true");
      });
    }, { rootMargin: "-10% 0px -80% 0px" });
    Object.keys(links).forEach(function (id) {
      var section = document.getElementById(id);
      if (section) observer.observe(section);
    });
  })();
})();
