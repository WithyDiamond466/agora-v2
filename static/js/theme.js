/* theme.js — light/dark mode.
 *
 * Three states: "light" and "dark" pin the mode (stored in localStorage under
 * "agora-theme"); no stored value means "follow the OS". base.html applies the
 * stored value before first paint, so this file only handles changes.
 */
(function () {
  "use strict";

  var KEY = "agora-theme";
  var root = document.documentElement;

  function stored() {
    try {
      var v = localStorage.getItem(KEY);
      return v === "light" || v === "dark" ? v : null;
    } catch (e) {
      return null;
    }
  }

  function systemMode() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function effective() {
    return root.dataset.theme || systemMode();
  }

  function apply(mode) {
    if (mode === "light" || mode === "dark") {
      root.dataset.theme = mode;
      try { localStorage.setItem(KEY, mode); } catch (e) { /* ignore */ }
    } else {
      delete root.dataset.theme;
      try { localStorage.removeItem(KEY); } catch (e) { /* ignore */ }
    }
    sync();
    document.dispatchEvent(new CustomEvent("agora:themechange", { detail: { mode: effective() } }));
  }

  function sync() {
    var pinned = stored();
    var now = effective();
    document.querySelectorAll("[data-theme-toggle]").forEach(function (btn) {
      btn.setAttribute("aria-pressed", now === "dark" ? "true" : "false");
      btn.title = now === "dark" ? "Switch to light mode" : "Switch to dark mode";
    });
    document.querySelectorAll("[data-theme-radio]").forEach(function (radio) {
      radio.checked = radio.value === (pinned || "system");
    });
  }

  document.addEventListener("click", function (event) {
    var toggle = event.target.closest("[data-theme-toggle]");
    if (!toggle) return;
    apply(effective() === "dark" ? "light" : "dark");
  });

  document.addEventListener("change", function (event) {
    var radio = event.target.closest("[data-theme-radio]");
    if (!radio || !radio.checked) return;
    apply(radio.value === "system" ? null : radio.value);
  });

  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    var onChange = function () {
      if (!stored()) {
        sync();
        document.dispatchEvent(new CustomEvent("agora:themechange", { detail: { mode: effective() } }));
      }
    };
    if (mq.addEventListener) mq.addEventListener("change", onChange);
    else if (mq.addListener) mq.addListener(onChange);
  }

  sync();
})();
