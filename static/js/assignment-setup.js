/* Configure an ungraded assignment using an existing or new rubric. */
(function () {
  "use strict";
  var form = document.getElementById("assignment-setup");
  if (!form) return;
  var picker = document.getElementById("setup-rubric");
  var newRubric = document.getElementById("setup-new-rubric");
  var rows = document.getElementById("setup-criteria");
  var error = document.getElementById("setup-error");
  var sequence = 0;

  function addCriterion() {
    sequence += 1;
    var row = document.createElement("fieldset");
    row.className = "setup-criterion";
    var legend = document.createElement("legend");
    legend.textContent = "Criterion " + sequence;
    row.appendChild(legend);
    [
      ["title", "Criterion title", "input"],
      ["max_points", "Maximum points", "input"],
      ["description", "What earns credit", "textarea"]
    ].forEach(function (spec) {
      var field = document.createElement("div");
      field.className = "field";
      var label = document.createElement("label");
      var input = document.createElement(spec[2]);
      input.id = "setup-criterion-" + sequence + "-" + spec[0];
      input.dataset.setupField = spec[0];
      label.htmlFor = input.id;
      label.textContent = spec[1];
      if (spec[0] === "max_points") {
        input.type = "number"; input.min = "0.01"; input.step = "any";
        input.value = "10"; input.required = true;
      } else if (spec[0] === "title") {
        input.required = true; input.maxLength = 200;
      } else { input.rows = 2; input.maxLength = 10000; }
      field.appendChild(label); field.appendChild(input); row.appendChild(field);
    });
    var remove = document.createElement("button");
    remove.type = "button"; remove.className = "btn btn--sm";
    remove.textContent = "Remove criterion";
    remove.addEventListener("click", function () { row.remove(); });
    row.appendChild(remove); rows.appendChild(row);
    return row;
  }
  addCriterion();
  document.getElementById("setup-add-criterion").addEventListener("click", function () {
    addCriterion().querySelector("input").focus();
  });
  picker.addEventListener("change", function () {
    newRubric.hidden = picker.value !== "new";
    newRubric.disabled = picker.value !== "new";
  });
  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    if (!form.reportValidity()) return;
    error.hidden = true;
    var payload = { skill_id: Number(document.getElementById("setup-skill").value) };
    if (picker.value === "new") {
      payload.rubric_name = document.getElementById("setup-rubric-name").value;
      payload.criteria = Array.prototype.map.call(rows.children, function (row) {
        var criterion = {};
        row.querySelectorAll("[data-setup-field]").forEach(function (input) {
          criterion[input.dataset.setupField] = input.dataset.setupField === "max_points" ? Number(input.value) : input.value;
        });
        return criterion;
      });
      if (!payload.criteria.length) { error.textContent = "Add at least one criterion."; error.hidden = false; return; }
    } else { payload.rubric_id = Number(picker.value); }
    var button = form.querySelector("[type='submit']");
    button.disabled = true;
    try {
      var response = await fetch(form.dataset.setupUrl, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
      });
      var data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Check the rubric fields and try again.");
      sessionStorage.setItem("agora-flash", "Assignment setup saved.");
      window.location.reload();
    } catch (err) {
      error.textContent = err.message; error.hidden = false; button.disabled = false;
    }
  });
}());
