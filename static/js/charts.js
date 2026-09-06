/* charts.js — hand-rolled inline SVG charts, fed by the analytics JSON API.
 *
 * No chart library, no CDN, no network beyond the app's own endpoints.
 * Every chart follows the house spec (DESIGN.md § 5):
 *   · marks ≤ 24px thick, 4px rounded data-ends, 2px lines, r≥4 dots with a
 *     2px surface ring; area washes at 10%
 *   · ONE axis, solid hairline gridlines, recessive baseline
 *   · text always in text tokens (.axis-label/.axis-tick/.data-label), never a
 *     series color; colors come from CSS custom properties so light/dark are
 *     the designer's selected steps, not a runtime flip
 *   · direct labels only on the endpoint/extreme; ticks and the table twin
 *     carry the rest — nothing is tooltip-gated
 *   · hit targets larger than the mark, hover === keyboard focus, crosshair on
 *     lines snapping to the nearest x
 *   · legend whenever ≥ 2 series; single-series charts rely on the card title
 *
 * Mounts: <div class="chart-mount" data-chart="…" data-src="…">
 */
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";

  /* ---- element helpers ------------------------------------------------------ */

  function n(tag, attrs) {
    var node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        if (attrs[key] !== null && attrs[key] !== undefined) {
          node.setAttribute(key, String(attrs[key]));
        }
      });
    }
    return node;
  }

  function text(cls, x, y, value, anchor) {
    var node = n("text", { class: cls, x: x, y: y, "text-anchor": anchor || "start" });
    node.textContent = String(value);
    return node;
  }

  function h(tag, cls, value) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (value !== undefined && value !== null) node.textContent = String(value);
    return node;
  }

  function svgRoot(width, height, label) {
    var svg = n("svg", {
      class: "chart-svg",
      viewBox: "0 0 " + width + " " + height,
      role: "img",
      "aria-label": label
    });
    return svg;
  }

  function clip(value, max) {
    var s = String(value);
    return s.length > max ? s.slice(0, max - 1) + "…" : s;
  }

  function round(value, places) {
    var f = Math.pow(10, places || 0);
    return Math.round(Number(value) * f) / f;
  }

  /* Column/bar path: rounded at the data end, square at the baseline. */
  function columnPath(x, width, top, baseline) {
    var r = Math.min(4, width / 2, Math.max(0, baseline - top));
    return "M" + x + "," + baseline +
      " V" + (top + r) +
      " Q" + x + "," + top + " " + (x + r) + "," + top +
      " H" + (x + width - r) +
      " Q" + (x + width) + "," + top + " " + (x + width) + "," + (top + r) +
      " V" + baseline + " Z";
  }

  function barPath(x, y, length, thickness) {
    var r = Math.min(4, length, thickness / 2);
    return "M" + x + "," + y +
      " H" + (x + length - r) +
      " Q" + (x + length) + "," + y + " " + (x + length) + "," + (y + r) +
      " V" + (y + thickness - r) +
      " Q" + (x + length) + "," + (y + thickness) + " " + (x + length - r) + "," + (y + thickness) +
      " H" + x + " Z";
  }

  /* ---- shared tooltip -------------------------------------------------------- */

  var tip = document.getElementById("chart-tip");

  function showTip(title, rows, x, y) {
    if (!tip) return;
    tip.replaceChildren();
    if (title) tip.appendChild(h("div", "tip-title", title));
    rows.forEach(function (row) {
      var line = h("div", "tip-row");
      if (row.color) {
        var key = h("span", "tip-key");
        key.style.background = row.color;
        line.appendChild(key);
      }
      line.appendChild(h("span", "tip-val", row.value));
      line.appendChild(h("span", "tip-label", row.label));
      tip.appendChild(line);
    });
    tip.hidden = false;
    tip.style.position = "fixed";
    var box = tip.getBoundingClientRect();
    tip.style.left = Math.max(8, Math.min(x + 14, window.innerWidth - box.width - 8)) + "px";
    tip.style.top = Math.max(8, y - box.height - 10) + "px";
  }

  function hideTip() { if (tip) tip.hidden = true; }

  /* A hit target that is bigger than its mark, reachable by keyboard.
     `mark` (optional) is the painted mark that lifts while hovered/focused. */
  function hit(svg, attrs, title, rows, mark) {
    var rect = n("rect", Object.assign({ class: "mark-hit", tabindex: "0" }, attrs));
    var lift = function (on) { if (mark) mark.classList.toggle("is-hot", on); };
    rect.addEventListener("pointermove", function (event) {
      lift(true);
      showTip(title, rows, event.clientX, event.clientY);
    });
    rect.addEventListener("pointerleave", function () { lift(false); hideTip(); });
    rect.addEventListener("focus", function () {
      lift(true);
      var box = rect.getBoundingClientRect();
      showTip(title, rows, box.left + box.width / 2, box.top);
    });
    rect.addEventListener("blur", function () { lift(false); hideTip(); });
    svg.appendChild(rect);
    return rect;
  }

  /* ---- table twin ------------------------------------------------------------- */

  function tableTwin(headers, rows) {
    var details = h("details", "chart-table");
    details.appendChild(h("summary", null, "Data table"));
    var table = h("table", "table table--dense");
    var thead = document.createElement("thead");
    var headRow = document.createElement("tr");
    headers.forEach(function (header, index) {
      var th = h("th", index ? "num" : null, header);
      th.setAttribute("scope", "col");
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);
    var tbody = document.createElement("tbody");
    rows.forEach(function (row) {
      var tr = document.createElement("tr");
      row.forEach(function (cell, index) {
        tr.appendChild(h("td", index ? "num" : null, cell));
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    details.appendChild(table);
    return details;
  }

  function legend(entries) {
    var box = h("div", "legend");
    entries.forEach(function (entry) {
      var key = h("span", "key");
      var swatch = h("span", entry.shape === "rect" ? "swatch-rect" : "swatch-line");
      swatch.style.background = entry.color;
      key.appendChild(swatch);
      key.appendChild(document.createTextNode(entry.label));
      box.appendChild(key);
    });
    return box;
  }

  function mount(node, svg, extras, twin) {
    node.replaceChildren();
    var wrap = h("div", "chart-wrap");
    var figure = h("figure", "chart-figure");
    figure.appendChild(svg);
    wrap.appendChild(figure);
    node.appendChild(wrap);
    (extras || []).forEach(function (extra) { node.appendChild(extra); });
    if (twin) node.appendChild(twin);
  }

  function empty(node, message) {
    node.replaceChildren(h("p", "chart-empty", message));
  }

  /* ---- scales ------------------------------------------------------------------ */

  /* Clean tick steps: axis ticks must land on round numbers (1/2/5/10…), and
     the top of the scale is a whole number of steps. */
  function niceScale(maxValue, ticks) {
    var raw = Math.max(1, maxValue) / ticks;
    var pow = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var step = [1, 2, 2.5, 5, 10].map(function (m) { return m * pow; })
      .filter(function (candidate) { return candidate >= raw; })[0] || raw;
    if (step < 1) step = 1;                 // counts are whole submissions
    step = Math.ceil(step);
    return { step: step, max: step * ticks };
  }

  /* ==========================================================================
     1. Score distribution — single-series columns, ordered bins
     ========================================================================== */

  function distribution(node, buckets, subtitle) {
    var counts = buckets.map(function (b) { return b.count || 0; });
    var total = counts.reduce(function (a, b) { return a + b; }, 0);
    if (!total) { empty(node, "No graded submissions in this scope yet."); return; }

    var W = 520, H = 240, padL = 40, padR = 12, top = 16, base = 200;
    var plot = W - padL - padR;
    var band = plot / buckets.length;
    var barW = Math.min(24, band * 0.45);
    var ticks = 4;
    var scale = niceScale(Math.max.apply(null, counts), ticks);
    var max = scale.max;
    var y = function (value) { return base - (value / max) * (base - top); };
    var modal = counts.indexOf(Math.max.apply(null, counts));

    var svg = svgRoot(W, H,
      "Histogram of scores by percent band. Most submissions fall in the " +
      buckets[modal].label + " band, " + counts[modal] + " of " + total + ".");

    for (var i = 1; i <= ticks; i++) {
      var value = scale.step * i;
      svg.appendChild(n("line", { class: "gridline", x1: padL, y1: y(value), x2: W - padR, y2: y(value) }));
      svg.appendChild(text("axis-tick", padL - 8, y(value) + 4, value, "end"));
    }
    svg.appendChild(text("axis-tick", padL - 8, base + 4, "0", "end"));

    buckets.forEach(function (bucket, index) {
      var x = padL + band * index + (band - barW) / 2;
      var count = bucket.count || 0;
      var mark = null;
      if (count > 0) {
        mark = n("path", {
          class: "bar-mark", fill: "var(--chart-1)",
          d: columnPath(x, barW, y(count), base)
        });
        svg.appendChild(mark);
      }
      hit(svg, { x: padL + band * index + 2, y: top, width: band - 4, height: base - top },
        bucket.label + "%",
        [{ value: String(count), label: count === 1 ? "submission" : "submissions" }], mark);
      if (count > 0) {
        svg.appendChild(text("data-label", padL + band * index + band / 2, y(count) - 8, count, "middle"));
      }
      svg.appendChild(text("axis-label", padL + band * index + band / 2, base + 20, bucket.label, "middle"));
    });

    svg.appendChild(n("line", { class: "baseline", x1: padL, y1: base, x2: W - padR, y2: base }));

    var extras = [];
    if (subtitle) {
      var sub = node.parentNode && node.parentNode.querySelector("[data-chart-sub='distribution']");
      if (sub) sub.textContent = subtitle;
    }
    mount(node, svg, extras, tableTwin(["Score band", "Submissions"],
      buckets.map(function (b) { return [b.label + "%", b.count || 0]; })));
  }

  /* ==========================================================================
     2. Per-criterion averages — nominal bars, one hue over a same-ramp track
     ========================================================================== */

  function criteria(node, list) {
    list = (list || []).filter(function (c) { return c.average_percent !== null && c.average_percent !== undefined; });
    if (!list.length) { empty(node, "No criterion scores recorded yet."); return; }

    var W = 620, labelW = 180, barX = 190, barW = 330, thick = 12;
    var rowH = Math.max(34, Math.floor(276 / list.length));
    var H = list.length * rowH + 10;
    var svg = svgRoot(W, H,
      "Average percent of points earned per rubric criterion. Weakest: " +
      list[0].title + " at " + list[0].average_percent + " percent.");

    var pcts = list.map(function (c) { return Math.max(0, Math.min(100, Number(c.average_percent))); });
    var floor = Math.max(0, Math.floor((Math.min.apply(null, pcts) - 5) / 10) * 10);
    if (100 - floor < 20) { floor = 80; }
    var critSub = document.querySelector("[data-chart-sub='criteria']");
    if (critSub) { critSub.textContent = "Percent of the points available, weakest first · axis starts at " + floor + "%"; }
    list.forEach(function (criterion, index) {
      var yTop = index * rowH + Math.round((rowH - thick) / 2);
      var pct = Math.max(0, Math.min(100, Number(criterion.average_percent)));
      var length = Math.max(2, ((pct - floor) / (100 - floor)) * barW);

      svg.appendChild(text("axis-label", labelW, yTop + thick - 2, clip(criterion.title, 34), "end"));
      svg.appendChild(n("rect", { fill: "var(--seq-100)", x: barX, y: yTop, width: barW, height: thick, rx: 4 }));
      var mark = n("path", { class: "bar-mark", fill: "var(--chart-1)", d: barPath(barX, yTop, length, thick) });
      svg.appendChild(mark);
      hit(svg, { x: barX, y: yTop - 6, width: barW, height: thick + 12 }, criterion.title,
        [
          { value: pct + "%", label: "of points earned" },
          { value: String(round(criterion.average_score, 1)), label: "avg of " + round(criterion.max_points, 0) + " pts" },
          { value: String(criterion.count || 0), label: "graded" }
        ], mark);
      svg.appendChild(text("data-label", barX + barW + 8, yTop + thick - 2, pct + "%", "start"));
    });

    mount(node, svg, [], tableTwin(["Criterion", "Average %", "Avg points", "Max"],
      list.map(function (c) {
        return [c.title, c.average_percent, round(c.average_score, 1), round(c.max_points, 0)];
      })));
  }

  /* ==========================================================================
     3. Misconception tags — the second sequential context: rust, one hue
     ========================================================================== */

  function misconceptions(node, list, limit) {
    list = list || [];
    if (!list.length) { empty(node, "No misconception tags recorded yet."); return; }

    var shown = list.slice(0, limit || 7);
    var rest = list.slice(limit || 7);
    var W = 620, labelW = 215, barX = 225, barW = 285, thick = 12;
    var rowH = Math.max(34, Math.floor(276 / shown.length));
    var H = shown.length * rowH + 10;
    var max = Math.max.apply(null, shown.map(function (t) { return t.count; }));

    var svg = svgRoot(W, H,
      "Most common misconception tags. " + shown[0].tag + " leads with " + shown[0].count + ".");

    shown.forEach(function (tag, index) {
      var yTop = index * rowH + Math.round((rowH - thick) / 2);
      var length = Math.max(2, (tag.count / max) * barW);
      svg.appendChild(text("axis-label", labelW, yTop + thick - 2, clip(tag.tag, 40), "end"));
      var mark = n("path", { class: "bar-mark", fill: "var(--chart-2)", d: barPath(barX, yTop, length, thick) });
      svg.appendChild(mark);
      hit(svg, { x: barX, y: yTop - 6, width: barW, height: thick + 12 }, tag.tag,
        [
          { value: String(tag.count), label: tag.count === 1 ? "graded submission" : "graded submissions" },
          { value: String(tag.student_count || 0), label: "students" },
          { value: String(tag.assignment_count || 0), label: "assignments" }
        ], mark);
      svg.appendChild(text("data-label", barX + barW + 8, yTop + thick - 2, tag.count, "start"));
    });

    var rows = shown.map(function (t) { return [t.tag, t.count, t.student_count || 0]; });
    if (rest.length) {
      rows.push([
        rest.length + " smaller tags",
        rest.reduce(function (a, t) { return a + t.count; }, 0),
        "—"
      ]);
    }
    mount(node, svg, [], tableTwin(["Misconception tag", "Mentions", "Students"], rows));
  }

  /* ==========================================================================
     4/5. Trend lines — class average (single series) and student vs class
     ========================================================================== */

  function trendGeometry(points, W, H) {
    var padL = 44, padR = 24, top = 16, base = 200;
    var lows = points.map(function (p) { return p.low !== undefined && p.low !== null ? p.low : p.value; });
    var highs = points.map(function (p) { return p.high !== undefined && p.high !== null ? p.high : p.value; });
    var values = points.map(function (p) { return p.value; });
    var floor = Math.min.apply(null, lows.concat(values));
    var ceiling = Math.max.apply(null, highs.concat(values));
    var min = Math.max(0, Math.floor((floor - 5) / 10) * 10);
    var max = Math.min(100, Math.ceil((ceiling + 5) / 10) * 10);
    if (max - min < 20) { max = Math.min(100, min + 20); }
    if (max - min < 20) { min = Math.max(0, max - 20); }
    var span = max - min;
    var x = function (index) {
      return points.length === 1
        ? (padL + (W - padL - padR) / 2)
        : padL + (index / (points.length - 1)) * (W - padL - padR);
    };
    var y = function (value) { return base - ((value - min) / span) * (base - top); };
    return { padL: padL, padR: padR, top: top, base: base, min: min, max: max, x: x, y: y };
  }

  function axes(svg, geo, points, W) {
    for (var value = geo.min; value <= geo.max; value += 10) {
      svg.appendChild(n("line", { class: "gridline", x1: geo.padL, y1: geo.y(value), x2: W - geo.padR, y2: geo.y(value) }));
      svg.appendChild(text("axis-tick", geo.padL - 8, geo.y(value) + 4, value, "end"));
    }
    svg.appendChild(n("line", { class: "baseline", x1: geo.padL, y1: geo.base, x2: W - geo.padR, y2: geo.base }));
    points.forEach(function (point, index) {
      var anchor = index === 0 ? "start" : (index === points.length - 1 ? "end" : "middle");
      var lx = index === 0 ? geo.padL : (index === points.length - 1 ? W - geo.padR : geo.x(index));
      var shortName = String(point.name || "").split(" · ")[0];
      svg.appendChild(text("axis-label", lx, geo.base + 20, clip(shortName, 22), anchor));
    });
  }

  function crosshair(svg, geo, points, rowsFor, W) {
    var line = n("line", { class: "crosshair", x1: 0, y1: geo.top, x2: 0, y2: geo.base, style: "display:none" });
    svg.appendChild(line);
    svg.addEventListener("pointermove", function (event) {
      var box = svg.getBoundingClientRect();
      var vx = (event.clientX - box.left) * (W / box.width);
      var nearest = 0;
      points.forEach(function (point, index) {
        if (Math.abs(geo.x(index) - vx) < Math.abs(geo.x(nearest) - vx)) nearest = index;
      });
      line.setAttribute("x1", geo.x(nearest));
      line.setAttribute("x2", geo.x(nearest));
      line.style.display = "";
      showTip(points[nearest].name, rowsFor(nearest), event.clientX, event.clientY);
    });
    svg.addEventListener("pointerleave", function () {
      line.style.display = "none";
      hideTip();
    });
    return line;
  }

  function classTrend(node, points) {
    if (!points.length) { empty(node, "No graded assignments yet."); return; }

    var W = 520, H = 240;
    var geo = trendGeometry(points, W, H);
    var svg = svgRoot(W, H,
      "Line chart of the class average across " + points.length + " assignments, ending at " +
      points[points.length - 1].value + " percent.");

    axes(svg, geo, points, W);

    var hasBand = points.every(function (p) { return p.low !== null && p.high !== null && p.low !== undefined; });
    if (hasBand && points.length > 1) {
      var upper = points.map(function (p, i) { return geo.x(i) + "," + geo.y(p.high); });
      var lower = points.map(function (p, i) { return geo.x(i) + "," + geo.y(p.low); }).reverse();
      svg.appendChild(n("path", {
        class: "area-wash", fill: "var(--chart-1)",
        d: "M" + upper.join(" L") + " L" + lower.join(" L") + " Z"
      }));
      svg.appendChild(text("axis-label", W - geo.padR, geo.y(points[points.length - 1].low) + 16,
        "lowest–highest", "end"));
    }

    crosshair(svg, geo, points, function (index) {
      var point = points[index];
      var rows = [{ value: point.value + "%", label: "class average" }];
      if (point.low !== null && point.low !== undefined) {
        rows.push({ value: point.low + "–" + point.high, label: "lowest to highest" });
      }
      rows.push({ value: String(point.graded || 0), label: "graded" });
      return rows;
    }, W);

    svg.appendChild(n("polyline", {
      class: "series-line", stroke: "var(--chart-1)", fill: "none",
      points: points.map(function (p, i) { return geo.x(i) + "," + geo.y(p.value); }).join(" ")
    }));
    points.forEach(function (point, index) {
      svg.appendChild(n("circle", {
        class: "series-dot", fill: "var(--chart-1)",
        cx: geo.x(index), cy: geo.y(point.value), r: 4.5
      }));
    });

    var last = points[points.length - 1];
    var first = points[0];
    svg.appendChild(text("data-label", geo.x(0), geo.y(first.value) - 12, first.value + "%", "start"));
    svg.appendChild(text("data-label", geo.x(points.length - 1), geo.y(last.value) - 12, last.value + "%", "end"));

    mount(node, svg, [], tableTwin(["Assignment", "Average %", "Lowest", "Highest", "Graded"],
      points.map(function (p) {
        return [p.name, p.value, p.low === null || p.low === undefined ? "—" : p.low,
          p.high === null || p.high === undefined ? "—" : p.high, p.graded || 0];
      })));
  }

  function studentTrend(node, points, label) {
    if (!points.length) { empty(node, "No graded work yet."); return; }

    var W = 520, H = 240;
    var geo = trendGeometry(points, W, H);
    var hasClass = points.some(function (p) { return p.classValue !== null && p.classValue !== undefined; });
    var svg = svgRoot(W, H,
      "Line chart of " + label + "'s percent score across " + points.length + " assignments" +
      (hasClass ? ", against the class average." : "."));

    axes(svg, geo, points, W);

    crosshair(svg, geo, points, function (index) {
      var point = points[index];
      var rows = [{ value: point.value + "%", label: label, color: "var(--accent)" }];
      if (point.classValue !== null && point.classValue !== undefined) {
        rows.push({ value: point.classValue + "%", label: "class average", color: "var(--chart-deemph)" });
      }
      return rows;
    }, W);

    if (hasClass) {
      var classPoints = points
        .map(function (p, i) {
          return p.classValue === null || p.classValue === undefined ? null : geo.x(i) + "," + geo.y(p.classValue);
        })
        .filter(Boolean);
      svg.appendChild(n("polyline", {
        class: "series-line", stroke: "var(--chart-deemph)", fill: "none", points: classPoints.join(" ")
      }));
    }

    svg.appendChild(n("polyline", {
      class: "series-line", stroke: "var(--accent)", fill: "none",
      points: points.map(function (p, i) { return geo.x(i) + "," + geo.y(p.value); }).join(" ")
    }));
    points.forEach(function (point, index) {
      svg.appendChild(n("circle", {
        class: "series-dot", fill: "var(--accent)", cx: geo.x(index), cy: geo.y(point.value), r: 4.5
      }));
    });

    var last = points[points.length - 1];
    var lastX = geo.x(points.length - 1);
    svg.appendChild(text("data-label", lastX, geo.y(last.value) - 12, last.value + "%", "end"));
    // Second end label only when it will not collide with the first.
    if (hasClass && last.classValue !== null && last.classValue !== undefined &&
        Math.abs(geo.y(last.classValue) - geo.y(last.value)) > 22) {
      svg.appendChild(text("axis-label", lastX, geo.y(last.classValue) - 10, last.classValue + "%", "end"));
    }

    var keys = [{ label: label, color: "var(--accent)" }];
    if (hasClass) keys.push({ label: "Class average", color: "var(--chart-deemph)" });

    mount(node, svg, [legend(keys)], tableTwin(["Assignment", "Score %", "Class %"],
      points.map(function (p) {
        return [p.name, p.value, p.classValue === null || p.classValue === undefined ? "—" : p.classValue];
      })));
  }

  /* ==========================================================================
     6. Sparklines (stat tiles) — context in the de-emphasis hue, current in accent
     ========================================================================== */

  function sparkline(svg) {
    var values = (svg.getAttribute("data-spark") || "")
      .split(",")
      .map(Number)
      .filter(function (v) { return !isNaN(v); });
    if (values.length < 2) return;
    var W = 96, H = 24, pad = 3;
    var min = Math.min.apply(null, values);
    var max = Math.max.apply(null, values);
    var span = max - min || 1;
    var x = function (i) { return (i / (values.length - 1)) * W; };
    var y = function (v) { return H - pad - ((v - min) / span) * (H - pad * 2); };
    var points = values.map(function (v, i) { return x(i) + "," + y(v); });
    svg.replaceChildren();
    svg.appendChild(n("polyline", {
      class: "spark-context", fill: "none", "stroke-width": 2,
      points: points.slice(0, points.length - 1).join(" ")
    }));
    svg.appendChild(n("polyline", {
      class: "spark-current", fill: "none", "stroke-width": 2,
      points: points.slice(points.length - 2).join(" ")
    }));
  }

  /* ==========================================================================
     Data plumbing
     ========================================================================== */

  var cache = {};

  async function load(url) {
    if (!cache[url]) {
      cache[url] = fetch(url, { headers: { Accept: "application/json" } }).then(function (response) {
        if (!response.ok) throw new Error(response.status + " " + response.statusText);
        return response.json();
      });
    }
    return cache[url];
  }

  function aggregateBuckets(assignments) {
    var totals = {};
    var order = [];
    assignments.forEach(function (block) {
      (block.distribution || []).forEach(function (bucket) {
        if (!(bucket.label in totals)) { totals[bucket.label] = 0; order.push(bucket.label); }
        totals[bucket.label] += bucket.count || 0;
      });
    });
    return order.map(function (label) { return { label: label, count: totals[label] }; });
  }

  function trendPoints(overview) {
    return (overview.assignments || [])
      .filter(function (block) { return block.average_percent !== null && block.average_percent !== undefined; })
      .map(function (block) {
        return {
          id: block.assignment_id,
          name: block.name,
          value: block.average_percent,
          low: block.low_percent,
          high: block.high_percent,
          graded: block.graded_count
        };
      });
  }

  function setStat(name, value) {
    var node = document.querySelector("[data-stat='" + name + "']");
    if (node) node.textContent = value;
  }

  function fillStats(overview, misc, assignmentId) {
    var blocks = overview.assignments || [];
    var scoped = assignmentId
      ? blocks.filter(function (b) { return String(b.assignment_id) === String(assignmentId); })
      : blocks;
    var buckets = aggregateBuckets(scoped);
    var graded = scoped.reduce(function (a, b) { return a + (b.graded_count || 0); }, 0);
    var collected = scoped.reduce(function (a, b) { return a + (b.submission_count || 0); }, 0);
    var weighted = scoped.reduce(function (a, b) {
      return b.average_percent === null || b.average_percent === undefined
        ? a : a + b.average_percent * (b.graded_count || 0);
    }, 0);
    var low = buckets.reduce(function (a, bucket) {
      return /^(0-59|60-69)$/.test(bucket.label) ? a + bucket.count : a;
    }, 0);

    setStat("average", graded ? round(weighted / graded, 1) + "%" : "—");
    setStat("graded", String(graded));
    setStat("graded-note", "of " + collected + " collected");
    setStat("below", String(low));
    setStat("below-note", graded ? round((low / graded) * 100, 0) + "% of graded work" : "no graded work");

    var top = (misc.misconceptions || [])[0];
    setStat("misconception", top ? "“" + top.tag + "”" : "—");
    setStat("misconception-note", top
      ? top.count + " tagged · " + (top.student_count || 0) + " students"
      : "nothing tagged yet");
  }

  /* ---- page wiring ------------------------------------------------------------ */

  async function renderAnalytics(assignmentId) {
    var mounts = document.querySelectorAll(".chart-mount[data-chart]");
    if (!mounts.length) return;
    var charts = document.getElementById("analytics-charts");
    if (charts) charts.classList.add("is-refetching");

    var byName = {};
    mounts.forEach(function (node) { byName[node.getAttribute("data-chart")] = node; });

    try {
      var overview = byName.distribution || byName.trend
        ? await load((byName.distribution || byName.trend).getAttribute("data-src"))
        : null;
      var misc = byName.misconceptions
        ? await load(byName.misconceptions.getAttribute("data-src"))
        : { misconceptions: [] };
      var crit = byName.criteria ? await load(byName.criteria.getAttribute("data-src")) : null;

      if (overview) {
        var blocks = overview.assignments || [];
        var scoped = assignmentId
          ? blocks.filter(function (b) { return String(b.assignment_id) === String(assignmentId); })
          : blocks;

        if (byName.distribution) {
          distribution(byName.distribution, aggregateBuckets(scoped));
          var title = document.querySelector("[data-chart-title='distribution']");
          if (title) {
            title.textContent = "Score distribution" +
              (scoped.length === 1 ? " — " + scoped[0].name : "");
          }
          var sub = document.querySelector("[data-chart-sub='distribution']");
          if (sub) {
            var graded = scoped.reduce(function (a, b) { return a + (b.graded_count || 0); }, 0);
            sub.textContent = graded + " graded submission" + (graded === 1 ? "" : "s") +
              ", binned by percent score";
          }
        }
        if (byName.trend) classTrend(byName.trend, trendPoints(overview));
        fillStats(overview, misc, assignmentId);
      }
      if (byName.criteria && crit) criteria(byName.criteria, crit.criteria || []);
      if (byName.misconceptions) {
        misconceptions(byName.misconceptions, misc.misconceptions || [],
          Number(byName.misconceptions.getAttribute("data-limit")) || 7);
      }

      var note = document.querySelector("[data-filter-note]");
      if (note) {
        note.textContent = assignmentId
          ? "Criterion and tag charts cover the whole course."
          : "";
      }
    } catch (err) {
      mounts.forEach(function (node) {
        if (node.getAttribute("data-chart") !== "student-trend") {
          empty(node, "Could not load analytics: " + err.message);
        }
      });
    } finally {
      if (charts) charts.classList.remove("is-refetching");
    }
  }

  async function renderStudentTrend(node) {
    try {
      var data = await load(node.getAttribute("data-src"));
      var classSrc = node.getAttribute("data-class-src");
      var classAverages = {};
      var classOverall = null;
      if (classSrc) {
        var overview = await load(classSrc);
        (overview.assignments || []).forEach(function (block) {
          classAverages[block.assignment_id] = block.average_percent;
        });
        classOverall = (overview.totals || {}).average_percent;
      }
      var points = (data.timeline || [])
        .filter(function (p) { return p.percent !== null && p.percent !== undefined; })
        .map(function (p) {
          return {
            name: p.name,
            value: p.percent,
            classValue: classAverages[p.assignment_id] === undefined ? null : classAverages[p.assignment_id]
          };
        });
      studentTrend(node, points, node.getAttribute("data-student-label") || "This student");

      var delta = document.querySelector("[data-class-delta]");
      if (delta && classOverall !== null && classOverall !== undefined) {
        var mine = Number(delta.getAttribute("data-student-average"));
        if (!isNaN(mine) && delta.getAttribute("data-student-average") !== "") {
          var diff = round(mine - classOverall, 1);
          delta.classList.remove("flat");
          delta.classList.add(diff > 0 ? "up" : (diff < 0 ? "down" : "flat"));
          delta.textContent = (diff > 0 ? "↑ " : diff < 0 ? "↓ " : "→ ") +
            Math.abs(diff) + " pts vs the class average (" + classOverall + "%)";
        }
      }
    } catch (err) {
      empty(node, "Could not load the trend: " + err.message);
    }
  }

  function init() {
    document.querySelectorAll("svg.stat-spark[data-spark]").forEach(sparkline);

    var studentMount = document.querySelector(".chart-mount[data-chart='student-trend']");
    if (studentMount) renderStudentTrend(studentMount);

    if (document.getElementById("analytics-charts")) {
      var filter = document.querySelector("[data-filter='assignment']");
      renderAnalytics(filter ? filter.value : "");
      if (filter) {
        filter.addEventListener("change", function () { renderAnalytics(filter.value); });
      }
      var courseFilter = document.querySelector("[data-filter='course']");
      if (courseFilter) {
        courseFilter.addEventListener("change", function () {
          window.location.assign("/courses/" + courseFilter.value + "/analytics");
        });
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
