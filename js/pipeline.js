(function(){
  "use strict";

  // =====================================================================
  // SCREENING PROGRAMME CAPACITY MODEL
  //
  // A Simulink-style block model of the service around the classifier, solved
  // in closed form. The question it answers is the only one that matters for
  // deployment: given a population to screen, which single stage stops the
  // programme reaching it?
  //
  // The model is deliberately steady-state and deterministic. Queueing theory
  // would give better waiting times, but it needs arrival and service
  // distributions that nobody has for a programme that does not exist yet, and
  // inventing them would dress up a guess as a result. Demand over capacity, per
  // stage, needs no such assumptions and already settles the ordering — which is
  // the decision this page exists to inform.
  // =====================================================================

  const DEFAULTS = {
    target: 100000, perPatient: 2,
    stations: 4, perStation: 90, rejectPct: 12,
    bandwidth: 10, imgSizeMB: 2, netHours: 12,
    modelRate: 6, nodes: 1,
    referralPct: 18, docs: 1, perDoc: 60
  };

  const FIELDS = {
    target:"p-target", perPatient:"p-perpatient",
    stations:"p-stations", perStation:"p-perstation", rejectPct:"p-reject",
    bandwidth:"p-bandwidth", imgSizeMB:"p-imgsize", netHours:"p-nethours",
    modelRate:"p-modelrate", nodes:"p-nodes",
    referralPct:"p-referral", docs:"p-docs", perDoc:"p-perdoc"
  };

  const SVGNS = "http://www.w3.org/2000/svg";

  // ---------------------------------------------------------------------
  // SMALL HELPERS
  // ---------------------------------------------------------------------
  function $(id){ return document.getElementById(id); }
  function clamp(v,lo,hi){ return Math.min(hi, Math.max(lo, v)); }
  function num(id, fallback){
    const v = parseFloat($(id).value);
    return (isFinite(v) && v >= 0) ? v : fallback;
  }
  function fmt(n){
    if (!isFinite(n)) return "unlimited";
    if (n >= 1e6) return (n/1e6).toFixed(n >= 1e7 ? 0 : 1) + "M";
    if (n >= 10000) return Math.round(n).toLocaleString();
    if (n >= 100) return Math.round(n).toLocaleString();
    if (n >= 10) return n.toFixed(0);
    return n.toFixed(1);
  }
  // A stage with zero capacity has no ratio to report — it is a stop, not a
  // percentage — so it says so rather than printing an em dash that reads as
  // missing data.
  function pct(x){ return isFinite(x) ? Math.round(x*100) + "%" : "no capacity"; }

  // ---------------------------------------------------------------------
  // UTILISATION COLOUR
  //
  // The colour on this page is a reading, not decoration: it is the same
  // utilisation figure printed beside every block, mapped onto one ramp with a
  // scale drawn under the diagram. The ramp is anchored at the two points that
  // mean something — 100%, where a stage stops keeping up, and roughly 75%,
  // where a steady-state model is already optimistic because real arrivals bunch.
  // Colour is never the only carrier: the choke point is also outlined heavily,
  // badged, and listed first in the table.
  // ---------------------------------------------------------------------
  const UTIL_STOPS = [
    [0.00,  13, 122,  99],   // deep teal — large headroom
    [0.50,  47, 143,  47],   // green
    [0.75, 224, 169,   0],   // amber — the point where variance starts to bite
    [0.95, 232,  99,  26],   // orange — effectively saturated
    [1.20, 192,   0,   0]    // red — over capacity, a queue is forming
  ];
  function utilColour(u){
    if (!isFinite(u)) u = 2;
    const t = clamp(u, 0, 1.20);
    for (let i=1;i<UTIL_STOPS.length;i++){
      if (t <= UTIL_STOPS[i][0]){
        const a = UTIL_STOPS[i-1], b = UTIL_STOPS[i];
        const f = (t - a[0]) / (b[0] - a[0]);
        return [
          Math.round(a[1] + (b[1]-a[1])*f),
          Math.round(a[2] + (b[2]-a[2])*f),
          Math.round(a[3] + (b[3]-a[3])*f)
        ];
      }
    }
    return UTIL_STOPS[UTIL_STOPS.length-1].slice(1);
  }
  function rgb(c){ return "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")"; }
  // Text colour chosen from the fill's luminance rather than fixed, so the
  // amber part of the ramp stays readable instead of white-on-yellow.
  function inkOn(c){
    const l = (0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]) / 255;
    return l > 0.55 ? "#111111" : "#ffffff";
  }
  function tint(c, amount){
    return "rgb(" + [0,1,2].map(i => Math.round(c[i] + (255-c[i])*amount)).join(",") + ")";
  }

  // ---------------------------------------------------------------------
  // THE MODEL
  // ---------------------------------------------------------------------
  function readInputs(){
    return {
      target:      num(FIELDS.target, DEFAULTS.target),
      perPatient:  Math.max(1, num(FIELDS.perPatient, DEFAULTS.perPatient)),
      stations:    num(FIELDS.stations, DEFAULTS.stations),
      perStation:  num(FIELDS.perStation, DEFAULTS.perStation),
      rejectPct:   clamp(num(FIELDS.rejectPct, DEFAULTS.rejectPct), 0, 90),
      bandwidth:   num(FIELDS.bandwidth, DEFAULTS.bandwidth),
      imgSizeMB:   Math.max(0.01, num(FIELDS.imgSizeMB, DEFAULTS.imgSizeMB)),
      netHours:    clamp(num(FIELDS.netHours, DEFAULTS.netHours), 0.1, 24),
      modelRate:   Math.max(0.01, num(FIELDS.modelRate, DEFAULTS.modelRate)),
      nodes:       Math.max(1, num(FIELDS.nodes, DEFAULTS.nodes)),
      referralPct: clamp(num(FIELDS.referralPct, DEFAULTS.referralPct), 0, 100),
      docs:        num(FIELDS.docs, DEFAULTS.docs),
      perDoc:      num(FIELDS.perDoc, DEFAULTS.perDoc)
    };
  }

  function solve(v){
    const patientsPerDay = v.target / 365;
    const usableImages   = patientsPerDay * v.perPatient;
    const rejectFrac     = v.rejectPct / 100;
    // Retakes load the camera and nothing else: quality assessment runs on the
    // capture device, so a rejected frame is never uploaded, never inferred and
    // never reviewed.
    const captureAttempts = usableImages / (1 - rejectFrac);
    const referrals       = patientsPerDay * (v.referralPct / 100);

    const stages = [
      {
        key: "acq",
        name: "Image acquisition",
        sub: "fundus camera stations",
        unit: "images/day",
        demand: captureAttempts,
        capacity: v.stations * v.perStation,
        demandNote: "usable images plus retakes",
        capacityNote: fmt(v.stations) + " station(s) × " + fmt(v.perStation) + " images/day",
        relief: "add a camera station, or cut the reject rate so fewer captures are wasted"
      },
      {
        key: "qc",
        name: "Quality gate",
        sub: "on-device, before upload",
        unit: "images/day",
        demand: captureAttempts,
        // The check is a handful of pixel statistics on the capture device: it
        // is not a stage that can bind, and pretending it might would put a
        // decorative block on the critical path.
        capacity: Infinity,
        demandNote: "every capture is checked",
        capacityNote: "runs on the capture device",
        relief: "not a constraint; lowering the reject rate helps acquisition, not this stage"
      },
      {
        key: "net",
        name: "Network transfer",
        sub: "PHC uplink to inference",
        unit: "images/day",
        demand: usableImages,
        capacity: (v.bandwidth * 3600 * v.netHours) / (v.imgSizeMB * 8),
        demandNote: "only images that passed quality",
        capacityNote: fmt(v.bandwidth) + " Mbps × " + fmt(v.netHours) + " h/day at " + v.imgSizeMB + " MB/image",
        relief: "compress harder, upload overnight for more hours, or raise the link speed"
      },
      {
        key: "inf",
        name: "AI inference",
        sub: "MobileNet + Grad-CAM + lesions",
        unit: "images/day",
        demand: usableImages,
        capacity: v.modelRate * 1440 * v.nodes,
        demandNote: "one pass per image",
        capacityNote: fmt(v.modelRate) + " images/min × 1440 min × " + fmt(v.nodes) + " node(s)",
        relief: "add an inference node — the cheapest capacity on this diagram to buy"
      },
      {
        key: "triage",
        name: "Referral triage",
        sub: "automatic gate on model output",
        unit: "patients/day",
        demand: patientsPerDay,
        capacity: Infinity,
        demandNote: "every screened patient is sorted",
        capacityNote: "a comparison, not a queue",
        relief: "not a constraint; the referral rate it produces is what loads review"
      },
      {
        key: "review",
        name: "Specialist review",
        sub: "flagged patients only",
        unit: "patients/day",
        demand: referrals,
        capacity: v.docs * v.perDoc,
        demandNote: v.referralPct + "% of patients flagged",
        capacityNote: fmt(v.docs) + " ophthalmologist(s) × " + fmt(v.perDoc) + " reviews/day",
        relief: "more specialist time, or a better-calibrated threshold so fewer normals are flagged"
      }
    ];

    stages.forEach(s => {
      s.util = s.capacity > 0 ? s.demand / s.capacity : Infinity;
      if (!isFinite(s.capacity)) s.util = 0;
      // What this stage alone would let the programme screen per year, if every
      // other stage were infinitely fast.
      s.ceiling = ceilingFor(s, v);
      s.flow = Math.min(s.demand, s.capacity);
    });

    const binding = stages.filter(s => isFinite(s.capacity));
    let choke = binding[0];
    binding.forEach(s => { if (s.util > choke.util) choke = s; });

    const maxUtil = choke.util;
    const over = maxUtil > 1;
    const achievable = over ? v.target / maxUtil : v.target;
    // The shortfall is in patients, which is the unit the programme is set in;
    // stages measured in images are converted back so one number can be quoted.
    const shortfallPerDay = Math.max(0, patientsPerDay - achievable/365);

    const order = binding.slice().sort((a,b) => b.util - a.util);

    return {
      v, stages, binding, choke, maxUtil, over, achievable, shortfallPerDay,
      patientsPerDay, usableImages, captureAttempts, referrals, order,
      headroom: choke.ceiling
    };
  }

  // Annual patient throughput this one stage permits, holding every other input
  // fixed. Inverting each stage's own arithmetic rather than scaling a single
  // aggregate keeps the retake and referral factors where they belong.
  function ceilingFor(s, v){
    const perPatient = v.perPatient;
    const keepFrac = 1 - v.rejectPct/100;
    switch (s.key){
      case "acq":    return s.capacity * keepFrac / perPatient * 365;
      case "net":
      case "inf":    return s.capacity / perPatient * 365;
      case "review": return v.referralPct > 0 ? s.capacity / (v.referralPct/100) * 365 : Infinity;
      case "qc":
      case "triage": return Infinity;
      default:       return Infinity;
    }
  }

  // ---------------------------------------------------------------------
  // SVG CONSTRUCTION
  // ---------------------------------------------------------------------
  function el(tag, attrs, text){
    const n = document.createElementNS(SVGNS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    return n;
  }

  const BW = 178, BH = 104;   // block geometry
  const HDR = 24;             // coloured header strip height

  // One Simulink-style block: a coloured header carrying the stage name and its
  // utilisation, and a white body carrying the two numbers that produced it.
  const NEUTRAL = [110, 110, 110];
  function block(g, x, y, s, isChoke){
    // A stage with no capacity limit is drawn neutral rather than green. Green
    // on this page means measured headroom; a block that cannot bind has not
    // been measured at all, and colouring it the same would be a false reading.
    const bounded = isFinite(s.capacity);
    const c = bounded ? utilColour(s.util) : NEUTRAL;
    const fill = rgb(c);

    g.appendChild(el("rect", {
      x, y, width: BW, height: BH, rx: 3,
      class: "blk" + (isChoke ? " blk-choke" : ""),
      fill: tint(c, bounded ? 0.90 : 0.94),
      stroke: isChoke ? "#111" : fill
    }));

    // header strip, clipped to the block's rounded top by simply overdrawing
    g.appendChild(el("path", {
      d: "M" + x + " " + (y+HDR) + " L" + x + " " + (y+3) +
         " q0 -3 3 -3 L" + (x+BW-3) + " " + y + " q3 0 3 3 L" + (x+BW) + " " + (y+HDR) + " Z",
      fill: fill, stroke: "none"
    }));
    g.appendChild(el("text", {
      x: x+9, y: y+16, class: "hdr-t", fill: inkOn(c)
    }, s.name));
    g.appendChild(el("text", {
      x: x+BW-9, y: y+16, class: "hdr-t", fill: inkOn(c), "text-anchor": "end"
    }, bounded ? pct(s.util) : "no limit"));

    g.appendChild(el("text", { x: x+9, y: y+HDR+15, class: "b-sub" }, s.sub));

    if (bounded){
      g.appendChild(el("text", { x: x+9, y: y+HDR+35, class: "b-lbl" }, "demand"));
      g.appendChild(el("text", { x: x+BW-9, y: y+HDR+35, class: "b-num", "text-anchor": "end" },
        fmt(s.demand)));
      g.appendChild(el("text", { x: x+9, y: y+HDR+52, class: "b-lbl" }, "capacity"));
      g.appendChild(el("text", { x: x+BW-9, y: y+HDR+52, class: "b-num", "text-anchor": "end" },
        fmt(s.capacity)));
      g.appendChild(el("text", { x: x+BW-9, y: y+HDR+66, class: "b-sub", "text-anchor": "end" },
        s.unit));
    } else {
      g.appendChild(el("text", { x: x+9, y: y+HDR+37, class: "b-lbl" }, "not a constraint"));
      g.appendChild(el("text", { x: x+9, y: y+HDR+53, class: "b-sub" }, s.capacityNote));
    }

    if (isChoke){
      const bw = 96;
      g.appendChild(el("rect", { x: x + BW/2 - bw/2, y: y - 17, width: bw, height: 16,
        fill: "#c00000", stroke: "#111", "stroke-width": 1 }));
      g.appendChild(el("text", { x: x + BW/2, y: y - 5, class: "choke-badge",
        "text-anchor": "middle" }, "CHOKE POINT"));
    }
  }

  // Wire thickness carries the flow, so the diagram shows the narrowing without
  // needing to be read. It is capped because an over-thick line stops looking
  // like a signal and starts looking like a block.
  function wireWidth(flow, maxFlow){
    if (!(maxFlow > 0)) return 1.5;
    return clamp(1.2 + 5.5*Math.sqrt(flow/maxFlow), 1.2, 7);
  }

  function wire(g, d, flow, maxFlow, label, lx, ly, anchor){
    g.appendChild(el("path", { d, class: "wire", "stroke-width": wireWidth(flow, maxFlow),
      "marker-end": "url(#arrowhead)" }));
    if (label){
      g.appendChild(el("text", { x: lx, y: ly, class: "wire-lbl",
        "text-anchor": anchor || "middle" }, label));
    }
  }

  function renderFlow(m){
    const svg = $("flow");
    // <title> and <desc> are the diagram's accessible text and are authored in
    // the HTML; everything else was drawn by the last update and is replaced.
    Array.prototype.slice.call(svg.childNodes).forEach(n => {
      if (n.nodeName !== "title" && n.nodeName !== "desc") svg.removeChild(n);
    });

    const defs = el("defs", {});
    // markerUnits must be userSpaceOnUse. By default a marker scales with the
    // stroke width, and these wires vary their width to carry the flow — so on
    // the busiest wire the arrowhead grew to about forty units and swallowed the
    // whole gap between two blocks, taking the flow label with it.
    const mk = el("marker", { id: "arrowhead", viewBox: "0 0 12 12", refX: "11", refY: "6",
      markerWidth: "12", markerHeight: "12", markerUnits: "userSpaceOnUse",
      orient: "auto-start-reverse" });
    mk.appendChild(el("path", { d: "M0 0.5 L12 6 L0 11.5 z", fill: "#111" }));
    defs.appendChild(mk);
    svg.appendChild(defs);

    const g = el("g", {});
    svg.appendChild(g);

    const by = m.stages.reduce((o,s) => (o[s.key] = s, o), {});
    // Wires carry the demand handed to the next stage, not the flow the previous
    // one manages. That is the honest reading for this diagram: every block shows
    // what is asked of it against what it can do, and a wire that quietly shrank
    // to a saturated stage's output would make every stage downstream of the
    // choke point look comfortable when it has simply been starved. Where a
    // stage cannot meet the demand on its wire, the red queue badge says so.
    const maxFlow = Math.max.apply(null, m.stages.map(s => s.demand || 0).concat([1]));

    // ---- row 1: capture through inference
    const R1 = 70, xs = [10, 240, 458, 676, 894];
    const row1 = ["acq", "qc", "net", "inf"];

    // source
    g.appendChild(el("rect", { x: xs[0], y: R1+18, width: 150, height: 68, rx: 3, class: "sink" }));
    // patient intake is a source, not a stage: it has no capacity to exceed
    g.appendChild(el("text", { x: xs[0]+12, y: R1+40, class: "hdr-t", fill: "#111" }, "Patient intake"));
    g.appendChild(el("text", { x: xs[0]+12, y: R1+58, class: "b-num" }, fmt(m.patientsPerDay) + " patients/day"));
    g.appendChild(el("text", { x: xs[0]+12, y: R1+73, class: "b-sub" },
      fmt(m.v.target) + "/year ÷ 365"));

    let px = xs[0] + 150;
    row1.forEach((key, i) => {
      const s = by[key];
      const x = xs[i+1] - 40;
      const isChoke = (s === m.choke);
      block(g, x, R1, s, isChoke);
      wire(g, "M" + px + " " + (R1+BH/2) + " L" + x + " " + (R1+BH/2),
        s.demand, maxFlow, fmt(s.demand), (px+x)/2, R1+BH/2 + 17);
      px = x + BW;
    });

    // retake feedback: quality gate back to acquisition
    const rejected = m.captureAttempts - m.usableImages;
    const qcX = xs[2]-40, acqX = xs[1]-40;
    wire(g,
      "M" + (qcX+BW/2) + " " + (R1+BH) + " L" + (qcX+BW/2) + " " + (R1+BH+34) +
      " L" + (acqX+BW/2) + " " + (R1+BH+34) + " L" + (acqX+BW/2) + " " + (R1+BH),
      rejected, maxFlow, "retakes · " + fmt(rejected) + "/day",
      (qcX+acqX)/2 + BW/2, R1+BH+48);

    // ---- link down to row 2
    const R2 = 305;
    const infX = xs[4]-40;
    const triX = 234-40, revX = 560-40;
    wire(g,
      "M" + (infX+BW/2) + " " + (R1+BH) + " L" + (infX+BW/2) + " " + (R1+BH+70) +
      " L" + (triX+BW/2) + " " + (R1+BH+70) + " L" + (triX+BW/2) + " " + R2,
      by.inf.demand, maxFlow, "graded results · " + fmt(m.patientsPerDay) + " patients/day",
      (infX+BW/2) - 14, R1+BH+62, "end");

    block(g, triX, R2, by.triage, by.triage === m.choke);
    block(g, revX, R2, by.review, by.review === m.choke);

    wire(g, "M" + (triX+BW) + " " + (R2+BH/2) + " L" + revX + " " + (R2+BH/2),
      m.referrals, maxFlow, "flagged · " + fmt(m.referrals) + "/day",
      (triX+BW+revX)/2, R2+BH/2 - 10);

    // cleared patients leave the pipeline
    const cleared = m.patientsPerDay - m.referrals;
    wire(g, "M" + (triX+BW/2) + " " + (R2+BH) + " L" + (triX+BW/2) + " " + (R2+BH+52),
      cleared, maxFlow, "", 0, 0);
    g.appendChild(el("rect", { x: triX-6, y: R2+BH+52, width: BW+12, height: 46, rx: 3, class: "sink" }));
    g.appendChild(el("text", { x: triX+4, y: R2+BH+72, class: "hdr-t", fill: "#111" }, "Cleared — routine recall"));
    g.appendChild(el("text", { x: triX+4, y: R2+BH+88, class: "b-sub" },
      fmt(cleared) + " patients/day · no specialist time used"));

    // referred onward
    const seen = Math.min(m.referrals, by.review.capacity);
    wire(g, "M" + (revX+BW) + " " + (R2+BH/2) + " L" + (revX+BW+56) + " " + (R2+BH/2),
      seen, maxFlow, "", 0, 0);
    g.appendChild(el("rect", { x: revX+BW+56, y: R2+20, width: 190, height: 64, rx: 3, class: "sink" }));
    g.appendChild(el("text", { x: revX+BW+66, y: R2+41, class: "hdr-t", fill: "#111" }, "Treatment pathway"));
    g.appendChild(el("text", { x: revX+BW+66, y: R2+59, class: "b-num" }, fmt(seen) + " patients/day"));
    g.appendChild(el("text", { x: revX+BW+66, y: R2+75, class: "b-sub" }, "reviewed and referred on"));

    // A queue is drawn only where one actually forms, so the diagram does not
    // suggest a backlog that the numbers do not support.
    m.binding.forEach(s => {
      if (s.util <= 1.001) return;
      const pos = blockPos(s.key);
      if (!pos) return;
      const q = s.demand - s.capacity;
      const bw = 104, bx = pos.x - 5, byq = pos.y + BH - 9;
      g.appendChild(el("rect", { x: bx, y: byq, width: bw, height: 18, rx: 2,
        fill: "#c00000", stroke: "#111", "stroke-width": 1 }));
      g.appendChild(el("text", { x: bx + bw/2, y: byq + 12.5, class: "choke-badge",
        "text-anchor": "middle" }, "queue +" + fmt(q) + "/day"));
    });

    function blockPos(key){
      const i = row1.indexOf(key);
      if (i >= 0) return { x: xs[i+1]-40, y: R1 };
      if (key === "triage") return { x: triX, y: R2 };
      if (key === "review") return { x: revX, y: R2 };
      return null;
    }
  }

  // ---------------------------------------------------------------------
  // VERDICT, TABLE, RELIEF ORDER
  // ---------------------------------------------------------------------
  function renderVerdict(m){
    const c = utilColour(m.choke.util);
    const box = $("verdict");
    const years = m.over ? (m.v.target / m.achievable) : 1;

    const stopped = !isFinite(m.maxUtil);

    let html = '<div class="verdict" style="border-color:' + rgb(c) + '">';
    html += '<p class="headline">The choke point is <span class="who" style="background:' +
      rgb(c) + ';color:' + inkOn(c) + '">' + m.choke.name + '</span> ' +
      (stopped ? 'and it has no capacity at all.' : 'at ' + pct(m.choke.util) + ' of capacity.') +
      '</p>';

    if (stopped){
      html += '<p>Nothing gets through. ' + m.choke.name + ' is configured with zero capacity, so ' +
        'the pipeline stops there however fast every other stage runs, and no one is screened at ' +
        'all.</p><ul><li>Unblocking it: ' + m.choke.relief + '.</li>';
      const nxt = m.order.filter(x => isFinite(x.util))[0];
      if (nxt) html += '<li>Once it has any capacity, <strong>' + nxt.name + '</strong> is the next ' +
        'limit at ' + pct(nxt.util) + '.</li>';
      html += '</ul></div>';
      box.innerHTML = html;
      return;
    }

    if (m.over){
      html += '<p>Every other stage could go faster and it would change nothing. At these settings the ' +
        'programme screens <strong>' + fmt(m.achievable) + ' people a year</strong> against a target of ' +
        fmt(m.v.target) + ' &mdash; ' + pct(m.achievable/m.v.target) + ' of it &mdash; and takes about ' +
        years.toFixed(1) + '&times; as long as planned to get through the population once.</p>';
      html += '<ul>';
      html += '<li><strong>' + fmt(m.shortfallPerDay) + ' patients a day</strong> are not screened and join a queue.</li>';
      html += '<li>Unblocking it: ' + m.choke.relief + '.</li>';
      const next = m.order[1];
      if (next) html += '<li>Do that and <strong>' + next.name + '</strong> becomes the limit at ' +
        pct(next.util) + ', capping the programme at ' + fmt(next.ceiling) + ' people/year.</li>';
      html += '</ul>';
    } else {
      html += '<p>No stage is over capacity: the target of ' + fmt(m.v.target) +
        ' people a year is reachable as configured. ' + m.choke.name + ' is the stage with the least ' +
        'headroom, so it is the one that will bind first as the programme grows.</p><ul>';
      html += '<li>Room to grow before it binds: <strong>' + fmt(m.choke.ceiling) +
        ' people/year</strong> (' + ((m.choke.ceiling/m.v.target)).toFixed(2) + '&times; the current target).</li>';
      if (m.choke.util > 0.75) html += '<li>It is already above 75%, where a steady-state model like ' +
        'this one starts to flatter reality &mdash; real arrivals bunch, so expect a queue here well ' +
        'before the figure reaches 100%.</li>';
      html += '<li>When it does bind: ' + m.choke.relief + '.</li>';
      html += '</ul>';
    }
    html += '</div>';
    box.innerHTML = html;
  }

  function renderTable(m){
    const rows = m.stages.map(s => {
      const isChoke = (s === m.choke);
      const c = utilColour(s.util);
      const chip = isFinite(s.capacity)
        ? '<span class="util-chip" style="background:' + tint(c, 0.55) + ';color:#111">' + pct(s.util) + '</span>'
        : '<span class="small">n/a</span>';
      return '<tr class="' + (isChoke ? "choke" : "") + '">' +
        '<td>' + s.name + (isChoke ? ' <strong>&mdash; choke point</strong>' : '') +
          '<br><span class="small">' + s.sub + '</span></td>' +
        '<td class="num">' + fmt(s.demand) + '<br><span class="small">' + s.demandNote + '</span></td>' +
        '<td class="num">' + (isFinite(s.capacity) ? fmt(s.capacity) : "—") +
          '<br><span class="small">' + s.capacityNote + '</span></td>' +
        '<td class="num">' + chip + '</td>' +
        '<td class="num">' + (isFinite(s.ceiling) ? fmt(s.ceiling) : "—") + '</td>' +
        '</tr>';
    }).join("");

    $("stage-table").innerHTML =
      '<thead><tr><th>Stage</th><th>Demand</th><th>Capacity</th><th>Utilisation</th>' +
      '<th>Ceiling it sets<br><span class="small">people/year</span></th></tr></thead><tbody>' +
      rows + '</tbody>';
  }

  function renderRelief(m){
    $("relief").innerHTML = m.order.map((s, i) => {
      const c = utilColour(s.util);
      return '<li><span class="relief-rank" style="background:' + tint(c, 0.6) + '">' + (i+1) + '</span>' +
        '<span><strong>' + s.name + '</strong> — ' + pct(s.util) + ' used, ceiling ' +
        (isFinite(s.ceiling) ? fmt(s.ceiling) + ' people/year' : 'none') +
        '. <span class="small">' + s.relief + '</span></span></li>';
    }).join("");
  }

  // ---------------------------------------------------------------------
  // BACKLOG CHART
  // ---------------------------------------------------------------------
  function renderBacklog(m){
    const svg = $("backlog");
    Array.prototype.slice.call(svg.childNodes).forEach(n => {
      if (n.nodeName !== "title" && n.nodeName !== "desc") svg.removeChild(n);
    });

    const W = 1120, H = 260, padL = 74, padR = 20, padT = 22, padB = 34;
    const months = 12;
    const perMonth = m.shortfallPerDay * 30.44;
    const series = [];
    for (let i=0;i<=months;i++) series.push(perMonth*i);
    const peak = series[months];
    const c = utilColour(m.choke.util);

    const x = i => padL + (W-padL-padR)*(i/months);
    const y = v => (H-padB) - (H-padT-padB)*(peak > 0 ? v/peak : 0);

    const g = el("g", {});
    svg.appendChild(g);

    if (peak <= 0){
      g.appendChild(el("text", { x: W/2, y: H/2, "text-anchor": "middle",
        "font-size": "14", "font-weight": "700", fill: rgb(c) },
        "No backlog — capacity meets demand at every stage."));
      g.appendChild(el("text", { x: W/2, y: H/2 + 22, "text-anchor": "middle" },
        m.choke.name + " is the tightest stage at " + pct(m.choke.util) + " of capacity, but it keeps up."));
      return;
    }

    // gridlines and y labels
    const ticks = 4;
    for (let t=0;t<=ticks;t++){
      const v = peak*t/ticks;
      g.appendChild(el("line", { x1: padL, x2: W-padR, y1: y(v), y2: y(v),
        class: t === 0 ? "zero" : "axis" }));
      g.appendChild(el("text", { x: padL-8, y: y(v)+3.5, "text-anchor": "end" }, fmt(v)));
    }
    for (let i=0;i<=months;i+=2){
      g.appendChild(el("text", { x: x(i), y: H-padB+16, "text-anchor": "middle" }, "m" + i));
    }

    const grad = el("linearGradient", { id: "bkfill", x1: "0", y1: "0", x2: "0", y2: "1" });
    grad.appendChild(el("stop", { offset: "0%", "stop-color": rgb(c), "stop-opacity": "0.75" }));
    grad.appendChild(el("stop", { offset: "100%", "stop-color": rgb(c), "stop-opacity": "0.10" }));
    const defs = el("defs", {}); defs.appendChild(grad); svg.insertBefore(defs, g);

    let d = "M" + x(0) + " " + y(0);
    for (let i=1;i<=months;i++) d += " L" + x(i) + " " + y(series[i]);
    g.appendChild(el("path", { d: d + " L" + x(months) + " " + y(0) + " Z", fill: "url(#bkfill)" }));
    g.appendChild(el("path", { d, fill: "none", stroke: rgb(c), "stroke-width": 2.5 }));

    for (let i=1;i<=months;i++){
      g.appendChild(el("circle", { cx: x(i), cy: y(series[i]), r: 3, fill: rgb(c) }));
    }
    g.appendChild(el("text", { x: x(months)-6, y: y(peak)-9, "text-anchor": "end",
      "font-size": "12.5", "font-weight": "700", fill: rgb(c) },
      fmt(peak) + " patients waiting after 12 months"));
    g.appendChild(el("text", { x: padL, y: padT-6, "font-size": "10.5" },
      "Queue behind " + m.choke.name + ", in patients not yet screened"));
  }

  // ---------------------------------------------------------------------
  // WIRING
  // ---------------------------------------------------------------------
  function update(){
    const m = solve(readInputs());
    renderFlow(m);
    renderVerdict(m);
    renderTable(m);
    renderRelief(m);
    renderBacklog(m);
  }

  window.addEventListener("DOMContentLoaded", () => {
    Object.keys(FIELDS).forEach(k => {
      const input = $(FIELDS[k]);
      input.addEventListener("input", update);
      input.addEventListener("change", update);
    });
    $("p-reset").addEventListener("click", () => {
      Object.keys(FIELDS).forEach(k => { $(FIELDS[k]).value = DEFAULTS[k]; });
      update();
    });
    update();
  });

})();
