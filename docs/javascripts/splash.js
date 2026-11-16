// Pause the splash hero's animation when it is off-screen or the tab is hidden,
// and replay the black-hole transition after each instant navigation.
// Works with Material's instant navigation via the document$ observable.
(function () {
  function wire() {
    var hero = document.querySelector("[data-st-hero]");
    if (!hero || hero.dataset.stWired) return;
    hero.dataset.stWired = "1";
    var scene = hero.querySelector(".st-scene");
    var ship = hero.querySelector(".st-cruiser");
    scene.addEventListener("click", function () { ship.classList.add("is-jet"); });
    ship.addEventListener("animationend", function (e) {
      if (e.animationName === "st-jet") ship.classList.remove("is-jet");
    });
    var visible = true;
    function sync() {
      hero.classList.toggle("is-paused", !visible || document.hidden);
    }
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
        sync();
      }).observe(hero);
    }
    document.addEventListener("visibilitychange", sync);
  }
  // Guide level select: one ship that flies tile to tile. Turning around is a half loop:
  // up and over when the next tile is level or higher, down and under when it is lower.
  function wireLevels() {
    var nav = document.querySelector(".st-levels");
    if (!nav || nav.dataset.stWired) return;
    nav.dataset.stWired = "1";
    var ship = document.createElement("span");
    ship.className = "st-lv-ship";
    // reuse the hero's chunk exhaust (splash.css)
    ship.innerHTML = '<div class="st-exhaust">' + "<i></i>".repeat(7) + "</div>";
    nav.appendChild(ship);
    var still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var LOOP_MS = 1000; // time spent turning around
    var at = null;
    var face = 1; // 1 = right, -1 = left
    var base = 0; // level heading in deg; kept unwrapped so rotations never spin the long way
    var anim = null;
    function wrap(x) { return ((x + 540) % 360) - 180; }
    function set(p, rot, sy) {
      ship.style.translate = p[0] + "px " + p[1] + "px";
      ship.style.rotate = rot + "deg";
      ship.style.scale = "1 " + sy;
    }
    function flight(from, to, f) {
      var pts = []; // {x, y, rot, sy, ms}: ms = time since the previous point
      var turn = f !== face;
      var d = face, i, t, th;
      var newBase = base;
      if (turn) {
        var s = to[1] > from[1] + 20 ? 1 : -1; // 1 = loop down, -1 = loop up
        // size the loop to land on the target's row, so the leg after it stays level
        var R = Math.max(26, Math.abs(to[1] - from[1]) / 2);
        for (i = 0; i <= 40; i++) {
          th = Math.PI * i / 40;
          pts.push({
            x: from[0] + d * R * Math.sin(th),
            y: from[1] + s * R * (1 - Math.cos(th)),
            rot: base + s * d * th * 180 / Math.PI,
            sy: face,
            ms: i ? LOOP_MS / 40 : 0,
          });
        }
        newBase = base + s * d * 180;
        from = [from[0], from[1] + 2 * s * R];
      }
      // cubic leg: leave level (tangent to the loop, or a gentle hop) and arrive level
      var len = Math.hypot(to[0] - from[0], to[1] - from[1]);
      var p1, p2;
      if (turn) {
        var k = 0.35 * len;
        p1 = [from[0] - d * k, from[1]];
        p2 = [to[0] - f * k, to[1]];
      } else {
        var c = [(from[0] + to[0]) / 2, Math.min(from[1], to[1]) - 30];
        p1 = [from[0] + 2 / 3 * (c[0] - from[0]), from[1] + 2 / 3 * (c[1] - from[1])];
        p2 = [to[0] + 2 / 3 * (c[0] - to[0]), to[1] + 2 / 3 * (c[1] - to[1])];
      }
      var legMs = Math.max(500, Math.min(1600, 400 + len * 2));
      for (i = turn ? 1 : 0; i <= 40; i++) {
        t = i / 40;
        var u = 1 - t;
        var dx = 3 * u * u * (p1[0] - from[0]) + 6 * u * t * (p2[0] - p1[0]) + 3 * t * t * (to[0] - p2[0]);
        var dy = 3 * u * u * (p1[1] - from[1]) + 6 * u * t * (p2[1] - p1[1]) + 3 * t * t * (to[1] - p2[1]);
        var off = wrap(Math.atan2(dy, dx) * 180 / Math.PI - newBase);
        // nose follows the path, easing back to level by the end
        var pitch = Math.max(-45, Math.min(45, off)) * (turn ? Math.min(1, u * 3) : Math.sin(Math.PI * t));
        // roll upright over the first half of the leg after a loop
        var r = turn ? Math.min(1, t / 0.5) : 1;
        pts.push({
          x: u * u * u * from[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * to[0],
          y: u * u * u * from[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * to[1],
          rot: newBase + pitch,
          sy: face + (f - face) * (0.5 - 0.5 * Math.cos(Math.PI * r)),
          ms: i ? legMs / 40 : 0,
        });
      }
      var total = 0, acc = 0;
      pts.forEach(function (p) { total += p.ms; });
      var frames = pts.map(function (p) {
        acc += p.ms;
        return { translate: p.x + "px " + p.y + "px", rotate: p.rot + "deg", scale: "1 " + p.sy, offset: acc / total };
      });
      base = newBase;
      ship.classList.add("is-moving");
      var a = ship.animate(frames, { duration: total, easing: "ease-in-out" });
      a.onfinish = function () { anim = null; ship.classList.remove("is-moving"); };
      return a;
    }
    function fly(a) {
      ship.classList.remove("is-warp");
      ship.classList.add("is-on");
      var y = a.offsetTop + a.offsetHeight / 2 - 7;
      var f = at === null ? 1 : (a.offsetLeft + a.offsetWidth / 2 > at[0] + 20 ? 1 : -1);
      // park on the near side of the tile, nose pointing at it
      var to = [f > 0 ? a.offsetLeft - 14 : a.offsetLeft + a.offsetWidth - 26, y];
      if (anim) {
        var cur = getComputedStyle(ship).translate.split(" ").map(parseFloat);
        if (cur.length === 2) at = cur;
        anim.cancel();
        ship.classList.remove("is-moving");
      }
      if (at !== null && !still) anim = flight(at, to, f);
      else base = f > 0 ? 0 : -180;
      set(to, base, f);
      at = to;
      face = f;
    }
    nav.querySelectorAll("a").forEach(function (a) {
      a.addEventListener("mouseenter", function () { fly(a); });
      a.addEventListener("focus", function () { fly(a); });
      a.addEventListener("mousedown", function () { ship.classList.add("is-warp"); });
    });
    nav.addEventListener("mouseleave", function () { ship.classList.remove("is-on"); });
  }
  if (window.document$ && typeof window.document$.subscribe === "function") {
    var first = true;
    window.document$.subscribe(function () {
      wire();
      wireLevels();
      var content = document.body;
      if (!first && content) {
        content.classList.remove("st-enter");
        void content.offsetWidth; // restart the animation
        content.classList.add("st-enter");
      }
      first = false;
    });
  } else {
    document.addEventListener("DOMContentLoaded", function () { wire(); wireLevels(); });
  }
})();
