// Ctrl+B (Cmd+B on macOS) toggles the primary sidebar, like an editor.
document.addEventListener("keydown", (event) => {
  if (!(event.ctrlKey || event.metaKey) || event.altKey || event.key.toLowerCase() !== "b") return;
  const target = event.target;
  if (target && (target.isContentEditable || /^(input|textarea|select)$/i.test(target.tagName))) return;
  const collapse = document.querySelector("#pst-collapse-sidebar-button");
  const drawer = document.querySelector(".bd-header .primary-toggle");
  const button = collapse && collapse.offsetParent !== null ? collapse : drawer;
  if (!button) return;
  event.preventDefault();
  button.click();
});
document.addEventListener("DOMContentLoaded", () => {
  const collapse = document.querySelector("#pst-collapse-sidebar-button");
  if (collapse) collapse.title = "Toggle the sidebar (Ctrl+B)";
});

// The theme pins the primary sidebar below a fixed header height and sizes
// it to the rest of the viewport. This header is taller than that, and the
// announcement bar sits above it until scrolled away, so measure where the
// header actually ends and let the stylesheet place the sidebar under it.
(() => {
  const root = document.documentElement;
  const update = () => {
    const header = document.querySelector(".bd-header");
    if (!header) return;
    const bottom = Math.max(0, Math.round(header.getBoundingClientRect().bottom));
    root.style.setProperty("--pp-sidebar-top", `${bottom}px`);
  };
  document.addEventListener("DOMContentLoaded", update);
  window.addEventListener("resize", update);
  window.addEventListener("scroll", update, { passive: true });
})();

// Remember whether the reader collapsed the sidebar, and restore that on the
// next page. The theme's own handler toggles the collapse; this only records
// the choice and replays it once the handler is installed.
(() => {
  const KEY = "pp-sidebar-collapsed";
  const read = () => { try { return localStorage.getItem(KEY) === "1"; } catch { return false; } };
  const write = (collapsed) => { try { localStorage.setItem(KEY, collapsed ? "1" : "0"); } catch {} };
  window.addEventListener("load", () => {
    const button = document.querySelector("#pst-collapse-sidebar-button");
    const sidebar = document.querySelector(".bd-sidebar-primary");
    if (!button || !sidebar) return;
    // Capture phase: runs before the theme's handler, so the attribute still
    // shows the state being left.
    button.addEventListener("click", () => write(button.getAttribute("aria-expanded") !== "false"), true);
    if (read() && button.getAttribute("aria-expanded") !== "false") {
      // Collapse the way the theme does under prefers-reduced-motion: pin
      // each item's width, add the class, and mark the button collapsed.
      // Its later expand click then behaves as on any collapsed sidebar.
      Array.from(sidebar.children).forEach((child) => {
        child.style.width = `${child.getBoundingClientRect().width}px`;
      });
      sidebar.style.transition = "none";
      sidebar.classList.add("pst-squeeze");
      button.setAttribute("aria-expanded", "false");
      button.dataset.busy = "false";
      requestAnimationFrame(() => requestAnimationFrame(() => { sidebar.style.transition = ""; }));
    }
  });
})();
