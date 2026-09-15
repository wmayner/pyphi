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
