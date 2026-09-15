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
