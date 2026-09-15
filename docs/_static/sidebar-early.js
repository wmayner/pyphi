// Runs before the body renders: if the reader collapsed the sidebar, mark the
// document so the stylesheet draws it collapsed from the first paint. site.js
// completes the restore once the page has loaded.
try {
  if (localStorage.getItem("pp-sidebar-collapsed") === "1") {
    document.documentElement.classList.add("pp-sidebar-collapsed");
  }
} catch (error) {}
