(() => {
  const root = document.documentElement;
  const toggle = document.querySelector("[data-theme-toggle]");
  const themeColor = document.querySelector('meta[name="theme-color"]');

  if (!toggle) return;

  const applyTheme = (isDark) => {
    if (isDark) {
      root.dataset.theme = "dark";
    } else {
      root.removeAttribute("data-theme");
    }

    toggle.setAttribute("aria-checked", String(isDark));
    toggle.setAttribute("aria-label", isDark ? "Use light mode" : "Use dark mode");
    toggle.title = isDark ? "Use light mode" : "Use dark mode";

    if (themeColor) {
      themeColor.content = isDark ? "#101210" : "#f7f7f5";
    }
  };

  applyTheme(root.dataset.theme === "dark");

  toggle.addEventListener("click", () => {
    const isDark = root.dataset.theme !== "dark";
    applyTheme(isDark);

    try {
      localStorage.setItem("color-theme", isDark ? "dark" : "light");
    } catch (error) {}
  });
})();
