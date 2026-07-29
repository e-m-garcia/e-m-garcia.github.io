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

(() => {
  const toc = document.querySelector("[data-article-toc]");
  const article = document.querySelector(".research-article--research .prose--article");

  if (!toc || !article) return;

  const headings = Array.from(article.querySelectorAll("h2"));
  const tocContainer = toc.closest(".article-toc");

  if (headings.length === 0) {
    if (tocContainer) tocContainer.hidden = true;
    return;
  }

  const usedIds = new Set();
  const slugify = (text) =>
    text
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9\s-]/g, "")
      .replace(/\s+/g, "-")
      .replace(/-+/g, "-");

  headings.forEach((heading, index) => {
    let id = heading.id || slugify(heading.textContent) || `section-${index + 1}`;
    const baseId = id;
    let suffix = 2;

    while (usedIds.has(id) || (document.getElementById(id) && document.getElementById(id) !== heading)) {
      id = `${baseId}-${suffix}`;
      suffix += 1;
    }

    heading.id = id;
    usedIds.add(id);
  });

  const list = document.createElement("ol");
  list.className = "article-toc__list";

  const links = headings.map((heading) => {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.className = "article-toc__link";
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent;
    item.appendChild(link);
    list.appendChild(item);
    return link;
  });

  toc.appendChild(list);

  let activeId = "";
  const setActive = (id) => {
    if (!id || id === activeId) return;
    activeId = id;

    links.forEach((link) => {
      const isActive = link.hash === `#${id}`;
      link.classList.toggle("is-active", isActive);

      if (isActive) {
        link.setAttribute("aria-current", "location");
      } else {
        link.removeAttribute("aria-current");
      }
    });
  };

  let framePending = false;
  const updateActiveSection = () => {
    framePending = false;
    const threshold = Math.max(110, window.innerHeight * 0.2);
    let current = headings[0];

    headings.forEach((heading) => {
      if (heading.getBoundingClientRect().top <= threshold) current = heading;
    });

    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
      current = headings[headings.length - 1];
    }

    setActive(current.id);
  };

  const requestUpdate = () => {
    if (framePending) return;
    framePending = true;
    window.requestAnimationFrame(updateActiveSection);
  };

  window.addEventListener("scroll", requestUpdate, { passive: true });
  window.addEventListener("resize", requestUpdate);
  window.addEventListener("hashchange", requestUpdate);
  updateActiveSection();
})();
