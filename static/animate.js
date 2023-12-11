const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("show");
      }
    });
  }, {
    root: null,
    rootMargin: "0px",
    threshold: 0.1,
  });
  
  const hiddenLeftElements = document.querySelectorAll(".hidden-left");
  const hiddenRightElements = document.querySelectorAll(".hidden-right");
  const hiddenTopElements = document.querySelectorAll(".hidden-top");
  const hiddenBottomElements = document.querySelectorAll(".hidden-bottom");
  const hiddenStaticElements = document.querySelectorAll(".hidden-static");
  const hiddenHoverElements = document.querySelectorAll(".hidden-hover");
  
  hiddenLeftElements.forEach((el) => observer.observe(el));
  hiddenRightElements.forEach((el) => observer.observe(el));
  hiddenTopElements.forEach((el) => observer.observe(el));
  hiddenBottomElements.forEach((el) => observer.observe(el));
  hiddenStaticElements.forEach((el) => observer.observe(el));
  hiddenHoverElements.forEach((el) => observer.observe(el));
  