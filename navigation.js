// Wait until the DOM is fully loaded
document.addEventListener("DOMContentLoaded", function () {
    // Scroll to sections when clicking on anchor links
    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
        anchor.addEventListener("click", function (e) {
            e.preventDefault();
            const navHeight = document.querySelector("nav").offsetHeight; // Dynamic height of the navigation bar
            const targetElement = document.querySelector(this.getAttribute("href"));
            const offsetTop = targetElement.offsetTop;

            window.scrollTo({
                top: offsetTop - navHeight * 1.5, // Use dynamic height for the offset
                behavior: "smooth",
            });
        });
    });
});