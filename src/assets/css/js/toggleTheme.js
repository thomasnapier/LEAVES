document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded and parsed");

    setTimeout(function() {
        const themeToggleButton = document.getElementById('theme-toggle');
        if (!themeToggleButton) {
            console.error("Theme toggle button not found!");
            return;
        }

        const savedTheme = localStorage.getItem('theme') || 'dark';
        document.documentElement.classList.add(savedTheme);
        console.log(`Initial theme set to: ${savedTheme}`);

        themeToggleButton.addEventListener('click', function() {
            const currentTheme = document.documentElement.classList.contains('dark') ? 'dark' : 'light';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.documentElement.classList.remove(currentTheme);
            document.documentElement.classList.add(newTheme);
            localStorage.setItem('theme', newTheme);
            console.log(`Theme toggled to: ${newTheme}`);
        });
    }, 500);
});
