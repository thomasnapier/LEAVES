window.addEventListener('load', function () {
    var progressBar = document.getElementById('progress-bar');
    var renderer = window.dashRenderer;

    if (renderer) {
        renderer.on('request', function () {
            progressBar.style.width = '50%';  // Example: Update width on request
        });

        renderer.on('response', function () {
            progressBar.style.width = '100%';  // Example: Set to 100% on response
        });
    }
});