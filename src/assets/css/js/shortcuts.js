document.addEventListener('keydown', function(event) {
    switch(event.key) {
        case 'ArrowLeft':
            document.getElementById('previous-point').click();
            break;
        case 'ArrowRight':
            document.getElementById('next-point').click();
            break;
        case ' ':
            event.preventDefault();  // Prevent the default action of the spacebar
            document.getElementById('play-audio').click();
            break;
        default:
            break;
    }
});