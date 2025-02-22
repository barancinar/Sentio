document.addEventListener('DOMContentLoaded', () => {
    const pupils = document.querySelectorAll('.pupil');
    const maxMovement = 5; // 3'ten 5'e çıkardık - daha geniş hareket alanı

    document.addEventListener('mousemove', (e) => {
        requestAnimationFrame(() => { // Performance için requestAnimationFrame ekledik
            const mouseX = e.clientX;
            const mouseY = e.clientY;

            pupils.forEach(pupil => {
                const pupilRect = pupil.getBoundingClientRect();
                const centerX = pupilRect.left + (pupilRect.width / 2);
                const centerY = pupilRect.top + (pupilRect.height / 2);

                const angle = Math.atan2(mouseY - centerY, mouseX - centerX);
                const movementX = Math.cos(angle) * maxMovement;
                const movementY = Math.sin(angle) * maxMovement;

                pupil.style.transform = `translate(${movementX}px, ${movementY}px)`;
            });
        });
    });
});

/* filepath: /Users/barancinar/Desktop/EmotionDetection/static/assets/js/style.js */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        const headerOffset = 80;
        const elementPosition = target.getBoundingClientRect().top;
        const offsetPosition = elementPosition - headerOffset;

        window.scrollBy({
            top: offsetPosition,
            behavior: 'smooth'
        });
    });
});